import hashlib

from parser.ranking_parser import RankingParser
from memory.lesson_rule import build_retrieval_profile
from prompt.builder import DEFAULT_STAGE2_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
from prompt.repair_prompt_builder import RepairPromptBuilder
from prompt.ranking_prompt_builder import RankingPromptBuilder
from prompt.stage1_prompt_builder import Stage1PromptBuilder
from prompt.stage2_prompt_builder import Stage2PromptBuilder
from .exceptions import AgentsException
from .slm_agent import SLM_4b_Agent


class AgentNeuronHelper:
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    DEFAULT_STAGE2_SYSTEM_PROMPT = DEFAULT_STAGE2_SYSTEM_PROMPT

    def __init__(self, system_prompt=None, stage2_system_prompt=None):
        self.SYSTEM_PROMPT = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.STAGE2_SYSTEM_PROMPT = stage2_system_prompt or self.DEFAULT_STAGE2_SYSTEM_PROMPT
        self.stage1_prompt_builder = Stage1PromptBuilder()
        self.stage2_prompt_builder = Stage2PromptBuilder()
        self.repair_prompt_builder = RepairPromptBuilder()
        self.ranking_prompt_builder = RankingPromptBuilder()
        self.ranking_parser = RankingParser()

    def build_stage1_message(
        self,
        question,
        formers,
        tool_context: str = "",
        reflection_context: str = "",
    ):
        return self.stage1_prompt_builder.build(
            question=question,
            formers=formers,
            tool_context=tool_context,
            reflection_context=reflection_context,
        )

    def build_stage1_reflection_context(self, question, memory_tool, runtime=None, limit: int = 2):
        """Build stage1 round0 memory guidance from Query/Task + Insight graphs."""
        normalized_question = str(question or "").strip()
        if runtime is None or not normalized_question:
            return ""

        query_graph = getattr(runtime, "query_task_graph", None)
        insight_graph = getattr(runtime, "insight_graph", None)
        if query_graph is None or insight_graph is None:
            return ""

        try:
            task_id = self._resolve_stage1_task_id(runtime, normalized_question)
            attachment_type = self._resolve_attachment_type(runtime)

            query_graph.register_task(
                task_id,
                normalized_question,
                metadata={
                    "source": "stage1_round0",
                    "attachment_type": attachment_type,
                },
            )
            classification = query_graph.classify_task(
                normalized_question,
                attachment_type=attachment_type,
            )
            query_graph.link_task_signals(task_id, classification)
            retrieval = query_graph.retrieve_for_stage1_round0(
                task_id,
                normalized_question,
                limit=max(limit, 3),
            )
            insights = insight_graph.retrieve_insights(
                task_type=retrieval.get("task_type", "general_reasoning"),
                trigger_terms=retrieval.get("trigger_terms", []),
                failure_modes=retrieval.get("failure_modes", []),
                limit=max(limit, 3),
            )
            guidance = query_graph.build_stage1_guidance_prompt(
                retrieval,
                insights=insights,
                max_failures=1,
            )
            runtime.record_memory_read(
                {
                    "stage": "stage1_round0",
                    "source": "query_task_graph+insight_graph",
                    "task_id": task_id,
                    "task_type": retrieval.get("task_type"),
                    "trigger_terms": retrieval.get("trigger_terms", []),
                    "insight_ids": [
                        item.get("insight_id")
                        for item in insights
                        if isinstance(item, dict) and item.get("insight_id")
                    ],
                }
            )
            return guidance
        except Exception as exc:
            print(f"[WARN] stage1 graph memory guidance failed: {exc}")
            return ""

    def _resolve_stage1_task_id(self, runtime, question: str) -> str:
        context = getattr(runtime, "current_context", {}) or {}
        for key in ("task_id", "id", "sample_id"):
            value = str(context.get(key, "") or "").strip()
            if value:
                return value

        digest = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"gaia_task_{digest}"

    def _resolve_attachment_type(self, runtime) -> str | None:
        attachment = getattr(runtime, "current_attachment", None) or {}
        for key in ("extension", "file_extension", "type"):
            value = str(attachment.get(key, "") or "").strip().lower().lstrip(".")
            if value:
                return value
        path = str(attachment.get("path", "") or attachment.get("file_path", "") or "").strip()
        if "." in path:
            return path.rsplit(".", 1)[-1].lower()
        return None

    def _summarize_reflection_content(self, content: str) -> str:
        normalized = " ".join(str(content or "").split()).strip()
        if not normalized:
            return ""

        import re

        lesson_match = re.search(
            r"Lesson:\s*(.+?)(?:\s+Tags:|\s+Applicability:|$)",
            normalized,
            flags=re.IGNORECASE,
        )
        error_type_match = re.search(
            r"Error type:\s*([A-Za-z0-9_\-]+)",
            normalized,
            flags=re.IGNORECASE,
        )
        applicability_match = re.search(
            r"Applicability:\s*(.+?)(?:\s+Observed mismatch:|$)",
            normalized,
            flags=re.IGNORECASE,
        )
        if lesson_match:
            parts = []
            if error_type_match:
                parts.append(f"error_type={error_type_match.group(1).strip()}")
            parts.append(f"lesson={lesson_match.group(1).strip()}")
            if applicability_match:
                parts.append(f"use_when={applicability_match.group(1).strip()}")
            return " | ".join(parts)

        if len(normalized) > 220:
            return normalized[:217].rstrip() + "..."
        return normalized

    def _build_memory_profile(self, question: str) -> dict:
        return build_retrieval_profile(question)

    def build_repair_prompt(self, expected_weight_count):
        return self.repair_prompt_builder.build(
            expected_weight_count=expected_weight_count,
        )

    def build_stage2_prompts(self, question, stage1_result, importance, tool_context):
        return self.stage2_prompt_builder.build(
            question=question,
            stage1_result=stage1_result,
            importance=importance,
            tool_context=tool_context,
            system_prompt=self.STAGE2_SYSTEM_PROMPT,
        )

    def build_ranking_message(self, responses, question):
        return self.ranking_prompt_builder.build(
            responses=responses,
            question=question,
        )

    def generate_answer(self, predecessors_answers, model_name):
        try:
            agent = SLM_4b_Agent(model_name=model_name)
            response = agent.think(predecessors_answers)

            content = response.choices[0].message.content
            prompt_tokens = (
                response.usage.prompt_tokens
                if response.usage and hasattr(response.usage, "prompt_tokens")
                else 0
            )
            completion_tokens = (
                response.usage.completion_tokens
                if response.usage and hasattr(response.usage, "completion_tokens")
                else 0
            )

            return content, prompt_tokens, completion_tokens

        except Exception as e:
            print(f"[ERROR] SLM API failed: {e}")
            raise AgentsException(f"SLM API failed: {str(e)}")

    def listwise_ranker_2(self, responses, question, model_name):
        assert 2 <= len(responses)
        message = self.build_ranking_message(responses, question)
        completion, prompt_tokens, completion_tokens = self.generate_answer(
            [message], model_name
        )
        return (
            self.ranking_parser.parse(completion, max_num=len(responses)),
            prompt_tokens,
            completion_tokens,
        )
