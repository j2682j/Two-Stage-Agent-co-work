from parser.ranking_parser import RankingParser
from memory.lesson_rule import (
    build_retrieval_profile,
    parse_semantic_lesson_memory,
    select_relevant_semantic_lessons,
)
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
        self.STAGE2_SYSTEM_PROMPT = (
            stage2_system_prompt or self.DEFAULT_STAGE2_SYSTEM_PROMPT
        )
        self.stage1_prompt_builder = Stage1PromptBuilder()
        self.stage2_prompt_builder = Stage2PromptBuilder()
        self.repair_prompt_builder = RepairPromptBuilder()
        self.ranking_prompt_builder = RankingPromptBuilder()
        self.ranking_parser = RankingParser()

    def build_stage1_message(self, question, formers, tool_context: str = "", reflection_context: str = ""):
        return self.stage1_prompt_builder.build(
            question=question,
            formers=formers,
            tool_context=tool_context,
            reflection_context=reflection_context,
        )

    def build_stage1_reflection_context(self, question, memory_tool, runtime=None, limit: int = 2):
        memory_manager = getattr(memory_tool, "memory_manager", None)
        normalized_question = str(question or "").strip()
        if memory_manager is None or not normalized_question:
            return ""

        profile = (
            runtime.build_memory_profile(normalized_question)
            if runtime is not None and hasattr(runtime, "build_memory_profile")
            else self._build_memory_profile(normalized_question)
        )
        lesson_queries = (
            profile.lesson_queries if hasattr(profile, "lesson_queries") else profile["lesson_queries"]
        )
        seen_ids = set()
        lesson_objects = []
        for query_text in lesson_queries:
            try:
                retrieved = memory_manager.retrieve_memories(
                    query=query_text,
                    memory_types=["semantic"],
                    limit=max(limit * 2, 4),
                    min_importance=0.0,
                )
            except Exception as exc:
                print(f"[WARN] stage1 reflection memory 檢索失敗: {exc}")
                continue

            for memory in retrieved:
                memory_id = str(getattr(memory, "id", "") or "")
                if memory_id and memory_id in seen_ids:
                    continue
                if memory_id:
                    seen_ids.add(memory_id)
                lesson = parse_semantic_lesson_memory(memory)
                if lesson is None:
                    continue
                lesson_objects.append(lesson)
                if len(lesson_objects) >= limit * 6:
                    break
            if len(lesson_objects) >= limit * 6:
                break

        selected = select_relevant_semantic_lessons(
            lessons=lesson_objects,
            profile=profile,
            min_score=1.5,
            limit=limit,
        )
        if not selected:
            return ""
        summaries = [
            lesson.to_summary().replace("| applicability=", "| use_when=")
            for lesson, _ in selected
        ]
        return "\n".join(f"- {lesson}" for lesson in summaries[:limit])

    def _summarize_reflection_content(self, content: str) -> str:
        normalized = " ".join(str(content or "").split()).strip()
        if not normalized:
            return ""

        import re

        lesson_match = re.search(r"Lesson:\s*(.+?)(?:\s+Tags:|\s+Applicability:|$)", normalized, flags=re.IGNORECASE)
        error_type_match = re.search(r"Error type:\s*([A-Za-z0-9_\-]+)", normalized, flags=re.IGNORECASE)
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
            print(f"[ERROR] 呼叫 SLM API 失敗: {e}")
            raise AgentsException(f"SLM 呼叫失敗: {str(e)}")

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
