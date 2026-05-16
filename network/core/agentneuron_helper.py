import hashlib

from parser.ranking_parser import RankingParser
from prompt.builder import DEFAULT_STAGE2_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
from prompt.repair_prompt_builder import RepairPromptBuilder
from prompt.ranking_prompt_builder import RankingPromptBuilder
from prompt.stage1_prompt_builder import Stage1PromptBuilder
from prompt.stage2_prompt_builder import Stage2PromptBuilder
from ..exceptions import AgentsException
from ..slm_agent import SLM_Agent, estimate_chat_tokens, estimate_text_tokens
from .task_context import TaskContext


class AgentNeuronHelper:
    """AgentNeuronHelper 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    DEFAULT_STAGE2_SYSTEM_PROMPT = DEFAULT_STAGE2_SYSTEM_PROMPT

    def __init__(self, system_prompt=None, stage2_system_prompt=None):
        """初始化 AgentNeuronHelper 實例。
        
        參數:
            system_prompt: 此流程需要使用的輸入資料。
            stage2_system_prompt: 此流程需要使用的輸入資料。
        """
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
        contract=None,
        task_context=None,
    ):
        """建立 build_stage1_message 所需的資料或輸出。
        
        參數:
            question: 此流程需要使用的輸入資料。
            formers: 此流程需要使用的輸入資料。
            tool_context: 此流程需要使用的輸入資料。
            reflection_context: 此流程需要使用的輸入資料。
            contract: 此流程需要使用的輸入資料。
            task_context: 此流程需要使用的輸入資料。
        """
        return self.stage1_prompt_builder.build(
            question=question,
            formers=formers,
            tool_context=tool_context,
            reflection_context=reflection_context,
            contract=contract,
            task_context=task_context,
        )

    def build_stage1_reflection_context(self, question, memory_tool, runtime=None, limit: int = 2):
        """建立 build_stage1_reflection_context 所需的資料或輸出。
        
        參數:
            question: 此流程需要使用的輸入資料。
            memory_tool: 此流程需要使用的輸入資料。
            runtime: 此流程需要使用的輸入資料。
            limit: 此流程需要使用的輸入資料。
        """
        normalized_question = str(question or "").strip()
        if runtime is None or not normalized_question:
            return ""

        memory_service = getattr(runtime, "memory_service", None)
        if memory_service is None:
            return ""

        try:
            task_id = self._resolve_stage1_task_id(runtime, normalized_question)
            task_context = runtime.get_task_context() if hasattr(runtime, "get_task_context") else TaskContext.from_dict(getattr(runtime, "current_context", {}) or {})
            source = task_context.source_label
            result = memory_service.retrieve_context(
                question=normalized_question,
                stage="stage1_round0",
                injection_target="stage1_round0",
                source=source,
                task_id=task_id,
                limit=max(limit, 3),
            )
            if result is None:
                return ""
            guidance = str(result.get("guidance", "") or "")
            return guidance
        except Exception as exc:
            print(f"[WARN] stage1 graph memory guidance failed: {exc}")
            return ""

    def _resolve_stage1_task_id(self, runtime, question: str) -> str:
        """處理 resolve_stage1_task_id 流程並回傳結果。
        
        參數:
            runtime: 此流程需要使用的輸入資料。
            question: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        task_context = runtime.get_task_context() if hasattr(runtime, "get_task_context") else TaskContext.from_dict(getattr(runtime, "current_context", {}) or {})
        if task_context.task_id:
            return task_context.task_id

        digest = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"gaia_task_{digest}"

    def _resolve_attachment_type(self, runtime) -> str | None:
        """處理 resolve_attachment_type 流程並回傳結果。
        
        參數:
            runtime: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        if hasattr(runtime, "get_task_context"):
            task_context = runtime.get_task_context()
            if task_context.attachment_type:
                return task_context.attachment_type
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
        """處理 summarize_reflection_content 流程並回傳結果。
        
        參數:
            content: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
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

    def build_repair_prompt(self, expected_weight_count, contract=None, task_context=None, question: str = ""):
        """建立 build_repair_prompt 所需的資料或輸出。
        
        參數:
            expected_weight_count: 此流程需要使用的輸入資料。
            contract: 此流程需要使用的輸入資料。
            task_context: 此流程需要使用的輸入資料。
            question: 此流程需要使用的輸入資料。
        """
        return self.repair_prompt_builder.build(
            expected_weight_count=expected_weight_count,
            contract=contract,
            task_context=task_context,
            question=question,
        )

    def build_stage2_prompts(self, question, stage1_result, importance, tool_context, contract=None, task_context=None):
        """建立 build_stage2_prompts 所需的資料或輸出。
        
        參數:
            question: 此流程需要使用的輸入資料。
            stage1_result: 此流程需要使用的輸入資料。
            importance: 此流程需要使用的輸入資料。
            tool_context: 此流程需要使用的輸入資料。
            contract: 此流程需要使用的輸入資料。
            task_context: 此流程需要使用的輸入資料。
        """
        return self.stage2_prompt_builder.build(
            question=question,
            stage1_result=stage1_result,
            importance=importance,
            tool_context=tool_context,
            system_prompt=self.STAGE2_SYSTEM_PROMPT,
            contract=contract,
            task_context=task_context,
        )

    def build_ranking_message(self, responses, question):
        """建立 build_ranking_message 所需的資料或輸出。
        
        參數:
            responses: 此流程需要使用的輸入資料。
            question: 此流程需要使用的輸入資料。
        """
        return self.ranking_prompt_builder.build(
            responses=responses,
            question=question,
        )

    def generate_answer(self, predecessors_answers, model_name):
        """處理 generate_answer 流程並回傳結果。
        
        參數:
            predecessors_answers: 此流程需要使用的輸入資料。
            model_name: 此流程需要使用的輸入資料。
        """
        try:
            agent = SLM_Agent(model_name=model_name)
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
            if prompt_tokens <= 0:
                prompt_tokens = estimate_chat_tokens(predecessors_answers)
            if completion_tokens <= 0:
                completion_tokens = estimate_text_tokens(content)

            return content, prompt_tokens, completion_tokens

        except Exception as e:
            print(f"[ERROR] SLM API failed: {e}")
            raise AgentsException(f"SLM API failed: {str(e)}")

    def listwise_ranker_2(self, responses, question, model_name):
        """處理 listwise_ranker_2 流程並回傳結果。
        
        參數:
            responses: 此流程需要使用的輸入資料。
            question: 此流程需要使用的輸入資料。
            model_name: 此流程需要使用的輸入資料。
        """
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
