import hashlib

from parser.ranking_parser import RankingParser
from prompt.builder import DEFAULT_STAGE2_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
from prompt.repair_prompt_builder import RepairPromptBuilder
from prompt.ranking_prompt_builder import RankingPromptBuilder
from prompt.stage1_prompt_builder import Stage1PromptBuilder
from prompt.stage2_prompt_builder import Stage2PromptBuilder
from .exceptions import AgentsException
from .slm_agent import SLM_4b_Agent, estimate_chat_tokens, estimate_text_tokens


class AgentNeuronHelper:
    """
    負責在 network.agentneuron_helper 中封裝 AgentNeuronHelper，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        system_prompt: 此流程需要使用的輸入資料。
        stage2_system_prompt: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    DEFAULT_STAGE2_SYSTEM_PROMPT = DEFAULT_STAGE2_SYSTEM_PROMPT

    def __init__(self, system_prompt=None, stage2_system_prompt=None):
        """
        負責執行 AgentNeuronHelper 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            system_prompt: 此流程需要使用的輸入資料。
            stage2_system_prompt: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    ):
        """
        負責執行 AgentNeuronHelper 中的 build_stage1_message 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            formers: 此流程需要使用的輸入資料。
            tool_context: 此流程需要使用的輸入資料。
            reflection_context: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.stage1_prompt_builder.build(
            question=question,
            formers=formers,
            tool_context=tool_context,
            reflection_context=reflection_context,
        )

    def build_stage1_reflection_context(self, question, memory_tool, runtime=None, limit: int = 2):
        """
        負責執行 AgentNeuronHelper 中的 build_stage1_reflection_context 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            memory_tool: 記憶系統提供的檢索結果、寫入資料或操作介面。
            runtime: 目前流程所需的上下文、狀態或附加資訊。
            limit: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        normalized_question = str(question or "").strip()
        if runtime is None or not normalized_question:
            return ""

        graph_memory = getattr(runtime, "graph_memory", None)
        if graph_memory is None:
            return ""

        try:
            task_id = self._resolve_stage1_task_id(runtime, normalized_question)
            attachment_type = self._resolve_attachment_type(runtime)
            context = getattr(runtime, "current_context", {}) or {}
            source = str(context.get("benchmark") or context.get("source") or "system").strip().lower() or "system"
            result = graph_memory.retrieve_context(
                task_id=task_id,
                input_text=normalized_question,
                source=source,
                attachment_type=attachment_type,
                limit=max(limit, 3),
                injection_target="stage1_round0",
            )
            guidance = str(result.get("guidance", "") or "")
            retrieval = result.get("retrieval", {}) or {}
            insights = result.get("insights", []) or []
            runtime.record_memory_read(
                {
                    "stage": "stage1_round0",
                    "source": "graph_memory",
                    "task_id": task_id,
                    "task_type": retrieval.get("task_type"),
                    "trigger_terms": retrieval.get("trigger_terms", []),
                    "related_task_ids": result.get("related_task_ids", []),
                    "insight_ids": [
                        item.get("insight_id")
                        for item in insights
                        if isinstance(item, dict) and item.get("insight_id")
                    ],
                    "seed_task_hits": result.get("seed_task_hits", []),
                    "expanded_task_hits": result.get("expanded_task_hits", []),
                }
            )
            return guidance
        except Exception as exc:
            print(f"[WARN] stage1 graph memory guidance failed: {exc}")
            return ""

    def _resolve_stage1_task_id(self, runtime, question: str) -> str:
        """
        負責執行 AgentNeuronHelper 中的 _resolve_stage1_task_id 流程，依照 AgentNeuronHelper 的流程需求處理 _resolve_stage1_task_id 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            runtime: 目前流程所需的上下文、狀態或附加資訊。
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        context = getattr(runtime, "current_context", {}) or {}
        for key in ("task_id", "id", "sample_id"):
            value = str(context.get(key, "") or "").strip()
            if value:
                return value

        digest = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"gaia_task_{digest}"

    def _resolve_attachment_type(self, runtime) -> str | None:
        """
        負責執行 AgentNeuronHelper 中的 _resolve_attachment_type 流程，依照 AgentNeuronHelper 的流程需求處理 _resolve_attachment_type 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            runtime: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 AgentNeuronHelper 中的 _summarize_reflection_content 流程，依照 AgentNeuronHelper 的流程需求處理 _summarize_reflection_content 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            content: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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

    def build_repair_prompt(self, expected_weight_count):
        """
        負責執行 AgentNeuronHelper 中的 build_repair_prompt 流程，組裝提示詞內容，將任務、記憶、證據或格式要求整理成模型可讀的輸入。
        
        Args:
            expected_weight_count: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.repair_prompt_builder.build(
            expected_weight_count=expected_weight_count,
        )

    def build_stage2_prompts(self, question, stage1_result, importance, tool_context):
        """
        負責執行 AgentNeuronHelper 中的 build_stage2_prompts 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            stage1_result: 評估、推理或工具執行後產生的結果與分數資料。
            importance: 此流程需要使用的輸入資料。
            tool_context: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.stage2_prompt_builder.build(
            question=question,
            stage1_result=stage1_result,
            importance=importance,
            tool_context=tool_context,
            system_prompt=self.STAGE2_SYSTEM_PROMPT,
        )

    def build_ranking_message(self, responses, question):
        """
        負責執行 AgentNeuronHelper 中的 build_ranking_message 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            responses: 此流程需要使用的輸入資料。
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.ranking_prompt_builder.build(
            responses=responses,
            question=question,
        )

    def generate_answer(self, predecessors_answers, model_name):
        """
        負責執行 AgentNeuronHelper 中的 generate_answer 流程，依照 AgentNeuronHelper 的流程需求處理 generate_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            predecessors_answers: 此流程需要使用的輸入資料。
            model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
            if prompt_tokens <= 0:
                prompt_tokens = estimate_chat_tokens(predecessors_answers)
            if completion_tokens <= 0:
                completion_tokens = estimate_text_tokens(content)

            return content, prompt_tokens, completion_tokens

        except Exception as e:
            print(f"[ERROR] SLM API failed: {e}")
            raise AgentsException(f"SLM API failed: {str(e)}")

    def listwise_ranker_2(self, responses, question, model_name):
        """
        負責執行 AgentNeuronHelper 中的 listwise_ranker_2 流程，依照 AgentNeuronHelper 的流程需求處理 listwise_ranker_2 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            responses: 此流程需要使用的輸入資料。
            question: 目前要處理的任務、問題或查詢文字。
            model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
