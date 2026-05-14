from __future__ import annotations

import os
from typing import Any

from memory.graph import NetworkMemory


class NetworkRuntime:
    """
    負責在 network.network_runtime 中封裝 NetworkRuntime，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        tool_manager: 此流程需要使用的輸入資料。
        memory_tool: 記憶系統提供的檢索結果、寫入資料或操作介面。
        memory_config: 記憶系統提供的檢索結果、寫入資料或操作介面。
        shared_memory_user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        enable_shared_memory: 控制是否啟用此項資料、功能或處理分支的布林開關。
        memory_mode: 記憶系統提供的檢索結果、寫入資料或操作介面。
        evidence_builder: 此流程需要使用的輸入資料。
        debug_print_stage1_first_round_prompt: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        tool_manager: Any,
        memory_tool: Any = None,
        memory_config: Any = None,
        shared_memory_user_id: str = "network_shared",
        enable_shared_memory: bool = False,
        memory_mode: str = "disabled",
        evidence_builder: Any = None,
        debug_print_stage1_first_round_prompt: bool = False,
    ):
        """
        負責執行 NetworkRuntime 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            tool_manager: 此流程需要使用的輸入資料。
            memory_tool: 記憶系統提供的檢索結果、寫入資料或操作介面。
            memory_config: 記憶系統提供的檢索結果、寫入資料或操作介面。
            shared_memory_user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            enable_shared_memory: 控制是否啟用此項資料、功能或處理分支的布林開關。
            memory_mode: 記憶系統提供的檢索結果、寫入資料或操作介面。
            evidence_builder: 此流程需要使用的輸入資料。
            debug_print_stage1_first_round_prompt: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.tool_manager = tool_manager
        self.memory_tool = memory_tool
        self.memory_config = memory_config
        self.shared_memory_user_id = shared_memory_user_id
        self.enable_shared_memory = enable_shared_memory
        self.memory_mode = memory_mode
        self.evidence_builder = evidence_builder
        self.enable_stage1_tools = False
        self.debug_print_stage1_first_round_prompt = debug_print_stage1_first_round_prompt
        self.last_stage1_first_round_prompt: str | None = None
        self.current_context: dict[str, Any] = {}
        self.current_attachment: dict[str, Any] | None = None
        self.shared_attachment_bundle: dict[str, Any] | None = None
        self.enable_stage1_attachment_after_first_round = False

        self.shared_tool_traces: list[dict[str, Any]] = []
        self.shared_memory_reads: list[dict[str, Any]] = []
        self.shared_memory_writes: list[dict[str, Any]] = []
        self.shared_token_usage: list[dict[str, Any]] = []
        self.shared_stage2_search_bundle: dict[str, Any] | None = None
        self.current_stage2_stage1_result: str | None = None
        self.current_stage2_top_k_answers: list[str] = []
        self.current_stage2_judge_scores: list[float] = []
        self.graph_memory: NetworkMemory | None = None
        self._init_graph_memory()

    @property
    def query_task_graph(self):
        """
        負責執行 NetworkRuntime 中的 query_task_graph 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.graph_memory.query_task_graph if self.graph_memory is not None else None

    @property
    def insight_graph(self):
        """
        負責執行 NetworkRuntime 中的 insight_graph 流程，依照 NetworkRuntime 的流程需求處理 insight_graph 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.graph_memory.insight_graph if self.graph_memory is not None else None

    def _init_graph_memory(self) -> None:
        """
        負責執行 NetworkRuntime 中的 _init_graph_memory 流程，依照 NetworkRuntime 的流程需求處理 _init_graph_memory 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        auto_connect = os.getenv("GAIA_GRAPH_MEMORY_NEO4J", "0") == "1"
        try:
            self.graph_memory = NetworkMemory(auto_connect=auto_connect, namespace="system")
        except Exception as exc:
            print(f"[WARN] graph memory init failed; graph guidance disabled: {exc}")
            self.graph_memory = None

    def record_tool_trace(self, trace: dict[str, Any]) -> None:
        """
        負責執行 NetworkRuntime 中的 record_tool_trace 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            trace: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.shared_tool_traces.append(trace)

    def record_memory_read(self, trace: dict[str, Any]) -> None:
        """
        負責執行 NetworkRuntime 中的 record_memory_read 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            trace: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.shared_memory_reads.append(trace)

    def record_memory_write(self, trace: dict[str, Any]) -> None:
        """
        負責執行 NetworkRuntime 中的 record_memory_write 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            trace: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.shared_memory_writes.append(trace)

    def record_token_usage(self, trace: dict[str, Any]) -> None:
        """
        負責執行 NetworkRuntime 中的 record_token_usage 流程，記錄或更新執行度量資料，供日誌與後續分析使用。
        
        Args:
            trace: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        prompt_tokens = int(trace.get("prompt_tokens", 0) or 0)
        completion_tokens = int(trace.get("completion_tokens", 0) or 0)
        self.shared_token_usage.append(
            {
                **trace,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(trace.get("total_tokens", prompt_tokens + completion_tokens) or 0),
            }
        )

    def token_usage_summary(self) -> dict[str, Any]:
        """
        負責執行 NetworkRuntime 中的 token_usage_summary 流程，依照 NetworkRuntime 的流程需求處理 token_usage_summary 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in self.shared_token_usage)
        completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in self.shared_token_usage)
        by_stage: dict[str, dict[str, int]] = {}
        by_model: dict[str, dict[str, int]] = {}
        for item in self.shared_token_usage:
            stage = str(item.get("stage", "unknown") or "unknown")
            model = str(item.get("model_name", "unknown") or "unknown")
            for bucket, key in ((by_stage, stage), (by_model, model)):
                stats = bucket.setdefault(key, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0})
                stats["prompt_tokens"] += int(item.get("prompt_tokens", 0) or 0)
                stats["completion_tokens"] += int(item.get("completion_tokens", 0) or 0)
                stats["total_tokens"] += int(item.get("total_tokens", 0) or 0)
                stats["calls"] += 1
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "calls": len(self.shared_token_usage),
            "by_stage": by_stage,
            "by_model": by_model,
            "records": list(self.shared_token_usage),
        }

    def clear_stage2_shared_state(self) -> None:
        """
        負責執行 NetworkRuntime 中的 clear_stage2_shared_state 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.shared_stage2_search_bundle = None
        self.current_stage2_stage1_result = None
        self.current_stage2_top_k_answers = []
        self.current_stage2_judge_scores = []

    def clear_current_context(self) -> None:
        """
        負責執行 NetworkRuntime 中的 clear_current_context 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.current_context = {}
        self.current_attachment = None
        self.shared_attachment_bundle = None

    def should_include_stage1_attachment(self, is_first_round: bool) -> bool:
        """
        負責執行 NetworkRuntime 中的 should_include_stage1_attachment 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
        
        Args:
            is_first_round: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.current_attachment is None:
            return False
        if is_first_round:
            return True
        return bool(self.enable_stage1_attachment_after_first_round)

    def prepare_shared_attachment_evidence(
        self,
        question: str,
        *,
        agent_id: str = "shared_attachment_reader",
        stage: str = "attachment_shared",
    ) -> dict[str, Any] | None:
        """
        負責執行 NetworkRuntime 中的 prepare_shared_attachment_evidence 流程，依照 NetworkRuntime 的流程需求處理 prepare_shared_attachment_evidence 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            agent_id: 目前執行或需要記錄的代理節點識別資訊。
            stage: 目前執行的階段、輪次或流程位置。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any] | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        builder = getattr(self, "evidence_builder", None)
        normalized_question = str(question or "").strip()
        if builder is None or not normalized_question or self.current_attachment is None:
            self.shared_attachment_bundle = None
            return None

        try:
            bundle = builder.build_shared_attachment_bundle(
                question=normalized_question,
                agent_id=agent_id,
                stage=stage,
            )
        except Exception as exc:
            print(f"[WARN] shared attachment evidence failed: {exc}")
            self.shared_attachment_bundle = None
            return None

        self.shared_attachment_bundle = bundle
        if bundle.get("tool_usage"):
            self.record_tool_trace(
                {
                    "agent_id": agent_id,
                    "stage": stage,
                    "question": normalized_question,
                    "tool_usage": bundle.get("tool_usage", []),
                    "metadata": bundle.get("metadata", {}),
                }
            )

        return bundle

    def prepare_shared_stage2_search(
        self,
        question: str,
        *,
        router_model_name: str | None = None,
    ) -> dict[str, Any] | None:
        """
        負責執行 NetworkRuntime 中的 prepare_shared_stage2_search 流程，依照 NetworkRuntime 的流程需求處理 prepare_shared_stage2_search 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            router_model_name: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any] | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        builder = getattr(self, "evidence_builder", None)
        normalized_question = str(question or "").strip()
        if builder is None or not normalized_question:
            self.shared_stage2_search_bundle = None
            return None

        try:
            bundle = builder.build_shared_stage2_search_bundle(
                question=normalized_question,
                agent_id="shared_stage2_search",
                stage="stage2_shared_search",
                router_model_name=router_model_name,
            )
        except Exception as exc:
            print(f"[WARN] shared stage2 search failed: {exc}")
            self.shared_stage2_search_bundle = None
            return None

        self.shared_stage2_search_bundle = bundle
        if bundle.get("tool_usage"):
            self.record_tool_trace(
                {
                    "agent_id": "shared_stage2_search",
                    "stage": "stage2_shared_search",
                    "question": normalized_question,
                    "tool_usage": bundle.get("tool_usage", []),
                    "routing": bundle.get("routing", {}),
                    "shared_search_id": bundle.get("shared_search_id"),
                    "queries": bundle.get("queries", []),
                }
            )

        return bundle

    def dedupe_memory_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        負責執行 NetworkRuntime 中的 dedupe_memory_records 流程，依照 NetworkRuntime 的流程需求處理 dedupe_memory_records 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            records: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for record in records:
            memory_type = str(record.get("memory_type", "") or "").strip()
            content = str(record.get("content", "") or "").strip()
            if not memory_type or not content:
                continue

            key = (memory_type, content)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)

        return deduped
