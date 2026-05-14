from __future__ import annotations

from typing import Any


class MemoryUsageTracer:
    """
    負責在 evaluation.benchmarks.gaia.gaia_memory_trace 中封裝 MemoryUsageTracer，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        memory_tool: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, memory_tool: Any):
        """
        負責執行 MemoryUsageTracer 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            memory_tool: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.memory_tool = memory_tool
        self.memory_manager = getattr(memory_tool, "memory_manager", None)
        self.original_retrieve = None
        self.current_sample = None
        self.previous_markers: list[str] = []
        self.used_previous_memory = False
        self.hit_records: list[dict[str, Any]] = []

    def install(self) -> None:
        """
        負責執行 MemoryUsageTracer 中的 install 流程，依照 MemoryUsageTracer 的流程需求處理 install 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.memory_manager is None or self.original_retrieve is not None:
            return

        self.original_retrieve = self.memory_manager.retrieve_memories

        def traced_retrieve(*args, **kwargs):
            """
            負責執行 MemoryUsageTracer 中的 traced_retrieve 流程，依照 MemoryUsageTracer 的流程需求處理 traced_retrieve 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                *args: 記憶系統提供的檢索結果、寫入資料或操作介面。
                **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            results = self.original_retrieve(*args, **kwargs)
            if self.current_sample and self.previous_markers:
                for memory in results or []:
                    content = getattr(memory, "content", "") or ""
                    for marker in self.previous_markers:
                        if marker and marker in content:
                            self.used_previous_memory = True
                            self.hit_records.append(
                                {
                                    "query": kwargs.get("query", ""),
                                    "marker": marker,
                                    "memory_type": getattr(memory, "memory_type", ""),
                                    "preview": content[:160],
                                }
                            )
                            break
            return results

        self.memory_manager.retrieve_memories = traced_retrieve

    def start_sample(self, sample: dict[str, Any]) -> None:
        """
        負責執行 MemoryUsageTracer 中的 start_sample 流程，依照 MemoryUsageTracer 的流程需求處理 start_sample 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sample: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.current_sample = sample
        self.used_previous_memory = False
        self.hit_records = []

    def finish_sample(self) -> None:
        """
        負責執行 MemoryUsageTracer 中的 finish_sample 流程，依照 MemoryUsageTracer 的流程需求處理 finish_sample 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.current_sample = None

    def register_feedback_marker(self, sample: dict[str, Any], normalized_question: str) -> None:
        """
        負責執行 MemoryUsageTracer 中的 register_feedback_marker 流程，依照 MemoryUsageTracer 的流程需求處理 register_feedback_marker 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sample: 記憶系統提供的檢索結果、寫入資料或操作介面。
            normalized_question: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        task_id = sample.get("task_id", "")
        question = sample.get("question", "")
        markers = [task_id, normalized_question]
        if question:
            markers.append(question[:80])
        for marker in markers:
            if marker and marker not in self.previous_markers:
                self.previous_markers.append(marker)


def print_memory_stats(memory_tool: Any) -> None:
    """
    負責執行 evaluation.benchmarks.gaia.gaia_memory_trace 中的 print_memory_stats 流程，依照 evaluation.benchmarks.gaia.gaia_memory_trace 的流程需求處理 print_memory_stats 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        memory_tool: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    memory_manager = getattr(memory_tool, "memory_manager", None)
    if memory_manager is None:
        print("   [MEMORY][stats] unavailable")
        return

    stats = memory_manager.get_memory_stats()
    by_type = stats.get("memories_by_type", {}) or {}
    summary = {
        memory_type: details.get("count", 0)
        for memory_type, details in by_type.items()
    }
    print(f"   [MEMORY][stats] total={stats.get('total_memories', 0)} by_type={summary}")


def print_memory_debug(memory_tool: Any, normalized_question: str) -> None:
    """
    負責執行 evaluation.benchmarks.gaia.gaia_memory_trace 中的 print_memory_debug 流程，依照 evaluation.benchmarks.gaia.gaia_memory_trace 的流程需求處理 print_memory_debug 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        memory_tool: 記憶系統提供的檢索結果、寫入資料或操作介面。
        normalized_question: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    for memory_type in ["working", "semantic", "episodic"]:
        memory_debug = memory_tool.run(
            {
                "action": "search",
                "query": normalized_question,
                "memory_type": memory_type,
                "limit": 5,
            }
        )
        print(f"   [MEMORY][{memory_type}]")
        print(f"   {memory_debug}")


def print_memory_hits(tracer: MemoryUsageTracer | None) -> None:
    """
    負責執行 evaluation.benchmarks.gaia.gaia_memory_trace 中的 print_memory_hits 流程，依照 evaluation.benchmarks.gaia.gaia_memory_trace 的流程需求處理 print_memory_hits 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        tracer: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if tracer is None:
        return
    print(f"   used_previous_memory={tracer.used_previous_memory}")
    for hit in tracer.hit_records[:5]:
        print(
            "   [MEMORY-HIT] "
            f"type={hit['memory_type']} "
            f"marker={hit['marker']!r} "
            f"query={hit['query']!r}"
        )
        print(f"      preview={hit['preview']}")
