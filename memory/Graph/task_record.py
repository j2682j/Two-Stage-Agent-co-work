from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskRecord:
    """
    負責在 memory.graph.task_record 中封裝 TaskRecord，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    task_id: str
    source: str = "system"
    benchmark: str | None = None
    task_main: str = "system_task"
    task_description: str = ""
    input_text: str = ""
    final_answer: str = ""
    expected_answer: str = ""
    label: str = "unknown"
    score: float | None = None
    error_type: str | None = None
    failure_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state_chain: list[dict[str, Any]] = field(default_factory=list)
    task_trajectory: str = ""

    def to_dict(self) -> dict[str, Any]:
        """
        負責執行 TaskRecord 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        data = asdict(self)
        data["extra_fields"] = {
            **dict(self.metadata or {}),
            "source": self.source,
            "benchmark": self.benchmark,
            "input_text": self.input_text,
            "final_answer": self.final_answer,
            "expected_answer": self.expected_answer,
            "error_type": self.error_type,
            "failure_mode": self.failure_mode,
            "score": self.score,
        }
        return data

    def to_task_record_message(self) -> dict[str, Any]:
        """
        負責執行 TaskRecord 中的 to_task_record_message 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "task_id": self.task_id,
            "task_main": self.task_main,
            "task_description": self.task_description or self.input_text,
            "task_trajectory": self.task_trajectory,
            "label": self.label,
            "extra_fields": self.to_dict()["extra_fields"],
            "state_chain": list(self.state_chain or []),
        }

    @classmethod
    def from_interaction_graph(
        cls,
        interaction_graph: Any,
        *,
        source: str = "system",
        benchmark: str | None = None,
    ) -> "TaskRecord":
        """
        負責執行 TaskRecord 中的 from_interaction_graph 流程，依照 TaskRecord 的流程需求處理 from_interaction_graph 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            interaction_graph: 記憶系統提供的檢索結果、寫入資料或操作介面。
            source: 記憶系統提供的檢索結果、寫入資料或操作介面。
            benchmark: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'TaskRecord'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        message = (
            interaction_graph.to_mas_message()
            if hasattr(interaction_graph, "to_mas_message")
            else dict(interaction_graph or {})
        )
        extra = dict(message.get("extra_fields") or {})
        return cls(
            task_id=str(message.get("task_id", "") or ""),
            source=str(extra.get("source") or source),
            benchmark=extra.get("benchmark") or benchmark,
            task_main=str(message.get("task_main", "") or "system_task"),
            task_description=str(message.get("task_description", "") or ""),
            input_text=str(extra.get("input_text") or message.get("task_description") or ""),
            final_answer=str(extra.get("final_result") or extra.get("final_answer") or ""),
            expected_answer=str(extra.get("expected") or extra.get("expected_answer") or ""),
            label=str(message.get("label") or "unknown"),
            score=extra.get("score"),
            error_type=extra.get("error_type"),
            failure_mode=extra.get("failure_mode"),
            metadata=extra,
            state_chain=list(message.get("state_chain") or []),
            task_trajectory=str(message.get("task_trajectory", "") or ""),
        )
