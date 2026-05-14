from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib

from .interaction_graph import AgentMessage, InteractionGraph


def _clean_text(value: Any) -> str:
    """
    負責執行 memory.graph.memory_base 中的 _clean_text 流程，依照 memory.graph.memory_base 的流程需求處理 _clean_text 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return " ".join(str(value or "").split()).strip()


def _task_id_from_text(text: str) -> str:
    """
    負責執行 memory.graph.memory_base 中的 _task_id_from_text 流程，依照 memory.graph.memory_base 的流程需求處理 _task_id_from_text 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        text: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    digest = hashlib.sha1(_clean_text(text).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"memory_task_{digest}"


@dataclass
class GraphMemoryBase:
    """
    負責在 memory.graph.memory_base 中封裝 GraphMemoryBase，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    namespace: str = "system"
    working_dir: str | Path = Path("memory") / "storage" / "graph"
    global_config: dict[str, Any] = field(default_factory=dict)
    current_task_context: InteractionGraph | None = field(default=None, init=False)
    persist_dir: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """
        負責執行 GraphMemoryBase 中的 __post_init__ 流程，依照 GraphMemoryBase 的流程需求處理 __post_init__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        configured_dir = self.global_config.get("working_dir") if isinstance(self.global_config, dict) else None
        base_dir = Path(configured_dir or self.working_dir)
        self.persist_dir = str(base_dir / self.namespace)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------- inside-trial memory ----------------------------------
    def init_task_context(
        self,
        task_main: str,
        task_description: str | None = None,
        *,
        task_id: str | None = None,
        label: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> InteractionGraph:
        """
        負責執行 GraphMemoryBase 中的 init_task_context 流程，依照 GraphMemoryBase 的流程需求處理 init_task_context 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            task_main: 目前要處理的任務、問題或查詢文字。
            task_description: 目前要處理的任務、問題或查詢文字。
            task_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            label: 記憶系統提供的檢索結果、寫入資料或操作介面。
            extra_fields: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InteractionGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        resolved_id = _clean_text(task_id) or _task_id_from_text(task_description or task_main)
        self.current_task_context = InteractionGraph(
            task_id=resolved_id,
            task_main=_clean_text(task_main) or resolved_id,
            task_description=_clean_text(task_description),
            label=label,
            extra_fields=dict(extra_fields or {}),
        )
        return self.current_task_context

    def add_agent_node(
        self,
        agent_message: AgentMessage,
        upstream_agent_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        負責執行 GraphMemoryBase 中的 add_agent_node 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            agent_message: 記憶系統提供的檢索結果、寫入資料或操作介面。
            upstream_agent_ids: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.current_task_context is None:
            raise RuntimeError("The current inside-trial memory is empty.")
        return self.current_task_context.add_message_to_current_state(
            agent_message,
            upstream_agent_ids or [],
            **kwargs,
        )

    def move_memory_state(self, action: str, observation: str, **kwargs: Any) -> None:
        """
        負責執行 GraphMemoryBase 中的 move_memory_state 流程，依照 GraphMemoryBase 的流程需求處理 move_memory_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            action: 記憶系統提供的檢索結果、寫入資料或操作介面。
            observation: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.current_task_context is None:
            raise RuntimeError("The current inside-trial memory is empty.")
        self.current_task_context.move_state(action, observation, **kwargs)

    def save_task_context(
        self,
        label: str | bool,
        feedback: str | None = None,
        *,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InteractionGraph:
        """
        負責執行 GraphMemoryBase 中的 save_task_context 流程，保存任務記錄、記憶圖節點或檢索索引，讓後續任務可以取回使用。
        
        Args:
            label: 記憶系統提供的檢索結果、寫入資料或操作介面。
            feedback: 記憶系統提供的檢索結果、寫入資料或操作介面。
            score: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InteractionGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.current_task_context is None:
            raise RuntimeError("The current inside-trial memory is empty.")

        if isinstance(label, bool):
            resolved_label = "successful" if label else "failed"
        else:
            resolved_label = _clean_text(label) or "unknown"

        self.current_task_context.label = resolved_label
        if score is not None:
            self.current_task_context.add_extra_field("score", score)
        if metadata:
            for key, value in metadata.items():
                self.current_task_context.add_extra_field(key, value)
        if feedback:
            description = self.current_task_context.task_description or ""
            self.current_task_context.task_description = f"{description}\n- Environment feedback\n{feedback}\n".strip()

        self.add_memory(self.current_task_context)
        return self.current_task_context

    def summarize(self, **kwargs: Any) -> str:
        """
        負責執行 GraphMemoryBase 中的 summarize 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.current_task_context is None:
            return ""
        return _clean_text(
            f"{self.current_task_context.task_description or ''}\n{self.current_task_context.task_trajectory or ''}"
        )

    # ---------------------------------- cross-trials memory ----------------------------------
    def add_memory(self, mas_message: InteractionGraph) -> None:
        """
        負責執行 GraphMemoryBase 中的 add_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            mas_message: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return None

    def retrieve_memory(self, **kwargs: Any) -> tuple[list, list, list]:
        """
        負責執行 GraphMemoryBase 中的 retrieve_memory 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 tuple[list, list, list]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [], [], []

    def update_memory(self, query: str, **kwargs: Any) -> None:
        """
        負責執行 GraphMemoryBase 中的 update_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return None

    def backward(self, reward: Any, **kwargs: Any) -> None:
        """
        負責執行 GraphMemoryBase 中的 backward 流程，根據結果、評分或回饋更新節點狀態與權重資訊。
        
        Args:
            reward: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return None
