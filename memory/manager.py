"""整合 Working、Episodic、Semantic、Perceptual 的記憶管理器。"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import uuid

from .base import MemoryConfig, MemoryItem
from .types.episodic import EpisodicMemory
from .types.perceptual import PerceptualMemory
from .types.semantic import SemanticMemory
from .types.working import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    負責在 memory.manager 中封裝 MemoryManager，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        config: 控制此流程行為的設定資料。
        user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        enable_working: 控制是否啟用此項資料、功能或處理分支的布林開關。
        enable_episodic: 控制是否啟用此項資料、功能或處理分支的布林開關。
        enable_semantic: 控制是否啟用此項資料、功能或處理分支的布林開關。
        enable_perceptual: 控制是否啟用此項資料、功能或處理分支的布林開關。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = True,
    ):
        """
        負責執行 MemoryManager 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
            user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            enable_working: 控制是否啟用此項資料、功能或處理分支的布林開關。
            enable_episodic: 控制是否啟用此項資料、功能或處理分支的布林開關。
            enable_semantic: 控制是否啟用此項資料、功能或處理分支的布林開關。
            enable_perceptual: 控制是否啟用此項資料、功能或處理分支的布林開關。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.memory_types: Dict[str, Any] = {}

        if enable_working:
            self.memory_types["working"] = WorkingMemory(self.config)
        if enable_episodic:
            self.memory_types["episodic"] = EpisodicMemory(self.config)
        if enable_semantic:
            self.memory_types["semantic"] = SemanticMemory(self.config)
        if enable_perceptual:
            self.memory_types["perceptual"] = PerceptualMemory(self.config)

        logger.info(
            "MemoryManager 初始化完成，啟用記憶類型：%s",
            list(self.memory_types.keys()),
        )

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = True,
    ) -> str:
        """
        負責執行 MemoryManager 中的 add_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            memory_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            importance: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
            auto_classify: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if auto_classify:
            memory_type = self._classify_memory_type(content, metadata)

        if importance is None:
            importance = self._calculate_importance(content, metadata)

        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {},
        )

        if memory_type not in self.memory_types:
            raise ValueError(f"不支援的記憶類型：{memory_type}")

        memory_id = self.memory_types[memory_type].add(memory_item)
        logger.debug("已新增 %s 記憶：%s", memory_type, memory_id)
        return memory_id

    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        time_range: Optional[tuple] = None,
    ) -> List[MemoryItem]:
        """
        負責執行 MemoryManager 中的 retrieve_memories 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            memory_types: 記憶系統提供的檢索結果、寫入資料或操作介面。
            limit: 控制檢索、篩選或輸出數量的數值參數。
            min_importance: 控制檢索、篩選或輸出數量的數值參數。
            time_range: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[MemoryItem]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if memory_types is None:
            memory_types = list(self.memory_types.keys())

        if not memory_types:
            return []

        all_results: List[MemoryItem] = []
        per_type_limit = max(1, limit // len(memory_types))

        for memory_type in memory_types:
            if memory_type not in self.memory_types:
                continue

            memory_instance = self.memory_types[memory_type]
            try:
                type_results = memory_instance.retrieve(
                    query=query,
                    limit=per_type_limit,
                    min_importance=min_importance,
                    user_id=self.user_id,
                    time_range=time_range,
                )
                all_results.extend(type_results)
            except Exception as exc:
                logger.warning("檢索 %s 記憶失敗：%s", memory_type, exc)

        all_results.sort(key=lambda item: item.importance, reverse=True)
        return all_results[:limit]

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        負責執行 MemoryManager 中的 update_memory 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            importance: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for memory_instance in self.memory_types.values():
            if memory_instance.has_memory(memory_id):
                return memory_instance.update(memory_id, content, importance, metadata)

        logger.warning("找不到要更新的記憶 id：%s", memory_id)
        return False

    def remove_memory(self, memory_id: str) -> bool:
        """
        負責執行 MemoryManager 中的 remove_memory 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for memory_instance in self.memory_types.values():
            if memory_instance.has_memory(memory_id):
                return memory_instance.remove(memory_id)

        logger.warning("找不到要刪除的記憶 id：%s", memory_id)
        return False

    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        """
        負責執行 MemoryManager 中的 forget_memories 流程，依照 MemoryManager 的流程需求處理 forget_memories 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            strategy: 記憶系統提供的檢索結果、寫入資料或操作介面。
            threshold: 控制檢索、篩選或輸出數量的數值參數。
            max_age_days: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        total_forgotten = 0

        for memory_instance in self.memory_types.values():
            if hasattr(memory_instance, "forget"):
                forgotten = memory_instance.forget(strategy, threshold, max_age_days)
                total_forgotten += forgotten

        logger.info("本次共清理 %s 筆記憶", total_forgotten)
        return total_forgotten

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ) -> int:
        """
        負責執行 MemoryManager 中的 consolidate_memories 流程，依照 MemoryManager 的流程需求處理 consolidate_memories 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            from_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            to_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            importance_threshold: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if from_type not in self.memory_types or to_type not in self.memory_types:
            logger.warning("不支援的整併來源或目標類型：%s -> %s", from_type, to_type)
            return 0

        source_memory = self.memory_types[from_type]
        target_memory = self.memory_types[to_type]
        all_memories = source_memory.get_all()
        candidates = [item for item in all_memories if item.importance >= importance_threshold]

        consolidated_count = 0
        for memory in candidates:
            if source_memory.remove(memory.id):
                memory.memory_type = to_type
                memory.importance *= 1.1
                target_memory.add(memory)
                consolidated_count += 1

        logger.info(
            "已將 %s 筆記憶從 %s 整併到 %s",
            consolidated_count,
            from_type,
            to_type,
        )
        return consolidated_count

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        負責執行 MemoryManager 中的 get_memory_stats 流程，依照 MemoryManager 的流程需求處理 get_memory_stats 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        stats = {
            "user_id": self.user_id,
            "enabled_types": list(self.memory_types.keys()),
            "total_memories": 0,
            "memories_by_type": {},
            "config": {
                "max_capacity": self.config.max_capacity,
                "importance_threshold": self.config.importance_threshold,
                "decay_factor": self.config.decay_factor,
            },
        }

        for memory_type, memory_instance in self.memory_types.items():
            type_stats = memory_instance.get_stats()
            stats["memories_by_type"][memory_type] = type_stats
            stats["total_memories"] += type_stats.get("count", 0)

        return stats

    def clear_all_memories(self):
        """
        負責執行 MemoryManager 中的 clear_all_memories 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for memory_instance in self.memory_types.values():
            memory_instance.clear()
        logger.info("已清除所有記憶")

    def _classify_memory_type(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """
        負責執行 MemoryManager 中的 _classify_memory_type 流程，依照 MemoryManager 的流程需求處理 _classify_memory_type 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if metadata and metadata.get("type"):
            return metadata["type"]

        lowered = content.lower()
        if lowered.startswith("gaia success reminder"):
            return "working"
        if lowered.startswith("gaia correction lesson"):
            return "semantic"
        if lowered.startswith("gaia failure case"):
            return "episodic"

        if self._is_episodic_content(content):
            return "episodic"
        if self._is_semantic_content(content):
            return "semantic"
        return "working"

    def _is_episodic_content(self, content: str) -> bool:
        """
        負責執行 MemoryManager 中的 _is_episodic_content 流程，依照 MemoryManager 的流程需求處理 _is_episodic_content 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lowered = content.lower()
        episodic_markers = [
            "gaia failure case",
            "gaia judge reflection",
            "predicted answer:",
            "expected answer:",
            "exact match:",
            "partial match:",
            "score:",
            "during the previous task",
            "previous attempt",
            "the agent made a mistake",
            "corrected the answer",
            "session",
            "interaction",
            "conversation",
            "experience",
            "event",
        ]
        semantic_lesson_markers = [
            "gaia correction lesson",
            "error type:",
            "lesson:",
            "tags:",
            "applicability:",
            "rule:",
            "verified fact:",
        ]
        has_episodic_signal = any(marker in lowered for marker in episodic_markers)
        has_semantic_signal = any(marker in lowered for marker in semantic_lesson_markers)
        return has_episodic_signal and not has_semantic_signal

    def _is_semantic_content(self, content: str) -> bool:
        """
        負責執行 MemoryManager 中的 _is_semantic_content 流程，依照 MemoryManager 的流程需求處理 _is_semantic_content 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lowered = content.lower()
        semantic_markers = [
            "gaia correction lesson",
            "error type:",
            "lesson:",
            "tags:",
            "applicability:",
            "rule:",
            "verified fact:",
            "definition",
            "knowledge",
            "conclusion",
            "use for ",
        ]
        return any(marker in lowered for marker in semantic_markers)

    def _calculate_importance(self, content: str, metadata: Optional[Dict[str, Any]]) -> float:
        """
        負責執行 MemoryManager 中的 _calculate_importance 流程，依照 MemoryManager 的流程需求處理 _calculate_importance 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        importance = 0.5
        lowered = content.lower()

        if len(content) > 100:
            importance += 0.1

        important_keywords = ["important", "key", "critical", "result", "answer", "lesson"]
        if any(keyword in lowered for keyword in important_keywords):
            importance += 0.2

        if metadata:
            if metadata.get("priority") == "high":
                importance += 0.3
            elif metadata.get("priority") == "low":
                importance -= 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        """
        負責執行 MemoryManager 中的 __str__ 流程，依照 MemoryManager 的流程需求處理 __str__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        stats = self.get_memory_stats()
        return f"MemoryManager(user={self.user_id}, total={stats['total_memories']})"
