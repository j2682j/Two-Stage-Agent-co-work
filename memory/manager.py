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
    """統一管理各種記憶類型，並提供讀寫、整併與分類能力。"""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = True,
    ):
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
        """新增記憶，可依內容自動分類到合適的記憶類型。"""
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
        """從指定記憶類型中檢索與查詢相關的記憶。"""
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
        """依記憶 id 更新內容、重要度或 metadata。"""
        for memory_instance in self.memory_types.values():
            if memory_instance.has_memory(memory_id):
                return memory_instance.update(memory_id, content, importance, metadata)

        logger.warning("找不到要更新的記憶 id：%s", memory_id)
        return False

    def remove_memory(self, memory_id: str) -> bool:
        """依記憶 id 刪除記憶。"""
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
        """依策略清理過舊或不重要的記憶。"""
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
        """將高重要度記憶從一種記憶類型整併到另一種記憶類型。"""
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
        """回傳目前記憶系統的統計資訊。"""
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
        """清除所有記憶類型中的內容。"""
        for memory_instance in self.memory_types.values():
            memory_instance.clear()
        logger.info("已清除所有記憶")

    def _classify_memory_type(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """根據 metadata 與內容特徵判斷記憶類型。"""
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
        """判斷內容是否更像事件、過程、單次案例或經驗回顧。"""
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
        """判斷內容是否更像可泛化的知識、規則或 lesson。"""
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
        """依內容與 metadata 粗略估計記憶重要度。"""
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
        stats = self.get_memory_stats()
        return f"MemoryManager(user={self.user_id}, total={stats['total_memories']})"
