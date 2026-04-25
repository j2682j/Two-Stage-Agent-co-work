"""記憶系統的核心資料結構與抽象基底類別。"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.project_paths import get_memory_data_dir


class MemoryItem(BaseModel):
    """記憶系統中使用的結構化記憶項目。"""

    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    """記憶模組共用的設定物件。"""

    storage_path: str = Field(default_factory=lambda: str(get_memory_data_dir()))

    # 全域預設值
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作記憶設定
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # 感知記憶設定
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]


class BaseMemory(ABC):
    """所有記憶類型共同遵循的抽象介面。"""

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """新增記憶項目並回傳其 id。"""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """依查詢內容取回相關記憶。"""

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """更新既有記憶項目。"""

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """依 id 移除記憶項目。"""

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """判斷記憶庫中是否存在指定 id。"""

    @abstractmethod
    def clear(self):
        """清空目前儲存的所有記憶。"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """回傳記憶庫層級的統計資訊。"""

    def _generate_id(self) -> str:
        """產生新的唯一記憶 id。"""
        import uuid

        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """依內容估算簡單的重要性分數。"""
        importance = base_importance

        if len(content) > 100:
            importance += 0.1

        important_keywords = [
            "important",
            "key",
            "critical",
            "result",
            "answer",
            "lesson",
        ]
        lowered = content.lower()
        if any(keyword in lowered for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()
