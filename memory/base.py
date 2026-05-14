"""記憶系統的核心資料結構與抽象基底類別。"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from utils.project_paths import get_memory_data_dir


class MemoryItem(BaseModel):
    """
    負責在 memory.base 中封裝 MemoryItem，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}

    class Config:
        """
        負責在 memory.base 中封裝 Config，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
        
        Args:
            無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
        
        Returns:
            類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
        
        限制或副作用:
            方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
        """
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    """
    負責在 memory.base 中封裝 MemoryConfig，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

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
    """
    負責在 memory.base 中封裝 BaseMemory，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        config: 控制此流程行為的設定資料。
        storage_backend: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        """
        負責執行 BaseMemory 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
            storage_backend: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """
        負責執行 BaseMemory 中的 add 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            memory_item: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """
        負責執行 BaseMemory 中的 retrieve 流程，從記憶圖、向量索引或任務關聯中取回相關案例與策略提醒。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
            limit: 控制檢索、篩選或輸出數量的數值參數。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[MemoryItem]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        負責執行 BaseMemory 中的 update 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
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

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """
        負責執行 BaseMemory 中的 remove 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """
        負責執行 BaseMemory 中的 has_memory 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
        
        Args:
            memory_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """

    @abstractmethod
    def clear(self):
        """
        負責執行 BaseMemory 中的 clear 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        負責執行 BaseMemory 中的 get_stats 流程，依照 BaseMemory 的流程需求處理 get_stats 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """

    def _generate_id(self) -> str:
        """
        負責執行 BaseMemory 中的 _generate_id 流程，依照 BaseMemory 的流程需求處理 _generate_id 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        import uuid

        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """
        負責執行 BaseMemory 中的 _calculate_importance 流程，依照 BaseMemory 的流程需求處理 _calculate_importance 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
            base_importance: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 BaseMemory 中的 __str__ 流程，依照 BaseMemory 的流程需求處理 __str__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        """
        負責執行 BaseMemory 中的 __repr__ 流程，依照 BaseMemory 的流程需求處理 __repr__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.__str__()
