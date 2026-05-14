"""協議基類（概念性）

本模組定義了協議的基本介面概念。
實際實現中，各協議根據自己的特點獨立實現，不強制繼承這個基類。

協議介面概念：
- 協議標識：每個協議有唯一的名稱和版本
- 消息傳遞：支援發送和接收消息
- 資訊查詢：可以取得協議的基本資訊

實際使用：
- MCP: 使用 fastmcp 庫實現
- A2A: 使用官方 a2a 庫實現
- ANP: 使用概念性實現

注意：這個基類主要用於文檔說明，實際協議實現不需要繼承它。
"""

from enum import Enum


class ProtocolType(Enum):
    """
    負責在 protocols.base 中封裝 ProtocolType，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    MCP = "mcp"  # Model Context Protocol
    A2A = "a2a"  # Agent-to-Agent Protocol
    ANP = "anp"  # Agent Network Protocol


# 為了向後相容，保留 Protocol 類的定義
# 但標記為概念性，不建議實際使用
class Protocol:
    """
    負責在 protocols.base 中封裝 Protocol，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        protocol_type: 此流程需要使用的輸入資料。
        version: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(self, protocol_type: ProtocolType, version: str = "1.0.0"):
        """
        負責執行 Protocol 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            protocol_type: 此流程需要使用的輸入資料。
            version: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._protocol_type = protocol_type
        self._version = version
    
    @property
    def protocol_name(self) -> str:
        """
        負責執行 Protocol 中的 protocol_name 流程，依照 Protocol 的流程需求處理 protocol_name 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self._protocol_type.value
    
    @property
    def version(self) -> str:
        """
        負責執行 Protocol 中的 version 流程，依照 Protocol 的流程需求處理 version 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self._version
    
    def __str__(self) -> str:
        """
        負責執行 Protocol 中的 __str__ 流程，依照 Protocol 的流程需求處理 __str__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return f"{self.__class__.__name__}(protocol={self.protocol_name}, version={self.version})"
    
    def __repr__(self) -> str:
        """
        負責執行 Protocol 中的 __repr__ 流程，依照 Protocol 的流程需求處理 __repr__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.__str__()

冀