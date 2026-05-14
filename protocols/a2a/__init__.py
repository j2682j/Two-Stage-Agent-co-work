"""A2A (Agent-to-Agent Protocol) 協議實現

基於官方 a2a 庫的封裝，提供簡潔的 API 用於：
- Agent 間消息傳遞
- 任務委托與協商
- 多 Agent 協作

注意: A2A 功能需要安裝官方 SDK: pip install a2a
詳見文檔: docs/chapter10/A2A_GUIDE.md
"""

# A2A 是可選的，需要安裝官方 SDK
try:
    from .implementation import (
        A2AServer,
        A2AClient,
        AgentNetwork,
        AgentRegistry
    )
    __all__ = [
        "A2AServer",
        "A2AClient",
        "AgentNetwork",
        "AgentRegistry",
    ]
except ImportError as e:
    # 如果沒有安裝依賴，提供占位符
    __all__ = []
    
    class _A2ANotAvailable:
        """
        負責在 protocols.a2a.__init__ 中封裝 _A2ANotAvailable，封裝此模組的狀態資料與主要操作流程。
        
        Args:
            *args: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
        
        限制或副作用:
            方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
        """
        def __init__(self, *args, **kwargs):
            """
            負責執行 _A2ANotAvailable 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
            
            Args:
                *args: 此流程需要使用的輸入資料。
                **kwargs: 此流程需要使用的輸入資料。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            raise ImportError(
                "A2A protocol requires the official 'a2a' library. "
                "Install it with: pip install a2a\n"
                "See docs/chapter10/A2A_GUIDE.md for more information."
            )
    
    A2AServer = _A2ANotAvailable
    A2AClient = _A2ANotAvailable
    AgentNetwork = _A2ANotAvailable
    AgentRegistry = _A2ANotAvailable

# 為了向後相容，提供別名
A2AAgent = A2AServer
A2AMessage = dict  # 簡化的消息類型
MessageType = str

def create_message(*args, **kwargs):
    """
    負責執行 protocols.a2a.__init__ 中的 create_message 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        *args: 此流程需要使用的輸入資料。
        **kwargs: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    raise ImportError("Please install a2a library: pip install a2a")

def parse_message(*args, **kwargs):
    """
    負責執行 protocols.a2a.__init__ 中的 parse_message 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
    
    Args:
        *args: 此流程需要使用的輸入資料。
        **kwargs: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    raise ImportError("Please install a2a library: pip install a2a")

P