"""智慧代理通信協議模組

本模組提供三種主要的智慧代理通信協議：
- MCP (Model Context Protocol): 模型上下文協定
- A2A (Agent-to-Agent Protocol): 代理間通訊協議
- ANP (Agent Network Protocol): 智慧代理網路協議

簡潔匯入範例：
    >>> from hello_agents.protocols import MCPClient, MCPServer
    >>> from hello_agents.protocols import A2AServer, A2AClient, AgentNetwork
    >>> from hello_agents.protocols import ANPDiscovery, ANPNetwork

完整匯入範例（向後相容）：
    >>> from hello_agents.protocols.mcp import MCPClient, MCPServer
    >>> from hello_agents.protocols.a2a import A2AServer, A2AClient
    >>> from hello_agents.protocols.anp import ANPDiscovery, ANPNetwork
"""

from .base import Protocol

# MCP 協議 - 匯出所有常用類（可選，需要 fastmcp）
try:
    from .mcp import (
        MCPClient,
        MCPServer,
        create_context,
        parse_context,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # 提供占位符
    class MCPClient:
        """
        負責在 protocols.__init__ 中封裝 MCPClient，封裝此模組的狀態資料與主要操作流程。
        
        Args:
            *args: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            **kwargs: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
        
        限制或副作用:
            方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
        """
        def __init__(self, *args, **kwargs):
            """
            負責執行 MCPClient 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
            
            Args:
                *args: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
                **kwargs: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            raise ImportError("MCP requires fastmcp: pip install fastmcp")
    class MCPServer:
        """
        負責在 protocols.__init__ 中封裝 MCPServer，封裝此模組的狀態資料與主要操作流程。
        
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
            負責執行 MCPServer 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
            
            Args:
                *args: 此流程需要使用的輸入資料。
                **kwargs: 此流程需要使用的輸入資料。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            raise ImportError("MCP requires fastmcp: pip install fastmcp")
    def create_context(*args, **kwargs):
        """
        負責執行 protocols.__init__ 中的 create_context 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            *args: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise ImportError("MCP requires fastmcp: pip install fastmcp")
    def parse_context(*args, **kwargs):
        """
        負責執行 protocols.__init__ 中的 parse_context 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            *args: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise ImportError("MCP requires fastmcp: pip install fastmcp")

# A2A 協議 - 匯出所有常用類
from .a2a import (
    A2AAgent,
    A2AServer,
    A2AClient,
    AgentNetwork,
    AgentRegistry,
    A2AMessage,
    MessageType,
    create_message,
    parse_message,
)

# ANP 協議 - 匯出所有常用類
from .anp import (
    ANPDiscovery,
    ANPNetwork,
    ServiceInfo,
    register_service,
    discover_service,
)

__all__ = [
    # 基礎協議
    "Protocol",

    # MCP 協議（可選）
    "MCPClient",
    "MCPServer",
    "create_context",
    "parse_context",

    # A2A 協議（可選）
    "A2AAgent",
    "A2AServer",
    "A2AClient",
    "AgentNetwork",
    "AgentRegistry",
    "A2AMessage",
    "MessageType",
    "create_message",
    "parse_message",

    # ANP 協議
    "ANPDiscovery",
    "ANPNetwork",
    "ServiceInfo",
    "register_service",
    "discover_service",
]

栓