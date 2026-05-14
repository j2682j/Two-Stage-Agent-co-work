"""MCP (Model Context Protocol) 協議實現

基於 fastmcp 和 mcp 庫的封裝，提供簡潔的 API 用於：
- 建立 MCP 伺服器（需要 fastmcp）
- 連線 MCP 伺服器（需要 mcp，可選）
- 管理模型上下文
"""

from .utils import create_context, parse_context

# 伺服器需要 fastmcp
try:
    from .server import MCPServer
    MCP_SERVER_AVAILABLE = True
except ImportError:
    MCP_SERVER_AVAILABLE = False
    class MCPServer:
        """
        負責在 protocols.mcp.__init__ 中封裝 MCPServer，封裝此模組的狀態資料與主要操作流程。
        
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
            raise ImportError(
                "MCP server requires the 'fastmcp' library. "
                "Install it with: pip install fastmcp"
            )

# 客戶端需要 mcp
try:
    from .client import MCPClient
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    class MCPClient:
        """
        負責在 protocols.mcp.__init__ 中封裝 MCPClient，封裝此模組的狀態資料與主要操作流程。
        
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
            raise ImportError(
                "MCP client requires the 'mcp' library. "
                "Install it with: pip install mcp"
            )

__all__ = [
    "MCPClient",
    "MCPServer",
    "create_context",
    "parse_context",
]
