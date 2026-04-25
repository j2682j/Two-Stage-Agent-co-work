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
        def __init__(self, *args, **kwargs):
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
        def __init__(self, *args, **kwargs):
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
