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
        def __init__(self, *args, **kwargs):
            raise ImportError("MCP requires fastmcp: pip install fastmcp")
    class MCPServer:
        def __init__(self, *args, **kwargs):
            raise ImportError("MCP requires fastmcp: pip install fastmcp")
    def create_context(*args, **kwargs):
        raise ImportError("MCP requires fastmcp: pip install fastmcp")
    def parse_context(*args, **kwargs):
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