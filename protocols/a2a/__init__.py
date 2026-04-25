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
        def __init__(self, *args, **kwargs):
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
    """建立 A2A 消息（占位符）"""
    raise ImportError("Please install a2a library: pip install a2a")

def parse_message(*args, **kwargs):
    """解析 A2A 消息（占位符）"""
    raise ImportError("Please install a2a library: pip install a2a")

P