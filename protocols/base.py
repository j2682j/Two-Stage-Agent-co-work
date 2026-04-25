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
    """協議類型枚舉"""
    MCP = "mcp"  # Model Context Protocol
    A2A = "a2a"  # Agent-to-Agent Protocol
    ANP = "anp"  # Agent Network Protocol


# 為了向後相容，保留 Protocol 類的定義
# 但標記為概念性，不建議實際使用
class Protocol:
    """協議基類（概念性，不建議繼承）
    
    這個類定義了協議的基本概念，但實際實現不需要繼承它。
    各協議根據自己的特點獨立實現。
    """
    
    def __init__(self, protocol_type: ProtocolType, version: str = "1.0.0"):
        """初始化協議
        
        Args:
            protocol_type: 協議類型
            version: 協議版本
        """
        self._protocol_type = protocol_type
        self._version = version
    
    @property
    def protocol_name(self) -> str:
        """取得協議名稱"""
        return self._protocol_type.value
    
    @property
    def version(self) -> str:
        """取得協議版本"""
        return self._version
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(protocol={self.protocol_name}, version={self.version})"
    
    def __repr__(self) -> str:
        return self.__str__()

冀