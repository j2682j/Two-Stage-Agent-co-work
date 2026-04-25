"""例外體系。"""

class AgentsException(Exception):
    """HelloAgents 基底例外類別。"""
    pass

class LLMException(AgentsException):
    """LLM 相關例外。"""
    pass

class ConfigException(AgentsException):
    """設定相關例外。"""
    pass

class ToolException(AgentsException):
    """工具相關例外。"""
    pass
