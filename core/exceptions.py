"""異常體系"""

class HelloAgentsException(Exception):
    """HelloAgents基礎異常類"""
    pass

class LLMException(HelloAgentsException):
    """LLM相關異常"""
    pass

class AgentException(HelloAgentsException):
    """Agent相關異常"""
    pass

class ConfigException(HelloAgentsException):
    """設定相關異常"""
    pass

class ToolException(HelloAgentsException):
    """工具相關異常"""
    pass
0