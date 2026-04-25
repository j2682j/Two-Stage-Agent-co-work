"""HelloAgents ??????"""


# 設定第三方庫的日誌等級，減少噪音
# import logging
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("qdrant_client").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("neo4j").setLevel(logging.WARNING)
# logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

# from .version import __version__, __author__, __email__, __description__

# 核心元件
# from .core.llm import HelloAgentsLLM
# from .core.config import Config
# from .core.message import Message
# from .core.exceptions import HelloAgentsException

# Agent實現
# from .agents.simple_agent import SimpleAgent
# from .agents.function_call_agent import FunctionCallAgent
# from .agents.react_agent import ReActAgent
# from .agents.reflection_agent import ReflectionAgent
# from .agents.plan_solve_agent import PlanAndSolveAgent
# from .agents.tool_aware_agent import ToolAwareSimpleAgent

# 工具系統
# from .tools.registry import ToolRegistry, global_registry
from .tools.builtin.search_tool import SearchTool, search
from .tools.builtin.calculator import CalculatorTool, calculate
# from .tools.chain import ToolChain, ToolChainManager
# from .tools.async_executor import AsyncToolExecutor

__all__ = [
    # 版本資訊
    "__version__",
    "__author__",
    "__email__",
    "__description__",

    # 核心元件
    "HelloAgentsLLM",
    "Config",
    "Message",
    "HelloAgentsException",

    # Agent范式
    "SimpleAgent",
    "FunctionCallAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",
    "ToolAwareSimpleAgent",

    # 工具系統
    "ToolRegistry",
    "global_registry",
    "SearchTool",
    "search",
    "CalculatorTool",
    "calculate",
    "ToolChain",
    "ToolChainManager",
    "AsyncToolExecutor",
]
