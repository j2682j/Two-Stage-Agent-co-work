"""???????"""


from .base import Tool, ToolParameter
from .registry import ToolRegistry, global_registry

# 內建工具
from .builtin.search_tool import SearchTool
from .builtin.calculator import CalculatorTool
# from .builtin.rag_tool import RAGTool
# from .builtin.note_tool import NoteTool
# from .builtin.terminal_tool import TerminalTool

# 協議工具
# from .builtin.protocol_tools import MCPTool, A2ATool, ANPTool

# 評估工具（第12章）
# from .builtin.llm_judge_tool import LLMJudgeTool
# from .builtin.win_rate_tool import WinRateTool

# RL訓練工具（第11章）
# from .builtin.rl_training_tool import RLTrainingTool

# 高級功能
# from .chain import ToolChain, ToolChainManager, create_research_chain, create_simple_chain
# from .async_executor import AsyncToolExecutor, run_parallel_tools, run_batch_tool, run_parallel_tools_sync, run_batch_tool_sync

__all__ = [
    # 基礎工具系統
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "global_registry",

    # 內建工具
    "SearchTool",
    "CalculatorTool",
    "MemoryTool",
    "RAGTool",
    "NoteTool",
    "TerminalTool",

    # 協議工具
    "MCPTool",
    "A2ATool",
    "ANPTool",

    # 評估工具
    "LLMJudgeTool",
    "WinRateTool",

    # RL訓練工具
    "RLTrainingTool",

    # 工具鏈功能
    "ToolChain",
    "ToolChainManager",
    "create_research_chain",
    "create_simple_chain",

    # 非同步執行功能
    "AsyncToolExecutor",
    "run_parallel_tools",
    "run_batch_tool",
    "run_parallel_tools_sync",
    "run_batch_tool_sync",
]


def __getattr__(name):
    """
    負責執行 tools.__init__ 中的 __getattr__ 流程，依照 tools.__init__ 的流程需求處理 __getattr__ 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        name: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if name == "MemoryTool":
        from .builtin.memory_tool import MemoryTool

        return MemoryTool
    if name == "RAGTool":
        from .builtin.rag_tool import RAGTool

        return RAGTool
    raise AttributeError(name)
