"""內建工具模組

HelloAgents框架的內建工具集合，包括：
- SearchTool: 網頁搜尋工具
- CalculatorTool: 數學計算工具
- MemoryTool: 記憶工具
- RAGTool: 搜尋增強生成工具
- NoteTool: 結構化筆記工具
- TerminalTool: 命令行工具
- MCPTool: MCP 協議工具
- A2ATool: A2A 協議工具
- ANPTool: ANP 協議工具
- LLMJudgeTool: LLM Judge評估工具
- WinRateTool: Win Rate評估工具
"""

from .search_tool import SearchTool
from .calculator import CalculatorTool
# from .rag_tool import RAGTool
# from .note_tool import NoteTool
# from .terminal_tool import TerminalTool
# from .protocol_tools import MCPTool, A2ATool, ANPTool
# from .llm_judge_tool import LLMJudgeTool
# from .win_rate_tool import WinRateTool

__all__ = [
    "SearchTool",
    "CalculatorTool",
    "MemoryTool",
    "RAGTool",
    "NoteTool",
    "LLMJudgeTool",
    "WinRateTool",
]


def __getattr__(name):
    """
    負責執行 tools.builtin.__init__ 中的 __getattr__ 流程，依照 tools.builtin.__init__ 的流程需求處理 __getattr__ 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        name: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if name == "MemoryTool":
        from .memory_tool import MemoryTool

        return MemoryTool
    if name == "RAGTool":
        from .rag_tool import RAGTool

        return RAGTool
    raise AttributeError(name)
