"""tools.builtin.__init__ 模組。

提供此模組相關的資料結構、流程輔助或整合邏輯。
"""

from .search_tool import SearchTool
from .calculator import CalculatorTool

__all__ = [
    "SearchTool",
    "CalculatorTool",
    "RAGTool",
]


def __getattr__(name):
    """處理 getattr 流程並回傳結果。
    
    參數:
        name: 此流程需要使用的輸入資料。
    """
    if name == "RAGTool":
        from .rag_tool import RAGTool

        return RAGTool
    raise AttributeError(name)
