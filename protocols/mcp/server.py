"""
基於 fastmcp 庫的 MCP 伺服器實現

使用 fastmcp 庫實現 Model Context Protocol 伺服器功能。
fastmcp 是一個快速建立 MCP 伺服器的 Python 庫。
"""

from typing import Dict, Any, List, Optional, Callable
try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "fastmcp is required for MCP server functionality. "
        "Install it with: pip install fastmcp"
    )


class MCPServer:
    """
    負責在 protocols.mcp.server 中封裝 MCPServer，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        name: 此流程需要使用的輸入資料。
        description: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None
    ):
        """
        負責執行 MCPServer 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.mcp = FastMCP(name=name)
        self.name = name
        self.description = description or f"{name} MCP Server"
        
    def add_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        負責執行 MCPServer 中的 add_tool 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            func: 此流程需要使用的輸入資料。
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 使用裝飾器註冊工具
        if name or description:
            self.mcp.tool(name=name, description=description)(func)
        else:
            self.mcp.tool()(func)
        
    def add_resource(
        self,
        func: Callable,
        uri: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        負責執行 MCPServer 中的 add_resource 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            func: 此流程需要使用的輸入資料。
            uri: 此流程需要使用的輸入資料。
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 使用裝飾器註冊資源
        if uri:
            self.mcp.resource(uri)(func)
        else:
            self.mcp.resource()(func)
        
    def add_prompt(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        負責執行 MCPServer 中的 add_prompt 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            func: 此流程需要使用的輸入資料。
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 使用裝飾器註冊提示詞
        if name or description:
            self.mcp.prompt(name=name, description=description)(func)
        else:
            self.mcp.prompt()(func)
        
    def run(self, transport: str = "stdio", **kwargs):
        """
        負責執行 MCPServer 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            transport: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.mcp.run(transport=transport, **kwargs)
        
    def get_info(self) -> Dict[str, Any]:
        """
        負責執行 MCPServer 中的 get_info 流程，依照 MCPServer 的流程需求處理 get_info 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "name": self.name,
            "description": self.description,
            "protocol": "MCP"
        }


# 便捷的伺服器建構器
class MCPServerBuilder:
    """
    負責在 protocols.mcp.server 中封裝 MCPServerBuilder，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        name: 此流程需要使用的輸入資料。
        description: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, name: str, description: Optional[str] = None):
        """
        負責執行 MCPServerBuilder 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.server = MCPServer(name, description)
        
    def with_tool(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """
        負責執行 MCPServerBuilder 中的 with_tool 流程，依照 MCPServerBuilder 的流程需求處理 with_tool 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            func: 此流程需要使用的輸入資料。
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'MCPServerBuilder'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.server.add_tool(func, name, description)
        return self
        
    def with_resource(self, func: Callable, uri: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """
        負責執行 MCPServerBuilder 中的 with_resource 流程，依照 MCPServerBuilder 的流程需求處理 with_resource 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            func: 此流程需要使用的輸入資料。
            uri: 此流程需要使用的輸入資料。
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'MCPServerBuilder'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.server.add_resource(func, uri, name, description)
        return self
        
    def with_prompt(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """
        負責執行 MCPServerBuilder 中的 with_prompt 流程，依照 MCPServerBuilder 的流程需求處理 with_prompt 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            func: 此流程需要使用的輸入資料。
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'MCPServerBuilder'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.server.add_prompt(func, name, description)
        return self
        
    def build(self) -> MCPServer:
        """
        負責執行 MCPServerBuilder 中的 build 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 MCPServer。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.server
        
    def run(self):
        """
        負責執行 MCPServerBuilder 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.server.run()


# 範例：建立一個簡單的 MCP 伺服器
def create_example_server() -> MCPServer:
    """
    負責執行 protocols.mcp.server 中的 create_example_server 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 MCPServer。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    server = MCPServer(
        name="example-server",
        description="A simple example MCP server with calculator and greeting tools"
    )
    
    # 添加一個簡單的計算器工具
    def calculator(expression: str) -> str:
        """
        負責執行 protocols.mcp.server 中的 calculator 流程，依照 protocols.mcp.server 的流程需求處理 calculator 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            expression: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            # 安全的表達式求值（僅支援基本運算）
            allowed_chars = set("0123456789+-*/() .")
            if not all(c in allowed_chars for c in expression):
                return f"Error: Invalid characters in expression"
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    server.add_tool(calculator, name="calculator", description="Calculate a mathematical expression")
    
    # 添加一個問候工具
    def greet(name: str) -> str:
        """
        負責執行 protocols.mcp.server 中的 greet 流程，依照 protocols.mcp.server 的流程需求處理 greet 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return f"Hello, {name}! Welcome to the MCP server example."
    
    server.add_tool(greet, name="greet", description="Generate a friendly greeting")
    
    return server


if __name__ == "__main__":
    # 建立並執行範例伺服器
    server = create_example_server()
    print(f"🚀 Starting {server.name}...")
    print(f"📝 {server.description}")
    print(f"🔌 Protocol: MCP")
    print(f"📡 Transport: stdio")
    print()
    server.run()

n