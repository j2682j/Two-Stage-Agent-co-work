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
    """基於 fastmcp 庫的 MCP 伺服器"""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None
    ):
        """
        初始化 MCP 伺服器
        
        Args:
            name: 伺服器名稱
            description: 伺服器描述
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
        添加工具到伺服器
        
        Args:
            func: 工具函式
            name: 工具名稱（可選，預設使用函式名）
            description: 工具描述（可選，預設使用函式文檔字串）
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
        添加資源到伺服器
        
        Args:
            func: 資源處理函式
            uri: 資源 URI（可選）
            name: 資源名稱（可選）
            description: 資源描述（可選）
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
        添加提示詞模板到伺服器
        
        Args:
            func: 提示詞生成函式
            name: 提示詞名稱（可選）
            description: 提示詞描述（可選）
        """
        # 使用裝飾器註冊提示詞
        if name or description:
            self.mcp.prompt(name=name, description=description)(func)
        else:
            self.mcp.prompt()(func)
        
    def run(self, transport: str = "stdio", **kwargs):
        """執行伺服器

        Args:
            transport: 傳輸方式 ("stdio", "http", "sse")
            **kwargs: 傳輸特定的參數
                - host: HTTP 伺服器主機（預設 "127.0.0.1"）
                - port: HTTP 伺服器端口（預設 8000）
                - 其他 FastMCP.run() 支援的參數

        Examples:
            # Stdio 傳輸（預設）
            server.run()

            # HTTP 傳輸
            server.run(transport="http", host="0.0.0.0", port=8081)

            # SSE 傳輸
            server.run(transport="sse", host="0.0.0.0", port=8081)
        """
        self.mcp.run(transport=transport, **kwargs)
        
    def get_info(self) -> Dict[str, Any]:
        """
        取得伺服器資訊
        
        Returns:
            伺服器資訊字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "protocol": "MCP"
        }


# 便捷的伺服器建構器
class MCPServerBuilder:
    """MCP 伺服器建構器，提供鏈式 API"""

    def __init__(self, name: str, description: Optional[str] = None):
        self.server = MCPServer(name, description)
        
    def with_tool(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加工具（鏈式呼叫）"""
        self.server.add_tool(func, name, description)
        return self
        
    def with_resource(self, func: Callable, uri: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加資源（鏈式呼叫）"""
        self.server.add_resource(func, uri, name, description)
        return self
        
    def with_prompt(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加提示詞（鏈式呼叫）"""
        self.server.add_prompt(func, name, description)
        return self
        
    def build(self) -> MCPServer:
        """建構伺服器"""
        return self.server
        
    def run(self):
        """建構並執行伺服器"""
        self.server.run()


# 範例：建立一個簡單的 MCP 伺服器
def create_example_server() -> MCPServer:
    """建立一個範例 MCP 伺服器"""
    server = MCPServer(
        name="example-server",
        description="A simple example MCP server with calculator and greeting tools"
    )
    
    # 添加一個簡單的計算器工具
    def calculator(expression: str) -> str:
        """計算數學表達式
        
        Args:
            expression: 要計算的數學表達式，例如 "2 + 2" 或 "10 * 5"
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
        """生成友好的問候語
        
        Args:
            name: 要問候的人的名字
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