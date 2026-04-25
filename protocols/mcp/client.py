"""
增強的 MCP 客戶端實現

支援多種傳輸方式的 MCP 客戶端，用於教學和實際應用。
這個實現展示了如何使用不同的傳輸方式連線到 MCP 伺服器。

支援的傳輸方式：
1. Memory: 記憶體傳輸（用於測試，直接傳遞 FastMCP 實例）
2. Stdio: 標準輸入輸出傳輸（本地進程，Python/Node.js 腳本）
3. HTTP: HTTP 傳輸（遠程伺服器）
4. SSE: Server-Sent Events 傳輸（實時通信）

使用範例：
```python
# 1. 記憶體傳輸（測試）
from fastmcp import FastMCP
server = FastMCP("TestServer")
client = MCPClient(server)

# 2. Stdio 傳輸（本地腳本）
client = MCPClient("server.py")
client = MCPClient(["python", "server.py"])

# 3. HTTP 傳輸（遠程伺服器）
client = MCPClient("https://api.example.com/mcp")

# 4. SSE 傳輸（實時通信）
client = MCPClient("https://api.example.com/mcp", transport_type="sse")

# 5. 設定傳輸（高級使用方式）
config = {
    "transport": "stdio",
    "command": "python",
    "args": ["server.py"],
    "env": {"DEBUG": "1"}
}
client = MCPClient(config)
```
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import os

try:
    from fastmcp import Client, FastMCP
    from fastmcp.client.transports import PythonStdioTransport, SSETransport, StreamableHttpTransport
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    Client = None
    FastMCP = None
    PythonStdioTransport = None
    SSETransport = None
    StreamableHttpTransport = None


class MCPClient:
    """MCP 客戶端，支援多種傳輸方式"""

    def __init__(self,
                 server_source: Union[str, List[str], FastMCP, Dict[str, Any]],
                 server_args: Optional[List[str]] = None,
                 transport_type: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None,
                 **transport_kwargs):
        """
        初始化MCP 客戶端

        Args:
            server_source: 伺服器源，支援多種格式：
                - FastMCP 實例: 記憶體傳輸（用於測試）
                - 字串路徑: Python 腳本路徑（如 "server.py"）
                - HTTP URL: 遠程伺服器（如 "https://api.example.com/mcp"）
                - 命令列表: 完整命令（如 ["python", "server.py"]）
                - 設定字典: 傳輸設定
            server_args: 伺服器參數列表（可選）
            transport_type: 強制指定傳輸類型 ("stdio", "http", "sse", "memory")
            env: 環境變數字典（傳遞給MCP伺服器進程）
            **transport_kwargs: 傳輸特定的額外參數

        Raises:
            ImportError: 如果 fastmcp 庫未安裝
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "Enhanced MCP client requires the 'fastmcp' library (version 2.0+). "
                "Install it with: pip install fastmcp>=2.0.0"
            )

        self.server_args = server_args or []
        self.transport_type = transport_type
        self.env = env or {}
        self.transport_kwargs = transport_kwargs
        self.server_source = self._prepare_server_source(server_source)
        self.client: Optional[Client] = None
        self._context_manager = None

    def _prepare_server_source(self, server_source: Union[str, List[str], FastMCP, Dict[str, Any]]):
        """準備伺服器源，根據類型建立合適的傳輸設定"""
        
        # 1. FastMCP 實例 - 記憶體傳輸
        if isinstance(server_source, FastMCP):
            print(f"🧠 使用記憶體傳輸: {server_source.name}")
            return server_source
        
        # 2. 設定字典 - 根據設定建立傳輸
        if isinstance(server_source, dict):
            print(f"⚙️ 使用設定傳輸: {server_source.get('transport', 'stdio')}")
            return self._create_transport_from_config(server_source)
        
        # 3. HTTP URL - HTTP/SSE 傳輸
        if isinstance(server_source, str) and (server_source.startswith("http://") or server_source.startswith("https://")):
            transport_type = self.transport_type or "http"
            print(f"🌐 使用 {transport_type.upper()} 傳輸: {server_source}")
            if transport_type == "sse":
                return SSETransport(url=server_source, **self.transport_kwargs)
            else:
                return StreamableHttpTransport(url=server_source, **self.transport_kwargs)

        # 4. Python 腳本路徑 - Stdio 傳輸
        if isinstance(server_source, str) and server_source.endswith(".py"):
            print(f"🐍 使用 Stdio 傳輸 (Python): {server_source}")
            return PythonStdioTransport(
                script_path=server_source,
                args=self.server_args,
                env=self.env if self.env else None,
                **self.transport_kwargs
            )

        # 5. 命令列表 - Stdio 傳輸
        if isinstance(server_source, list) and len(server_source) >= 1:
            print(f"📝 使用 Stdio 傳輸 (命令): {' '.join(server_source)}")
            if server_source[0] == "python" and len(server_source) > 1 and server_source[1].endswith(".py"):
                # Python 腳本
                return PythonStdioTransport(
                    script_path=server_source[1],
                    args=server_source[2:] + self.server_args,
                    env=self.env if self.env else None,
                    **self.transport_kwargs
                )
            else:
                # 其他命令，使用通用 Stdio 傳輸
                from fastmcp.client.transports import StdioTransport
                return StdioTransport(
                    command=server_source[0],
                    args=server_source[1:] + self.server_args,
                    env=self.env if self.env else None,
                    **self.transport_kwargs
                )
        
        # 6. 其他情況 - 直接回傳，讓 FastMCP 自動推斷
        print(f"🔍 自動推斷傳輸: {server_source}")
        return server_source

    def _create_transport_from_config(self, config: Dict[str, Any]):
        """從設定字典建立傳輸"""
        transport_type = config.get("transport", "stdio")
        
        if transport_type == "stdio":
            # 檢查是否是 Python 腳本
            args = config.get("args", [])
            if args and args[0].endswith(".py"):
                return PythonStdioTransport(
                    script_path=args[0],
                    args=args[1:] + self.server_args,
                    env=config.get("env"),
                    cwd=config.get("cwd"),
                    **self.transport_kwargs
                )
            else:
                # 使用通用 Stdio 傳輸
                from fastmcp.client.transports import StdioTransport
                return StdioTransport(
                    command=config.get("command", "python"),
                    args=args + self.server_args,
                    env=config.get("env"),
                    cwd=config.get("cwd"),
                    **self.transport_kwargs
                )
        elif transport_type == "sse":
            return SSETransport(
                url=config["url"],
                headers=config.get("headers"),
                auth=config.get("auth"),
                **self.transport_kwargs
            )
        elif transport_type == "http":
            return StreamableHttpTransport(
                url=config["url"],
                headers=config.get("headers"),
                auth=config.get("auth"),
                **self.transport_kwargs
            )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

    async def __aenter__(self):
        """非同步上下文管理器入口"""
        print("🔗 連線到 MCP 伺服器...")
        self.client = Client(self.server_source)
        self._context_manager = self.client
        await self._context_manager.__aenter__()
        print("✅ 連線成功！")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同步上下文管理器出口"""
        if self._context_manager:
            await self._context_manager.__aexit__(exc_type, exc_val, exc_tb)
            self.client = None
            self._context_manager = None
        print("🔌 連線已斷開")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用的工具"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_tools()

        # 處理不同的回傳格式
        if hasattr(result, 'tools'):
            tools = result.tools
        elif isinstance(result, list):
            tools = result
        else:
            tools = []

        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            }
            for tool in tools
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """呼叫 MCP 工具"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.call_tool(tool_name, arguments)

        # 解析結果 - FastMCP 回傳 ToolResult 對象
        if hasattr(result, 'content') and result.content:
            if len(result.content) == 1:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'data'):
                    return content.data
            return [
                getattr(c, 'text', getattr(c, 'data', str(c)))
                for c in result.content
            ]
        return None

    async def list_resources(self) -> List[Dict[str, Any]]:
        """列出所有可用的資源"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_resources()
        return [
            {
                "uri": resource.uri,
                "name": resource.name or "",
                "description": resource.description or "",
                "mime_type": getattr(resource, 'mimeType', None)
            }
            for resource in result.resources
        ]

    async def read_resource(self, uri: str) -> Any:
        """讀取資源內容"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.read_resource(uri)

        # 解析資源內容
        if hasattr(result, 'contents') and result.contents:
            if len(result.contents) == 1:
                content = result.contents[0]
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'blob'):
                    return content.blob
            return [
                getattr(c, 'text', getattr(c, 'blob', str(c)))
                for c in result.contents
            ]
        return None

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """列出所有可用的提示詞模板"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.list_prompts()
        return [
            {
                "name": prompt.name,
                "description": prompt.description or "",
                "arguments": getattr(prompt, 'arguments', [])
            }
            for prompt in result.prompts
        ]

    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """取得提示詞內容"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")

        result = await self.client.get_prompt(prompt_name, arguments or {})

        # 解析提示詞消息
        if hasattr(result, 'messages') and result.messages:
            return [
                {
                    "role": msg.role,
                    "content": getattr(msg.content, 'text', str(msg.content)) if hasattr(msg.content, 'text') else str(msg.content)
                }
                for msg in result.messages
            ]
        return []

    async def ping(self) -> bool:
        """測試伺服器連線"""
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")
        
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    def get_transport_info(self) -> Dict[str, Any]:
        """取得傳輸資訊"""
        if not self.client:
            return {"status": "not_connected"}
        
        transport = getattr(self.client, 'transport', None)
        if transport:
            return {
                "status": "connected",
                "transport_type": type(transport).__name__,
                "transport_info": str(transport)
            }
        return {"status": "unknown"}
