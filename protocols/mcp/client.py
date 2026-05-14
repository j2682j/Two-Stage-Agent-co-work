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
    """
    負責在 protocols.mcp.client 中封裝 MCPClient，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        server_source: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        server_args: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        transport_type: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        env: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        **transport_kwargs: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self,
                 server_source: Union[str, List[str], FastMCP, Dict[str, Any]],
                 server_args: Optional[List[str]] = None,
                 transport_type: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None,
                 **transport_kwargs):
        """
        負責執行 MCPClient 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            server_source: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            server_args: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            transport_type: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            env: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            **transport_kwargs: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 MCPClient 中的 _prepare_server_source 流程，依照 MCPClient 的流程需求處理 _prepare_server_source 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            server_source: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        
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
        """
        負責執行 MCPClient 中的 _create_transport_from_config 流程，依照 MCPClient 的流程需求處理 _create_transport_from_config 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            config: 控制此流程行為的設定資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 MCPClient 中的 __aenter__ 流程，依照 MCPClient 的流程需求處理 __aenter__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        print("🔗 連線到 MCP 伺服器...")
        self.client = Client(self.server_source)
        self._context_manager = self.client
        await self._context_manager.__aenter__()
        print("✅ 連線成功！")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        負責執行 MCPClient 中的 __aexit__ 流程，依照 MCPClient 的流程需求處理 __aexit__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            exc_type: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            exc_val: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            exc_tb: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self._context_manager:
            await self._context_manager.__aexit__(exc_type, exc_val, exc_tb)
            self.client = None
            self._context_manager = None
        print("🔌 連線已斷開")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        負責執行 MCPClient 中的 list_tools 流程，依照 MCPClient 的流程需求處理 list_tools 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 MCPClient 中的 call_tool 流程，呼叫模型、工具或外部服務並整理回傳結果。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            arguments: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Any。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 MCPClient 中的 list_resources 流程，依照 MCPClient 的流程需求處理 list_resources 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 MCPClient 中的 read_resource 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
        
        Args:
            uri: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Any。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 MCPClient 中的 list_prompts 流程，依照 MCPClient 的流程需求處理 list_prompts 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 MCPClient 中的 get_prompt 流程，依照 MCPClient 的流程需求處理 get_prompt 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            prompt_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            arguments: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 MCPClient 中的 ping 流程，依照 MCPClient 的流程需求處理 ping 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.client:
            raise RuntimeError("Client not connected. Use 'async with client:' context manager.")
        
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    def get_transport_info(self) -> Dict[str, Any]:
        """
        負責執行 MCPClient 中的 get_transport_info 流程，依照 MCPClient 的流程需求處理 get_transport_info 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
