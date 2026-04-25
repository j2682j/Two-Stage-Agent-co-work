"""
協議工具集合

提供基於協議實現的工具介面：
- MCP Tool: 基於 fastmcp 庫，用於連線和呼叫 MCP 伺服器
- A2A Tool: 基於官方 a2a 庫，用於 Agent 間通信（需要安裝 a2a）
- ANP Tool: 基於概念實現，用於服務發現和網路管理
"""

from typing import Dict, Any, List, Optional
from ..base import Tool, ToolParameter
import os


# MCP伺服器環境變數映射表
# 用於自動檢測常見MCP伺服器需要的環境變數
MCP_SERVER_ENV_MAP = {
    "server-github": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
    "server-slack": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
    "server-google-drive": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"],
    "server-postgres": ["POSTGRES_CONNECTION_STRING"],
    "server-sqlite": [],  # 不需要環境變數
    "server-filesystem": [],  # 不需要環境變數
}


class MCPTool(Tool):
    """MCP (Model Context Protocol) 工具

    連線到 MCP 伺服器並呼叫其提供的工具、資源和提示詞。
    
    功能：
    - 列出伺服器提供的工具
    - 呼叫伺服器工具
    - 讀取伺服器資源
    - 取得提示詞模板

    使用範例:
        >>> from hello_agents.tools.builtin import MCPTool
        >>>
        >>> # 方式1: 使用內建演示伺服器
        >>> tool = MCPTool()  # 自動建立內建伺服器
        >>> result = tool.run({"action": "list_tools"})
        >>>
        >>> # 方式2: 連線到外部 MCP 伺服器
        >>> tool = MCPTool(server_command=["python", "examples/mcp_example.py"])
        >>> result = tool.run({"action": "list_tools"})
        >>>
        >>> # 方式3: 使用自定義 FastMCP 伺服器
        >>> from fastmcp import FastMCP
        >>> server = FastMCP("MyServer")
        >>> tool = MCPTool(server=server)

    注意：使用 fastmcp 庫，已包含在依賴中
    """
    
    def __init__(self,
                 name: str = "mcp",
                 description: Optional[str] = None,
                 server_command: Optional[List[str]] = None,
                 server_args: Optional[List[str]] = None,
                 server: Optional[Any] = None,
                 auto_expand: bool = True,
                 env: Optional[Dict[str, str]] = None,
                 env_keys: Optional[List[str]] = None):
        """
        初始化 MCP 工具

        Args:
            name: 工具名稱（預設為"mcp"，建議為不同伺服器指定不同名稱）
            description: 工具描述（可選，預設為通用描述）
            server_command: 伺服器啟動命令（如 ["python", "server.py"]）
            server_args: 伺服器參數列表
            server: FastMCP 伺服器實例（可選，用於記憶體傳輸）
            auto_expand: 是否自動展開為獨立工具（預設True）
            env: 環境變數字典（優先順序最高，直接傳遞給MCP伺服器）
            env_keys: 要從系統環境變數載入的key列表（優先順序中等）

        環境變數優先順序（從高到低）：
            1. 直接傳遞的env參數
            2. env_keys指定的環境變數
            3. 自動檢測的環境變數（根據server_command）

        注意：如果所有參數都為空，將建立內建演示伺服器

        範例：
            >>> # 方式1：直接傳遞環境變數（優先順序最高）
            >>> github_tool = MCPTool(
            ...     name="github",
            ...     server_command=["npx", "-y", "@modelcontextprotocol/server-github"],
            ...     env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"}
            ... )
            >>>
            >>> # 方式2：從.env檔案載入指定的環境變數
            >>> github_tool = MCPTool(
            ...     name="github",
            ...     server_command=["npx", "-y", "@modelcontextprotocol/server-github"],
            ...     env_keys=["GITHUB_PERSONAL_ACCESS_TOKEN"]
            ... )
            >>>
            >>> # 方式3：自動檢測（最簡單，推薦）
            >>> github_tool = MCPTool(
            ...     name="github",
            ...     server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
            ...     # 自動從環境變數載入GITHUB_PERSONAL_ACCESS_TOKEN
            ... )
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self.server = server
        self._client = None
        self._available_tools = []
        self.auto_expand = auto_expand
        self.prefix = f"{name}_" if auto_expand else ""

        # 環境變數處理（優先順序：env > env_keys > 自動檢測）
        self.env = self._prepare_env(env, env_keys, server_command)

        # 如果沒有指定任何伺服器，建立內建演示伺服器
        if not server_command and not server:
            self.server = self._create_builtin_server()

        # 自動發現工具
        self._discover_tools()

        # 設定預設描述或自動生成
        if description is None:
            description = self._generate_description()

        super().__init__(
            name=name,
            description=description
        )

    def _prepare_env(self,
                     env: Optional[Dict[str, str]],
                     env_keys: Optional[List[str]],
                     server_command: Optional[List[str]]) -> Dict[str, str]:
        """
        準備環境變數

        優先順序：env > env_keys > 自動檢測

        Args:
            env: 直接傳遞的環境變數字典
            env_keys: 要從系統環境變數載入的key列表
            server_command: 伺服器命令（用於自動檢測）

        Returns:
            合併後的環境變數字典
        """
        result_env = {}

        # 1. 自動檢測（優先順序最低）
        if server_command:
            # 從命令中提取伺服器名稱
            server_name = None
            for part in server_command:
                if "server-" in part:
                    # 提取類似 "@modelcontextprotocol/server-github" 中的 "server-github"
                    server_name = part.split("/")[-1] if "/" in part else part
                    break

            # 查找映射表
            if server_name and server_name in MCP_SERVER_ENV_MAP:
                auto_keys = MCP_SERVER_ENV_MAP[server_name]
                for key in auto_keys:
                    value = os.getenv(key)
                    if value:
                        result_env[key] = value
                        print(f"🔑 自動載入環境變數: {key}")

        # 2. env_keys指定的環境變數（優先順序中等）
        if env_keys:
            for key in env_keys:
                value = os.getenv(key)
                if value:
                    result_env[key] = value
                    print(f"🔑 從env_keys載入環境變數: {key}")
                else:
                    print(f"[WARN] 環境變數 {key} 未設定")

        # 3. 直接傳遞的env（優先順序最高）
        if env:
            result_env.update(env)
            for key in env.keys():
                print(f"🔑 使用直接傳遞的環境變數: {key}")

        return result_env

    def _create_builtin_server(self):
        """建立內建演示伺服器"""
        try:
            from fastmcp import FastMCP

            server = FastMCP("HelloAgents-BuiltinServer")

            @server.tool()
            def add(a: float, b: float) -> float:
                """加法計算器"""
                return a + b

            @server.tool()
            def subtract(a: float, b: float) -> float:
                """減法計算器"""
                return a - b

            @server.tool()
            def multiply(a: float, b: float) -> float:
                """乘法計算器"""
                return a * b

            @server.tool()
            def divide(a: float, b: float) -> float:
                """除法計算器"""
                if b == 0:
                    raise ValueError("除數不能為零")
                return a / b

            @server.tool()
            def greet(name: str = "World") -> str:
                """友好問候"""
                return f"Hello, {name}! 歡迎使用 HelloAgents MCP 工具！"

            @server.tool()
            def get_system_info() -> dict:
                """取得系統資訊"""
                import platform
                import sys
                return {
                    "platform": platform.system(),
                    "python_version": sys.version,
                    "server_name": "HelloAgents-BuiltinServer",
                    "tools_count": 6
                }

            return server

        except ImportError:
            raise ImportError(
                "建立內建 MCP 伺服器需要 fastmcp 庫。請安裝: pip install fastmcp"
            )

    def _discover_tools(self):
        """發現MCP伺服器提供的所有工具"""
        try:
            from hello_agents.protocols.mcp.client import MCPClient
            import asyncio

            async def discover():
                client_source = self.server if self.server else self.server_command
                async with MCPClient(client_source, self.server_args, env=self.env) as client:
                    tools = await client.list_tools()
                    return tools

            # 執行非同步發現
            try:
                loop = asyncio.get_running_loop()
                # 如果已有循環，在新執行緒中執行
                import concurrent.futures
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(discover())
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    self._available_tools = future.result()
            except RuntimeError:
                # 沒有執行中的循環
                self._available_tools = asyncio.run(discover())

        except Exception as e:
            # 工具發現失敗不影響初始化
            self._available_tools = []

    def _generate_description(self) -> str:
        """生成增強的工具描述"""
        if not self._available_tools:
            return "連線到 MCP 伺服器，呼叫工具、讀取資源和取得提示詞。支援內建伺服器和外部伺服器。"

        if self.auto_expand:
            # 展開模式：簡單描述
            return f"MCP工具伺服器，包含{len(self._available_tools)}個工具。這些工具會自動展開為獨立的工具供Agent使用。"
        else:
            # 非展開模式：詳細描述
            desc_parts = [
                f"MCP工具伺服器，提供{len(self._available_tools)}個工具："
            ]

            # 列出所有工具
            for tool in self._available_tools:
                tool_name = tool.get('name', 'unknown')
                tool_desc = tool.get('description', '無描述')
                # 簡化描述，只取第一句
                short_desc = tool_desc.split('.')[0] if tool_desc else '無描述'
                desc_parts.append(f"  • {tool_name}: {short_desc}")

            # 添加呼叫格式說明
            desc_parts.append("\n呼叫格式：回傳JSON格式的參數")
            desc_parts.append('{"action": "call_tool", "tool_name": "工具名", "arguments": {...}}')

            # 添加範例
            if self._available_tools:
                first_tool = self._available_tools[0]
                tool_name = first_tool.get('name', 'example')
                desc_parts.append(f'\n範例：{{"action": "call_tool", "tool_name": "{tool_name}", "arguments": {{...}}}}')

            return "\n".join(desc_parts)

    def get_expanded_tools(self) -> List['Tool']:  # type: ignore
        """
        取得展開的工具列表

        將MCP伺服器的每個工具包裝成獨立的Tool對象

        Returns:
            Tool對象列表
        """
        if not self.auto_expand:
            return []

        from .mcp_wrapper_tool import MCPWrappedTool

        expanded_tools = []
        for tool_info in self._available_tools:
            wrapped_tool = MCPWrappedTool(
                mcp_tool=self,
                tool_info=tool_info,
                prefix=self.prefix
            )
            expanded_tools.append(wrapped_tool)

        return expanded_tools

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        執行 MCP 操作

        Args:
            parameters: 包含以下參數的字典
                - action: 操作類型 (list_tools, call_tool, list_resources, read_resource, list_prompts, get_prompt)
                  如果不指定action但指定了tool_name，會自動推斷為call_tool
                - tool_name: 工具名稱（call_tool 需要）
                - arguments: 工具參數（call_tool 需要）
                - uri: 資源 URI（read_resource 需要）
                - prompt_name: 提示詞名稱（get_prompt 需要）
                - prompt_arguments: 提示詞參數（get_prompt 可選）

        Returns:
            操作結果
        """
        from hello_agents.protocols.mcp.client import MCPClient

        # 智慧推斷action：如果沒有action但有tool_name，自動設定為call_tool
        action = parameters.get("action", "").lower()
        if not action and "tool_name" in parameters:
            action = "call_tool"
            parameters["action"] = action

        if not action:
            return "錯誤：必須指定 action 參數或 tool_name 參數"
        
        try:
            # 使用增強的非同步客戶端
            import asyncio
            from hello_agents.protocols.mcp.client import MCPClient

            async def run_mcp_operation():
                # 根據設定選擇客戶端建立方式
                if self.server:
                    # 使用內建伺服器（記憶體傳輸）
                    client_source = self.server
                else:
                    # 使用外部伺服器命令
                    client_source = self.server_command

                async with MCPClient(client_source, self.server_args, env=self.env) as client:
                    if action == "list_tools":
                        tools = await client.list_tools()
                        if not tools:
                            return "沒有找到可用的工具"
                        result = f"找到 {len(tools)} 個工具:\n"
                        for tool in tools:
                            result += f"- {tool['name']}: {tool['description']}\n"
                        return result

                    elif action == "call_tool":
                        tool_name = parameters.get("tool_name")
                        arguments = parameters.get("arguments", {})
                        if not tool_name:
                            return "錯誤：必須指定 tool_name 參數"
                        result = await client.call_tool(tool_name, arguments)
                        return f"工具 '{tool_name}' 執行結果:\n{result}"

                    elif action == "list_resources":
                        resources = await client.list_resources()
                        if not resources:
                            return "沒有找到可用的資源"
                        result = f"找到 {len(resources)} 個資源:\n"
                        for resource in resources:
                            result += f"- {resource['uri']}: {resource['name']}\n"
                        return result

                    elif action == "read_resource":
                        uri = parameters.get("uri")
                        if not uri:
                            return "錯誤：必須指定 uri 參數"
                        content = await client.read_resource(uri)
                        return f"資源 '{uri}' 內容:\n{content}"

                    elif action == "list_prompts":
                        prompts = await client.list_prompts()
                        if not prompts:
                            return "沒有找到可用的提示詞"
                        result = f"找到 {len(prompts)} 個提示詞:\n"
                        for prompt in prompts:
                            result += f"- {prompt['name']}: {prompt['description']}\n"
                        return result

                    elif action == "get_prompt":
                        prompt_name = parameters.get("prompt_name")
                        prompt_arguments = parameters.get("prompt_arguments", {})
                        if not prompt_name:
                            return "錯誤：必須指定 prompt_name 參數"
                        messages = await client.get_prompt(prompt_name, prompt_arguments)
                        result = f"提示詞 '{prompt_name}':\n"
                        for msg in messages:
                            result += f"[{msg['role']}] {msg['content']}\n"
                        return result

                    else:
                        return f"錯誤：不支援的操作 '{action}'"

            # 執行非同步操作
            try:
                # 檢查是否已有執行中的事件循環
                try:
                    loop = asyncio.get_running_loop()
                    # 如果有執行中的循環，在新執行緒中執行新的事件循環
                    import concurrent.futures
                    import threading

                    def run_in_thread():
                        # 在新執行緒中建立新的事件循環
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(run_mcp_operation())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        return future.result()
                except RuntimeError:
                    # 沒有執行中的循環，直接執行
                    return asyncio.run(run_mcp_operation())
            except Exception as e:
                return f"非同步操作失敗: {str(e)}"
                    
        except Exception as e:
            return f"MCP 操作失敗: {str(e)}"
    
    def get_parameters(self) -> List[ToolParameter]:
        """取得工具參數定義"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作類型: list_tools, call_tool, list_resources, read_resource, list_prompts, get_prompt",
                required=True
            ),
            ToolParameter(
                name="tool_name",
                type="string",
                description="工具名稱（call_tool 操作需要）",
                required=False
            ),
            ToolParameter(
                name="arguments",
                type="object",
                description="工具參數（call_tool 操作需要）",
                required=False
            ),
            ToolParameter(
                name="uri",
                type="string",
                description="資源 URI（read_resource 操作需要）",
                required=False
            ),
            ToolParameter(
                name="prompt_name",
                type="string",
                description="提示詞名稱（get_prompt 操作需要）",
                required=False
            ),
            ToolParameter(
                name="prompt_arguments",
                type="object",
                description="提示詞參數（get_prompt 操作可選）",
                required=False
            )
        ]


class A2ATool(Tool):
    """A2A (Agent-to-Agent Protocol) 工具

    連線到 A2A Agent 並進行通信。
    
    功能：
    - 向 Agent 提問
    - 取得 Agent 資訊
    - 發送自定義消息

    使用範例:
        >>> from hello_agents.tools.builtin import A2ATool
        >>> # 連線到 A2A Agent（使用預設名稱）
        >>> tool = A2ATool(agent_url="http://localhost:5000")
        >>> # 連線到 A2A Agent（自定義名稱和描述）
        >>> tool = A2ATool(
        ...     agent_url="http://localhost:5000",
        ...     name="tech_expert",
        ...     description="技術專家，回答技術相關問題"
        ... )
        >>> # 提問
        >>> result = tool.run({"action": "ask", "question": "計算 2+2"})
        >>> # 取得資訊
        >>> result = tool.run({"action": "get_info"})
    
    注意：需要安裝官方 a2a-sdk 庫: pip install a2a-sdk
    詳見文檔: docs/chapter10/A2A_GUIDE.md
    官方倉庫: https://github.com/a2aproject/a2a-python
    """
    
    def __init__(self, agent_url: str, name: str = "a2a", description: str = None):
        """
        初始化 A2A 工具

        Args:
            agent_url: Agent URL
            name: 工具名稱（可選，預設為 "a2a"）
            description: 工具描述（可選）
        """
        if description is None:
            description = "連線到 A2A Agent，支援提問和取得資訊。需要安裝官方 a2a-sdk 庫。"

        super().__init__(
            name=name,
            description=description
        )
        self.agent_url = agent_url
        
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        執行 A2A 操作
        
        Args:
            parameters: 包含以下參數的字典
                - action: 操作類型 (ask, get_info)
                - question: 問題文字（ask 需要）
        
        Returns:
            操作結果
        """
        try:
            from hello_agents.protocols.a2a.implementation import A2AClient, A2A_AVAILABLE
            if not A2A_AVAILABLE:
                return ("錯誤：需要安裝 a2a-sdk 庫\n"
                       "安裝命令: pip install a2a-sdk\n"
                       "詳見文檔: docs/chapter10/A2A_GUIDE.md\n"
                       "官方倉庫: https://github.com/a2aproject/a2a-python")
        except ImportError:
            return ("錯誤：無法匯入 A2A 模組\n"
                   "安裝命令: pip install a2a-sdk\n"
                   "詳見文檔: docs/chapter10/A2A_GUIDE.md\n"
                   "官方倉庫: https://github.com/a2aproject/a2a-python")

        action = parameters.get("action", "").lower()
        
        if not action:
            return "錯誤：必須指定 action 參數"
        
        try:
            client = A2AClient(self.agent_url)
            
            if action == "ask":
                question = parameters.get("question")
                if not question:
                    return "錯誤：必須指定 question 參數"
                response = client.ask(question)
                return f"Agent 回答:\n{response}"
                
            elif action == "get_info":
                info = client.get_info()
                result = "Agent 資訊:\n"
                for key, value in info.items():
                    result += f"- {key}: {value}\n"
                return result
                
            else:
                return f"錯誤：不支援的操作 '{action}'"
                
        except Exception as e:
            return f"A2A 操作失敗: {str(e)}"
    
    def get_parameters(self) -> List[ToolParameter]:
        """取得工具參數定義"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作類型: ask(提問), get_info(取得資訊)",
                required=True
            ),
            ToolParameter(
                name="question",
                type="string",
                description="問題文字（ask 操作需要）",
                required=False
            )
        ]


class ANPTool(Tool):
    """ANP (Agent Network Protocol) 工具

    提供智慧代理網路管理功能，包括服務發現、節點管理和消息路由。
    這是一個概念性實現，用於演示 Agent 網路管理的核心理念。
    
    功能：
    - 註冊和發現服務
    - 添加和管理網路節點
    - 消息路由
    - 網路統計

    使用範例:
        >>> from hello_agents.tools.builtin import ANPTool
        >>> tool = ANPTool()
        >>> # 註冊服務
        >>> result = tool.run({
        ...     "action": "register_service",
        ...     "service_id": "calc-1",
        ...     "service_type": "calculator",
        ...     "endpoint": "http://localhost:5001"
        ... })
        >>> # 發現服務
        >>> result = tool.run({
        ...     "action": "discover_services",
        ...     "service_type": "calculator"
        ... })
        >>> # 添加節點
        >>> result = tool.run({
        ...     "action": "add_node",
        ...     "node_id": "agent-1",
        ...     "endpoint": "http://localhost:5001"
        ... })
    
    注意：這是概念性實現，不需要額外依賴
    詳見文檔: docs/chapter10/ANP_CONCEPTS.md
    """
    
    def __init__(self, name: str = "anp", description: str = None, discovery=None, network=None):
        """初始化 ANP 工具

        Args:
            name: 工具名稱
            description: 工具描述
            discovery: 可選的 ANPDiscovery 實例，如果不提供則建立新實例
            network: 可選的 ANPNetwork 實例，如果不提供則建立新實例
        """
        if description is None:
            description = "智慧代理網路管理工具，支援服務發現、節點管理和消息路由。概念性實現。"

        super().__init__(
            name=name,
            description=description
        )
        from hello_agents.protocols.anp.implementation import ANPDiscovery, ANPNetwork
        self._discovery = discovery if discovery is not None else ANPDiscovery()
        self._network = network if network is not None else ANPNetwork()
        
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        執行 ANP 操作
        
        Args:
            parameters: 包含以下參數的字典
                - action: 操作類型 (register_service, discover_services, add_node, route_message, get_stats)
                - service_id, service_type, endpoint: 服務資訊（register_service 需要）
                - node_id, endpoint: 節點資訊（add_node 需要）
                - from_node, to_node, message: 路由資訊（route_message 需要）
        
        Returns:
            操作結果
        """
        from hello_agents.protocols.anp.implementation import ServiceInfo

        action = parameters.get("action", "").lower()
        
        if not action:
            return "錯誤：必須指定 action 參數"
        
        try:
            if action == "register_service":
                service_id = parameters.get("service_id")
                service_type = parameters.get("service_type")
                endpoint = parameters.get("endpoint")
                metadata = parameters.get("metadata", {})
                
                if not all([service_id, service_type, endpoint]):
                    return "錯誤：必須指定 service_id, service_type 和 endpoint 參數"
                
                service = ServiceInfo(service_id, service_type, endpoint, metadata)
                self._discovery.register_service(service)
                return f"[OK] 已註冊服務 '{service_id}'"

            elif action == "unregister_service":
                service_id = parameters.get("service_id")
                if not service_id:
                    return "錯誤：必須指定 service_id 參數"

                # 使用 ANPDiscovery 的 unregister_service 方法
                success = self._discovery.unregister_service(service_id)

                if success:
                    return f"[OK] 已注銷服務 '{service_id}'"
                else:
                    return f"錯誤：服務 '{service_id}' 不存在"

            elif action == "discover_services":
                service_type = parameters.get("service_type")
                services = self._discovery.discover_services(service_type)

                if not services:
                    return "沒有找到服務"

                result = f"找到 {len(services)} 個服務:\n\n"
                for service in services:
                    result += f"服務ID: {service.service_id}\n"
                    result += f"  名稱: {service.service_name}\n"
                    result += f"  類型: {service.service_type}\n"
                    result += f"  端點: {service.endpoint}\n"
                    if service.capabilities:
                        result += f"  能力: {', '.join(service.capabilities)}\n"
                    if service.metadata:
                        result += f"  元資料: {service.metadata}\n"
                    result += "\n"
                return result
                
            elif action == "add_node":
                node_id = parameters.get("node_id")
                endpoint = parameters.get("endpoint")
                metadata = parameters.get("metadata", {})
                
                if not all([node_id, endpoint]):
                    return "錯誤：必須指定 node_id 和 endpoint 參數"
                
                self._network.add_node(node_id, endpoint, metadata)
                return f"[OK] 已添加節點 '{node_id}'"
                
            elif action == "route_message":
                from_node = parameters.get("from_node")
                to_node = parameters.get("to_node")
                message = parameters.get("message", {})
                
                if not all([from_node, to_node]):
                    return "錯誤：必須指定 from_node 和 to_node 參數"
                
                path = self._network.route_message(from_node, to_node, message)
                if path:
                    return f"消息路由路徑: {' -> '.join(path)}"
                else:
                    return "無法找到路由路徑"
                
            elif action == "get_stats":
                stats = self._network.get_network_stats()
                result = "網路統計:\n"
                for key, value in stats.items():
                    result += f"- {key}: {value}\n"
                return result
                
            else:
                return f"錯誤：不支援的操作 '{action}'"
                
        except Exception as e:
            return f"ANP 操作失敗: {str(e)}"
    
    def get_parameters(self) -> List[ToolParameter]:
        """取得工具參數定義"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作類型: register_service, unregister_service, discover_services, add_node, route_message, get_stats",
                required=True
            ),
            ToolParameter(
                name="service_id",
                type="string",
                description="服務 ID（register_service, unregister_service 需要）",
                required=False
            ),
            ToolParameter(
                name="service_type",
                type="string",
                description="服務類型（register_service 需要）",
                required=False
            ),
            ToolParameter(
                name="endpoint",
                type="string",
                description="端點地址（register_service, add_node 需要）",
                required=False
            ),
            ToolParameter(
                name="node_id",
                type="string",
                description="節點 ID（add_node 需要）",
                required=False
            ),
            ToolParameter(
                name="from_node",
                type="string",
                description="源節點 ID（route_message 需要）",
                required=False
            ),
            ToolParameter(
                name="to_node",
                type="string",
                description="目標節點 ID（route_message 需要）",
                required=False
            ),
            ToolParameter(
                name="message",
                type="object",
                description="消息內容（route_message 需要）",
                required=False
            ),
            ToolParameter(
                name="metadata",
                type="object",
                description="元資料（register_service, add_node 可選）",
                required=False
            )
        ]

