"""
MCP工具包裝器 - 將單個MCP工具包裝成Agent System Tool

這個模組將MCP伺服器的每個工具展開為獨立的Agent System Tool對象，
使得Agent可以像呼叫普通工具一樣呼叫MCP工具。
"""

from typing import Dict, Any, Optional, List
from ..base import Tool, ToolParameter


class MCPWrappedTool(Tool):
    """
    MCP工具包裝器 - 將單個MCP工具包裝成Agent System Tool
    
    這個類將MCP伺服器的一個工具（如 read_file）包裝成一個獨立的Tool對象。
    Agent呼叫時只需提供參數，無需了解MCP的內部結構。
    
    範例：
        >>> # 內部使用，由MCPTool自動建立
        >>> wrapped_tool = MCPWrappedTool(
        ...     mcp_tool=mcp_tool_instance,
        ...     tool_info={
        ...         "name": "read_file",
        ...         "description": "Read a file...",
        ...         "input_schema": {...}
        ...     }
        ... )
    """
    
    def __init__(self,
                 mcp_tool: 'MCPTool',  # type: ignore
                 tool_info: Dict[str, Any],
                 prefix: str = ""):
        """
        初始化MCP包裝工具

        Args:
            mcp_tool: 父MCP工具實例
            tool_info: MCP工具資訊（包含name, description, input_schema）
            prefix: 工具名前綴（如 "filesystem_"）
        """
        self.mcp_tool = mcp_tool
        self.tool_info = tool_info
        self.mcp_tool_name = tool_info.get('name', 'unknown')

        # 建構工具名：prefix + mcp_tool_name
        tool_name = f"{prefix}{self.mcp_tool_name}" if prefix else self.mcp_tool_name

        # 取得描述
        description = tool_info.get('description', f'MCP工具: {self.mcp_tool_name}')

        # 解析參數schema
        self._parameters = self._parse_input_schema(tool_info.get('input_schema', {}))

        # 初始化父類
        super().__init__(
            name=tool_name,
            description=description
        )
    
    def _parse_input_schema(self, input_schema: Dict[str, Any]) -> List[ToolParameter]:
        """
        將MCP的input_schema轉換為HelloAgents的ToolParameter列表

        Args:
            input_schema: MCP工具的input_schema（JSON Schema格式）

        Returns:
            ToolParameter列表
        """
        parameters = []

        properties = input_schema.get('properties', {})
        required_fields = input_schema.get('required', [])

        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'string')
            param_desc = param_info.get('description', '')
            is_required = param_name in required_fields

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,  # 直接使用JSON Schema的類型字串
                description=param_desc,
                required=is_required
            ))

        return parameters
    
    def get_parameters(self) -> List[ToolParameter]:
        """
        取得工具參數定義

        Returns:
            ToolParameter列表
        """
        return self._parameters

    def run(self, params: Dict[str, Any]) -> str:
        """
        執行MCP工具

        Args:
            params: 工具參數（直接傳遞給MCP工具）

        Returns:
            執行結果
        """
        # 建構MCP呼叫參數
        mcp_params = {
            "action": "call_tool",
            "tool_name": self.mcp_tool_name,
            "arguments": params
        }

        # 呼叫父MCP工具
        return self.mcp_tool.run(mcp_params)

e