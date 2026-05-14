"""
MCP工具包裝器 - 將單個MCP工具包裝成Agent System Tool

這個模組將MCP伺服器的每個工具展開為獨立的Agent System Tool對象，
使得Agent可以像呼叫普通工具一樣呼叫MCP工具。
"""

from typing import Dict, Any, Optional, List
from ..base import Tool, ToolParameter


class MCPWrappedTool(Tool):
    """
    負責在 tools.builtin.mcp_wrapper_tool 中封裝 MCPWrappedTool，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        mcp_tool: 此流程需要使用的輸入資料。
        tool_info: 此流程需要使用的輸入資料。
        prefix: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(self,
                 mcp_tool: 'MCPTool',  # type: ignore
                 tool_info: Dict[str, Any],
                 prefix: str = ""):
        """
        負責執行 MCPWrappedTool 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            mcp_tool: 此流程需要使用的輸入資料。
            tool_info: 此流程需要使用的輸入資料。
            prefix: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        負責執行 MCPWrappedTool 中的 _parse_input_schema 流程，依照 MCPWrappedTool 的流程需求處理 _parse_input_schema 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            input_schema: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[ToolParameter]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        負責執行 MCPWrappedTool 中的 get_parameters 流程，依照 MCPWrappedTool 的流程需求處理 get_parameters 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[ToolParameter]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self._parameters

    def run(self, params: Dict[str, Any]) -> str:
        """
        負責執行 MCPWrappedTool 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            params: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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