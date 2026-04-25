"""???????"""


from typing import Optional, Iterator, TYPE_CHECKING
import re

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

class SimpleAgent(Agent):
    """簡單的對話Agent，支援可選的工具呼叫"""
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True
    ):
        """
        初始化SimpleAgent
        
        Args:
            name: Agent名稱
            llm: LLM實例
            system_prompt: 系統提示詞
            config: 設定對象
            tool_registry: 工具註冊表（可選，如果提供則啟用工具呼叫）
            enable_tool_calling: 是否啟用工具呼叫（只有在提供tool_registry時生效）
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
    
    def _get_enhanced_system_prompt(self) -> str:
        """建構增強的系統提示詞，包含工具資訊"""
        base_prompt = self.system_prompt or "你是一個有用的AI 助理。"
        
        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt
        
        # 取得工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暫無可用工具":
            return base_prompt
        
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具來幫助回答問題：\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具呼叫格式\n"
        tools_section += "當需要使用工具時，請使用以下格式：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"

        tools_section += "### 參數格式說明\n"
        tools_section += "1. **多個參數**：使用 `key=value` 格式，用逗號分隔\n"
        tools_section += "   範例：`[TOOL_CALL:calculator_multiply:a=12,b=8]`\n"
        tools_section += "   範例：`[TOOL_CALL:filesystem_read_file:path=README.md]`\n\n"
        tools_section += "2. **單個參數**：直接使用 `key=value`\n"
        tools_section += "   範例：`[TOOL_CALL:search:query=Python編程]`\n\n"
        tools_section += "3. **簡單查詢**：可以直接傳入文字\n"
        tools_section += "   範例：`[TOOL_CALL:search:Python編程]`\n\n"

        tools_section += "### 重要提示\n"
        tools_section += "- 參數名必須與工具定義的參數名完全匹配\n"
        tools_section += "- 數字參數直接寫數字，不需要引號：`a=12` 而不是 `a=\"12\"`\n"
        tools_section += "- 檔案路徑等字串參數直接寫：`path=README.md`\n"
        tools_section += "- 工具呼叫結果會自動插入到對話中，然後你可以基於結果繼續回答\n"

        return base_prompt + tools_section
    
    def _parse_tool_calls(self, text: str) -> list:
        """解析文字中的工具呼叫"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })
        
        return tool_calls
    
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """執行工具呼叫"""
        if not self.tool_registry:
            return f"❌ 錯誤：未設定工具註冊表"

        try:
            # 取得Tool對象
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"❌ 錯誤：找不到工具 '{tool_name}'"

            # 智慧參數解析
            param_dict = self._parse_tool_parameters(tool_name, parameters)

            # 呼叫工具
            result = tool.run(param_dict)
            return f"🔧 工具 {tool_name} 執行結果：\n{result}"

        except Exception as e:
            return f"❌ 工具呼叫失敗：{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智慧解析工具參數"""
        import json
        param_dict = {}

        # 嘗試解析JSON格式
        if parameters.strip().startswith('{'):
            try:
                param_dict = json.loads(parameters)
                # JSON解析成功，進行類型轉換
                param_dict = self._convert_parameter_types(tool_name, param_dict)
                return param_dict
            except json.JSONDecodeError:
                # JSON解析失敗，繼續使用其他方式
                pass

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多個參數：action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 單個參數：key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()

            # 類型轉換
            param_dict = self._convert_parameter_types(tool_name, param_dict)

            # 智慧推斷action（如果沒有指定）
            if 'action' not in param_dict:
                param_dict = self._infer_action(tool_name, param_dict)
        else:
            # 直接傳入參數，根據工具類型智慧推斷
            param_dict = self._infer_simple_parameters(tool_name, parameters)

        return param_dict

    def _convert_parameter_types(self, tool_name: str, param_dict: dict) -> dict:
        """
        根據工具的參數定義轉換參數類型

        Args:
            tool_name: 工具名稱
            param_dict: 參數字典

        Returns:
            類型轉換後的參數字典
        """
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        # 取得工具的參數定義
        try:
            tool_params = tool.get_parameters()
        except:
            return param_dict

        # 建立參數類型映射
        param_types = {}
        for param in tool_params:
            param_types[param.name] = param.type

        # 轉換參數類型
        converted_dict = {}
        for key, value in param_dict.items():
            if key in param_types:
                param_type = param_types[key]
                try:
                    if param_type == 'number' or param_type == 'integer':
                        # 轉換為數字
                        if isinstance(value, str):
                            converted_dict[key] = float(value) if param_type == 'number' else int(value)
                        else:
                            converted_dict[key] = value
                    elif param_type == 'boolean':
                        # 轉換為布爾值
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ('true', '1', 'yes')
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = value
                except (ValueError, TypeError):
                    # 轉換失敗，保持原值
                    converted_dict[key] = value
            else:
                converted_dict[key] = value

        return converted_dict

    def _infer_action(self, tool_name: str, param_dict: dict) -> dict:
        """根據工具類型和參數推斷action"""
        if tool_name == 'memory':
            if 'recall' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('recall')
            elif 'store' in param_dict:
                param_dict['action'] = 'add'
                param_dict['content'] = param_dict.pop('store')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'content' in param_dict:
                param_dict['action'] = 'add'
        elif tool_name == 'rag':
            if 'search' in param_dict:
                param_dict['action'] = 'search'
                param_dict['query'] = param_dict.pop('search')
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'text' in param_dict:
                param_dict['action'] = 'add_text'

        return param_dict

    def _infer_simple_parameters(self, tool_name: str, parameters: str) -> dict:
        """為簡單參數推斷完整的參數字典"""
        if tool_name == 'rag':
            return {'action': 'search', 'query': parameters}
        elif tool_name == 'memory':
            return {'action': 'search', 'query': parameters}
        else:
            return {'input': parameters}

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        執行SimpleAgent，支援可選的工具呼叫
        
        Args:
            input_text: 使用者輸入
            max_tool_iterations: 最大工具呼叫迭代次數（僅在啟用工具時有效）
            **kwargs: 其他參數
            
        Returns:
            Agent回應
        """
        # 建構消息列表
        messages = []
        
        # 添加系統消息（可能包含工具資訊）
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})
        
        # 添加歷史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # 添加目前使用者消息
        messages.append({"role": "user", "content": input_text})
        
        # 如果沒有啟用工具呼叫，使用原有邏輯
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response
        
        # 迭代處理，支援多輪工具呼叫
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # 呼叫LLM
            response = self.llm.invoke(messages, **kwargs)

            # 檢查是否有工具呼叫
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # 執行所有工具呼叫並收集結果
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 從回應中移除工具呼叫標記
                    clean_response = clean_response.replace(call['original'], "")

                # 建構包含工具結果的消息
                messages.append({"role": "assistant", "content": clean_response})

                # 添加工具結果
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"工具執行結果：\n{tool_results_text}\n\n請基於這些結果給出完整的回答。"})

                current_iteration += 1
                continue

            # 沒有工具呼叫，這是最終回答
            final_response = response
            break

        # 如果超過最大迭代次數，取得最後一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)
        
        # 保存到歷史紀錄
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        return final_response

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """
        添加工具到Agent（便利方法）

        Args:
            tool: Tool對象
            auto_expand: 是否自動展開可展開的工具（預設True）

        如果工具是可展開的（expandable=True），會自動展開為多個獨立工具
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 直接使用 ToolRegistry 的 register_tool 方法
        # ToolRegistry 會自動處理工具展開
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
        return False

    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """檢查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式執行Agent
        
        Args:
            input_text: 使用者輸入
            **kwargs: 其他參數
            
        Yields:
            Agent回應片段
        """
        # 建構消息列表
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": input_text})
        
        # 流式呼叫LLM
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        
        # 保存完整對話到歷史紀錄
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
