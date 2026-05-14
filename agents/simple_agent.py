


from typing import Optional, Iterator, TYPE_CHECKING
import re

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

class SimpleAgent(Agent):
    """
    負責在 agents.simple_agent 中封裝 SimpleAgent，封裝代理節點的推理、工具使用、訊息傳遞或協作控制邏輯。
    
    Args:
        name: 此流程需要使用的輸入資料。
        llm: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        system_prompt: 此流程需要使用的輸入資料。
        config: 控制此流程行為的設定資料。
        tool_registry: 此流程需要使用的輸入資料。
        enable_tool_calling: 控制是否啟用此項資料、功能或處理分支的布林開關。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
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
        負責執行 SimpleAgent 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            llm: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            system_prompt: 此流程需要使用的輸入資料。
            config: 控制此流程行為的設定資料。
            tool_registry: 此流程需要使用的輸入資料。
            enable_tool_calling: 控制是否啟用此項資料、功能或處理分支的布林開關。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
    
    def _get_enhanced_system_prompt(self) -> str:
        """
        負責執行 SimpleAgent 中的 _get_enhanced_system_prompt 流程，依照 SimpleAgent 的流程需求處理 _get_enhanced_system_prompt 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 SimpleAgent 中的 _parse_tool_calls 流程，依照 SimpleAgent 的流程需求處理 _parse_tool_calls 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 SimpleAgent 中的 _execute_tool_call 流程，依照 SimpleAgent 的流程需求處理 _execute_tool_call 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            parameters: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 SimpleAgent 中的 _parse_tool_parameters 流程，依照 SimpleAgent 的流程需求處理 _parse_tool_parameters 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            parameters: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        負責執行 SimpleAgent 中的 _convert_parameter_types 流程，依照 SimpleAgent 的流程需求處理 _convert_parameter_types 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            param_dict: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 SimpleAgent 中的 _infer_action 流程，依照 SimpleAgent 的流程需求處理 _infer_action 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            param_dict: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 SimpleAgent 中的 _infer_simple_parameters 流程，依照 SimpleAgent 的流程需求處理 _infer_simple_parameters 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            parameters: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if tool_name == 'rag':
            return {'action': 'search', 'query': parameters}
        elif tool_name == 'memory':
            return {'action': 'search', 'query': parameters}
        else:
            return {'input': parameters}

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        負責執行 SimpleAgent 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            input_text: 此流程需要使用的輸入資料。
            max_tool_iterations: 控制檢索、篩選或輸出數量的數值參數。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        負責執行 SimpleAgent 中的 add_tool 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            tool: 可呼叫的工具、工具名稱或工具註冊表。
            auto_expand: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 直接使用 ToolRegistry 的 register_tool 方法
        # ToolRegistry 會自動處理工具展開
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """
        負責執行 SimpleAgent 中的 remove_tool 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.tool_registry:
            return self.tool_registry.unregister_tool(tool_name)
        return False

    def list_tools(self) -> list:
        """
        負責執行 SimpleAgent 中的 list_tools 流程，依照 SimpleAgent 的流程需求處理 list_tools 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        """
        負責執行 SimpleAgent 中的 has_tools 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        負責執行 SimpleAgent 中的 stream_run 流程，依照 SimpleAgent 的流程需求處理 stream_run 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            input_text: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Iterator[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
