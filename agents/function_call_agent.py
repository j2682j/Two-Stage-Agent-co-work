"""?????????"""

from __future__ import annotations

import json
from typing import Iterator, Optional, Union, TYPE_CHECKING, Any, Dict

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


def _map_parameter_type(param_type: str) -> str:
    """將工具參數類型映射為JSON Schema允許的類型"""
    normalized = (param_type or "").lower()
    if normalized in {"string", "number", "integer", "boolean", "array", "object"}:
        return normalized
    return "string"


class FunctionCallAgent(Agent):
    """基於OpenAI原生函式呼叫機制的Agent"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        enable_tool_calling: bool = True,
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 3,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations

    def _get_system_prompt(self) -> str:
        """建構系統提示詞，注入工具描述"""
        base_prompt = self.system_prompt or "你是一個可靠的AI 助理，能夠在需要時呼叫工具完成任務。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暫無可用工具":
            return base_prompt

        prompt = base_prompt + "\n\n## 可用工具\n"
        prompt += "當你判斷需要外部資訊或執行動作時，可以直接通過函式呼叫使用以下工具：\n"
        prompt += tools_description + "\n"
        prompt += "\n請主動決定是否呼叫工具，合理利用多次呼叫來獲得完備答案。"
        return prompt

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        if not self.enable_tool_calling or not self.tool_registry:
            return []

        schemas: list[dict[str, Any]] = []

        # Tool對象
        for tool in self.tool_registry.get_all_tools():
            properties: Dict[str, Any] = {}
            required: list[str] = []

            try:
                parameters = tool.get_parameters()
            except Exception:
                parameters = []

            for param in parameters:
                properties[param.name] = {
                    "type": _map_parameter_type(param.type),
                    "description": param.description or ""
                }
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                if getattr(param, "required", True):
                    required.append(param.name)

            schema: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)

        # register_function 註冊的工具（直接訪問內部結構）
        function_map = getattr(self.tool_registry, "_functions", {})
        for name, info in function_map.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "輸入文字"
                                }
                            },
                            "required": ["input"]
                        }
                    }
                }
            )

        return schemas

    @staticmethod
    def _extract_message_content(raw_content: Any) -> str:
        """從OpenAI回應的message.content中安全提取文字"""
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(raw_content)

    @staticmethod
    def _parse_function_call_arguments(arguments: Optional[str]) -> dict[str, Any]:
        """解析模型回傳的JSON字串參數"""
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _convert_parameter_types(self, tool_name: str, param_dict: dict[str, Any]) -> dict[str, Any]:
        """根據工具定義盡可能轉換參數類型"""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            tool_params = tool.get_parameters()
        except Exception:
            return param_dict

        type_mapping = {param.name: param.type for param in tool_params}
        converted: dict[str, Any] = {}

        for key, value in param_dict.items():
            param_type = type_mapping.get(key)
            if not param_type:
                converted[key] = value
                continue

            try:
                normalized = param_type.lower()
                if normalized in {"number", "float"}:
                    converted[key] = float(value)
                elif normalized in {"integer", "int"}:
                    converted[key] = int(value)
                elif normalized in {"boolean", "bool"}:
                    if isinstance(value, bool):
                        converted[key] = value
                    elif isinstance(value, (int, float)):
                        converted[key] = bool(value)
                    elif isinstance(value, str):
                        converted[key] = value.lower() in {"true", "1", "yes"}
                    else:
                        converted[key] = bool(value)
                else:
                    converted[key] = value
            except (TypeError, ValueError):
                converted[key] = value

        return converted

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """執行工具呼叫並回傳字串結果"""
        if not self.tool_registry:
            return "❌ 錯誤：未設定工具註冊表"

        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                typed_arguments = self._convert_parameter_types(tool_name, arguments)
                return tool.run(typed_arguments)
            except Exception as exc:
                return f"❌ 工具呼叫失敗：{exc}"

        func = self.tool_registry.get_function(tool_name)
        if func:
            try:
                input_text = arguments.get("input", "")
                return func(input_text)
            except Exception as exc:
                return f"❌ 工具呼叫失敗：{exc}"

        return f"❌ 錯誤：找不到工具 '{tool_name}'"

    def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """呼叫底層OpenAI客戶端執行函式呼叫"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("HelloAgentsLLM 未正確初始化客戶端，無法執行函式呼叫。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> str:
        """
        執行函式呼叫范式的對話流程
        """
        messages: list[dict[str, Any]] = []
        system_prompt = self._get_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        tool_schemas = self._build_tool_schemas()
        if not tool_schemas:
            response_text = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response_text, "assistant"))
            return response_text

        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice

        current_iteration = 0
        final_response = ""

        while current_iteration < iterations_limit:
            response = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice=effective_tool_choice,
                **kwargs,
            )

            choice = response.choices[0]
            assistant_message = choice.message
            content = self._extract_message_content(assistant_message.content)
            tool_calls = list(assistant_message.tool_calls or [])

            if tool_calls:
                assistant_payload: dict[str, Any] = {"role": "assistant", "content": content}
                assistant_payload["tool_calls"] = []

                for tool_call in tool_calls:
                    assistant_payload["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )
                messages.append(assistant_payload)

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    arguments = self._parse_function_call_arguments(tool_call.function.arguments)
                    result = self._execute_tool_call(tool_name, arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                        }
                    )

                current_iteration += 1
                continue

            final_response = content
            messages.append({"role": "assistant", "content": final_response})
            break

        if current_iteration >= iterations_limit and not final_response:
            final_choice = self._invoke_with_tools(
                messages,
                tools=tool_schemas,
                tool_choice="none",
                **kwargs,
            )
            final_response = self._extract_message_content(final_choice.choices[0].message.content)
            messages.append({"role": "assistant", "content": final_response})

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        return final_response

    def add_tool(self, tool) -> None:
        """便捷方法：將工具註冊到目前Agent"""
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        if hasattr(tool, "auto_expand") and getattr(tool, "auto_expand"):
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for expanded_tool in expanded_tools:
                    self.tool_registry.register_tool(expanded_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展開為 {len(expanded_tools)} 個獨立工具")
                return

        self.tool_registry.register_tool(tool)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            before = set(self.tool_registry.list_tools())
            self.tool_registry.unregister(tool_name)
            after = set(self.tool_registry.list_tools())
            return tool_name in before and tool_name not in after
        return False

    def list_tools(self) -> list[str]:
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []

    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式呼叫暫未實現，直接回退到一次性呼叫"""
        result = self.run(input_text, **kwargs)
        yield result
