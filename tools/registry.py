"""???????"""


from typing import Any, Callable, Optional

from .base import Tool


class ToolRegistry:
    """管理工具與函式型工具的註冊、查詢與執行。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True) -> None:
        """註冊工具。

        若工具支援展開，會改為註冊其展開後的子工具。
        """
        if auto_expand and hasattr(tool, "expandable") and tool.expandable:
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                for sub_tool in expanded_tools:
                    if sub_tool.name in self._tools:
                        print(f"[WARN] 工具 '{sub_tool.name}' 已存在，將被覆蓋。")
                    self._tools[sub_tool.name] = sub_tool
                print(f"[OK] 工具 '{tool.name}' 已展開為 {len(expanded_tools)} 個獨立工具。")
                return

        if tool.name in self._tools:
            print(f"[WARN] 工具 '{tool.name}' 已存在，將被覆蓋。")

        self._tools[tool.name] = tool
        print(f"[OK] 工具 '{tool.name}' 已註冊。")

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[[str], str],
    ) -> None:
        """註冊函式型工具。"""
        if name in self._functions:
            print(f"[WARN] 工具 '{name}' 已存在，將被覆蓋。")

        self._functions[name] = {
            "description": description,
            "func": func,
        }
        print(f"[OK] 工具 '{name}' 已註冊。")

    def unregister(self, name: str) -> None:
        """移除工具或函式型工具。"""
        if name in self._tools:
            del self._tools[name]
            print(f"[INFO] 工具 '{name}' 已註銷。")
        elif name in self._functions:
            del self._functions[name]
            print(f"[INFO] 工具 '{name}' 已註銷。")
        else:
            print(f"[WARN] 工具 '{name}' 不存在。")

    def get_tool(self, name: str) -> Optional[Tool]:
        """取得工具實例。"""
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable[[str], str]]:
        """取得函式型工具。"""
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> str:
        """執行指定工具。"""
        if name in self._tools:
            tool = self._tools[name]
            try:
                return tool.run({"input": input_text})
            except Exception as e:
                return f"[ERROR] 工具 '{name}' 執行失敗: {str(e)}"

        if name in self._functions:
            func = self._functions[name]["func"]
            try:
                return func(input_text)
            except Exception as e:
                return f"[ERROR] 工具 '{name}' 執行失敗: {str(e)}"

        return f"[ERROR] 找不到名稱為 '{name}' 的工具。"

    def get_tools_description(self) -> str:
        """回傳所有已註冊工具的描述。"""
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "目前沒有已註冊工具"

    def list_tools(self) -> list[str]:
        """列出所有工具名稱。"""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """取得所有工具實例。"""
        return list(self._tools.values())

    def clear(self) -> None:
        """清空所有註冊內容。"""
        self._tools.clear()
        self._functions.clear()
        print("[OK] 所有工具已清空。")


global_registry = ToolRegistry()
