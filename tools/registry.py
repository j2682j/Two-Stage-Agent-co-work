"""???????"""


from typing import Any, Callable, Optional

from .base import Tool


class ToolRegistry:
    """
    負責在 tools.registry 中封裝 ToolRegistry，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self) -> None:
        """
        負責執行 ToolRegistry 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True) -> None:
        """
        負責執行 ToolRegistry 中的 register_tool 流程，依照 ToolRegistry 的流程需求處理 register_tool 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool: 可呼叫的工具、工具名稱或工具註冊表。
            auto_expand: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 ToolRegistry 中的 register_function 流程，依照 ToolRegistry 的流程需求處理 register_function 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
            func: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if name in self._functions:
            print(f"[WARN] 工具 '{name}' 已存在，將被覆蓋。")

        self._functions[name] = {
            "description": description,
            "func": func,
        }
        print(f"[OK] 工具 '{name}' 已註冊。")

    def unregister(self, name: str) -> None:
        """
        負責執行 ToolRegistry 中的 unregister 流程，依照 ToolRegistry 的流程需求處理 unregister 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if name in self._tools:
            del self._tools[name]
            print(f"[INFO] 工具 '{name}' 已註銷。")
        elif name in self._functions:
            del self._functions[name]
            print(f"[INFO] 工具 '{name}' 已註銷。")
        else:
            print(f"[WARN] 工具 '{name}' 不存在。")

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        負責執行 ToolRegistry 中的 get_tool 流程，依照 ToolRegistry 的流程需求處理 get_tool 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Tool]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable[[str], str]]:
        """
        負責執行 ToolRegistry 中的 get_function 流程，依照 ToolRegistry 的流程需求處理 get_function 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Callable[[str], str]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> str:
        """
        負責執行 ToolRegistry 中的 execute_tool 流程，依照 ToolRegistry 的流程需求處理 execute_tool 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            name: 此流程需要使用的輸入資料。
            input_text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 ToolRegistry 中的 get_tools_description 流程，依照 ToolRegistry 的流程需求處理 get_tools_description 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "目前沒有已註冊工具"

    def list_tools(self) -> list[str]:
        """
        負責執行 ToolRegistry 中的 list_tools 流程，依照 ToolRegistry 的流程需求處理 list_tools 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """
        負責執行 ToolRegistry 中的 get_all_tools 流程，依照 ToolRegistry 的流程需求處理 get_all_tools 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[Tool]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return list(self._tools.values())

    def clear(self) -> None:
        """
        負責執行 ToolRegistry 中的 clear 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._tools.clear()
        self._functions.clear()
        print("[OK] 所有工具已清空。")


global_registry = ToolRegistry()
