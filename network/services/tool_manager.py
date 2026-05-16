from typing import Any, Dict, List, Optional

from tools.builtin.calculator import CalculatorTool
from tools.builtin.rag_tool import RAGTool
from tools.builtin.search_tool import SearchTool
from tools.registry import ToolRegistry
from memory.base import MemoryConfig


class ToolManager:
    """
    負責在 network.tool_manager 中封裝 ToolManager，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        memory_config: 記憶系統提供的檢索結果、寫入資料或操作介面。
        shared_memory_user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
        enable_shared_memory: 控制是否啟用此項資料、功能或處理分支的布林開關。
        memory_types: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        *,
        memory_config: Optional[MemoryConfig] = None,
        shared_memory_user_id: str = "network_shared",
        enable_shared_memory: bool = False,
        memory_types: Optional[List[str]] = None,
    ):
        """
        負責執行 ToolManager 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            memory_config: 記憶系統提供的檢索結果、寫入資料或操作介面。
            shared_memory_user_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            enable_shared_memory: 控制是否啟用此項資料、功能或處理分支的布林開關。
            memory_types: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.registry = ToolRegistry()
        self.tools: dict[str, Any] = {}
        self.enabled_tools: set[str] = set()
        self.tool_traces: List[Dict[str, Any]] = []
        self.memory_tool = None
        self.rag_tool = None
        self.memory_config = memory_config
        self.shared_memory_user_id = shared_memory_user_id
        self.enable_shared_memory = enable_shared_memory
        self.memory_types = memory_types

        self.register_default_tools()

    def register_tool(self, tool, auto_expand=True):
        """
        負責執行 ToolManager 中的 register_tool 流程，依照 ToolManager 的流程需求處理 register_tool 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool: 可呼叫的工具、工具名稱或工具註冊表。
            auto_expand: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.tools[tool.name] = tool
        self.registry.register_tool(tool, auto_expand=auto_expand)

    def register_default_tools(self):
        """
        負責執行 ToolManager 中的 register_default_tools 流程，依照 ToolManager 的流程需求處理 register_default_tools 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        calculator = CalculatorTool()
        self.register_tool(calculator)
        self.enabled_tools.add(calculator.name)

        search_tool = SearchTool()
        self.register_tool(search_tool)
        self.enabled_tools.add(search_tool.name)

        try:
            rag_tool = RAGTool()
            if getattr(rag_tool, "initialized", False):
                self.rag_tool = rag_tool
                self.register_tool(self.rag_tool)
                self.enabled_tools.add(self.rag_tool.name)
            else:
                self.rag_tool = None
        except Exception as e:
            self.rag_tool = None
            print(f"[WARN] RAGTool 初始化失敗: {e}")

    def set_enabled_tools(self, tool_names):
        """
        負責執行 ToolManager 中的 set_enabled_tools 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            tool_names: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.enabled_tools = set(tool_names)

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        負責執行 ToolManager 中的 execute_tool 流程，依照 ToolManager 的流程需求處理 execute_tool 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            parameters: 此流程需要使用的輸入資料。
            agent_id: 目前執行或需要記錄的代理節點識別資訊。
            stage: 目前執行的階段、輪次或流程位置。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if tool_name not in self.enabled_tools:
            result = {
                "ok": False,
                "tool_name": tool_name,
                "output_text": "",
                "error": f"tool '{tool_name}' is not enabled",
            }
            self._record_trace(tool_name, parameters, result, agent_id, stage)
            return result

        tool = self.tools.get(tool_name) or self.registry.get_tool(tool_name)
        if tool is None:
            result = {
                "ok": False,
                "tool_name": tool_name,
                "output_text": "",
                "error": f"tool '{tool_name}' not found",
            }
            self._record_trace(tool_name, parameters, result, agent_id, stage)
            return result

        try:
            raw = tool.run(parameters)
            result = {
                "ok": True,
                "tool_name": tool_name,
                "output_text": str(raw),
                "raw_result": raw,
                "error": None,
            }
        except Exception as e:
            result = {
                "ok": False,
                "tool_name": tool_name,
                "output_text": "",
                "raw_result": None,
                "error": str(e),
            }

        self._record_trace(tool_name, parameters, result, agent_id, stage)
        return result

    def normalize_result(self, tool_name, raw_result):
        """
        負責執行 ToolManager 中的 normalize_result 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            raw_result: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        ...

    def _record_trace(self, tool_name, parameters, result, agent_id=None, stage=None):
        """
        負責執行 ToolManager 中的 _record_trace 流程，依照 ToolManager 的流程需求處理 _record_trace 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            parameters: 此流程需要使用的輸入資料。
            result: 評估、推理或工具執行後產生的結果與分數資料。
            agent_id: 目前執行或需要記錄的代理節點識別資訊。
            stage: 目前執行的階段、輪次或流程位置。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.tool_traces.append(
            {
                "tool_name": tool_name,
                "parameters": parameters,
                "agent_id": agent_id,
                "stage": stage,
                "ok": result.get("ok", False),
                "output_text": result.get("output_text", ""),
                "error": result.get("error"),
            }
        )
