from typing import Any, Dict, List, Optional

from tools.builtin.calculator import CalculatorTool
from tools.builtin.memory_tool import MemoryTool
from tools.builtin.rag_tool import RAGTool
from tools.builtin.search_tool import SearchTool
from tools.registry import ToolRegistry
from memory import MemoryConfig


class ToolManager:
    """
    管理工具註冊、啟用狀態與執行紀錄。
    """

    def __init__(
        self,
        *,
        memory_config: Optional[MemoryConfig] = None,
        shared_memory_user_id: str = "network_shared",
        enable_shared_memory: bool = True,
        memory_types: Optional[List[str]] = None,
    ):
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
        self.tools[tool.name] = tool
        self.registry.register_tool(tool, auto_expand=auto_expand)

    def register_default_tools(self):
        calculator = CalculatorTool()
        self.register_tool(calculator)
        self.enabled_tools.add(calculator.name)

        search_tool = SearchTool()
        self.register_tool(search_tool)
        self.enabled_tools.add(search_tool.name)

        configured_memory_types = self.memory_types or (
            ["working", "episodic", "semantic"] if self.enable_shared_memory else ["working"]
        )
        self.memory_tool = MemoryTool(
            user_id=self.shared_memory_user_id,
            memory_config=self.memory_config,
            memory_types=configured_memory_types,
        )
        self.register_tool(self.memory_tool)
        self.enabled_tools.add(self.memory_tool.name)

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
        self.enabled_tools = set(tool_names)

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        ...

    def _record_trace(self, tool_name, parameters, result, agent_id=None, stage=None):
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
