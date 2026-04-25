"""Built-in memory tool.

This tool is the user-facing entrypoint for the memory system.
It handles action routing and parameter parsing, while delegating
the core memory logic to ``MemoryManager``.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from memory import MemoryConfig, MemoryManager

from ..base import Tool, ToolParameter, tool_action


class MemoryTool(Tool):
    """Memory tool facade for agents and the tool registry."""

    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: Optional[MemoryConfig] = None,
        memory_types: Optional[List[str]] = None,
        expandable: bool = False,
        memory_manager: Optional[MemoryManager] = None,
    ):
        super().__init__(
            name="memory",
            description=(
                "Memory tool for adding, searching, summarizing, updating, "
                "forgetting, and consolidating memories."
            ),
            expandable=expandable,
        )

        self.user_id = user_id
        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memory_types or ["working", "episodic", "semantic"]

        self.memory_manager = memory_manager or MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types,
        )
        self.memory_config = self.memory_manager.config

        self.current_session_id = None
        self.conversation_count = 0

    def run(self, parameters: Dict[str, Any]) -> str:
        """Execute a memory action."""
        if not self.validate_parameters(parameters):
            return "[ERROR] 參數驗證失敗，請檢查必填欄位與型別。"

        action = parameters.get("action")

        if action == "add":
            return self._add_memory(
                content=parameters.get("content", ""),
                memory_type=parameters.get("memory_type", "working"),
                importance=parameters.get("importance", 0.5),
                file_path=parameters.get("file_path"),
                modality=parameters.get("modality"),
                metadata=parameters.get("metadata"),
            )
        if action == "search":
            return self._search_memory(
                query=parameters.get("query"),
                limit=parameters.get("limit", 5),
                memory_type=parameters.get("memory_type"),
                min_importance=parameters.get("min_importance", 0.1),
            )
        if action == "summary":
            return self._get_summary(limit=parameters.get("limit", 10))
        if action == "stats":
            return self._get_stats()
        if action == "update":
            return self._update_memory(
                memory_id=parameters.get("memory_id"),
                content=parameters.get("content"),
                importance=parameters.get("importance"),
            )
        if action == "remove":
            return self._remove_memory(memory_id=parameters.get("memory_id"))
        if action == "forget":
            return self._forget(
                strategy=parameters.get("strategy", "importance_based"),
                threshold=parameters.get("threshold", 0.1),
                max_age_days=parameters.get("max_age_days", 30),
            )
        if action == "consolidate":
            return self._consolidate(
                from_type=parameters.get("from_type", "working"),
                to_type=parameters.get("to_type", "episodic"),
                importance_threshold=parameters.get("importance_threshold", 0.7),
            )
        if action == "clear_all":
            return self._clear_all()

        return f"[ERROR] 不支援的 action: {action}"

    def get_parameters(self) -> List[ToolParameter]:
        """Return tool parameter definitions."""
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "可用操作: "
                    "add, search, summary, stats, update, remove, "
                    "forget, consolidate, clear_all"
                ),
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="記憶內容，供 add 或 update 使用。",
                required=False,
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜尋查詢字串，供 search 使用。",
                required=False,
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="記憶類型: working, episodic, semantic, perceptual。",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="importance",
                type="number",
                description="重要度，範圍 0.0 到 1.0，供 add 或 update 使用。",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="結果數量上限。",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="memory_id",
                type="string",
                description="記憶 ID，供 update 或 remove 使用。",
                required=False,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="多模態記憶的檔案路徑。",
                required=False,
            ),
            ToolParameter(
                name="modality",
                type="string",
                description="多模態類型，例如 text、image、audio。",
                required=False,
            ),
            ToolParameter(
                name="strategy",
                type="string",
                description="遺忘策略: importance_based, time_based, capacity_based。",
                required=False,
                default="importance_based",
            ),
            ToolParameter(
                name="threshold",
                type="number",
                description="遺忘門檻。",
                required=False,
                default=0.1,
            ),
            ToolParameter(
                name="max_age_days",
                type="integer",
                description="time_based 遺忘策略的最長保留天數。",
                required=False,
                default=30,
            ),
            ToolParameter(
                name="from_type",
                type="string",
                description="記憶整理的來源類型。",
                required=False,
                default="working",
            ),
            ToolParameter(
                name="to_type",
                type="string",
                description="記憶整理的目標類型。",
                required=False,
                default="episodic",
            ),
            ToolParameter(
                name="importance_threshold",
                type="number",
                description="記憶整理的重要度門檻。",
                required=False,
                default=0.7,
            ),
            ToolParameter(
                name="metadata",
                type="object",
                description="Optional structured metadata stored with the memory record.",
                required=False,
            ),
        ]

    @tool_action("memory_add", "新增一筆記憶")
    def _add_memory(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        file_path: str = None,
        modality: str = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Add a memory item."""
        record_metadata: Dict[str, Any] = dict(metadata or {})
        try:
            if self.current_session_id is None:
                self.current_session_id = (
                    f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

            if memory_type == "perceptual" and file_path:
                inferred = modality or self._infer_modality(file_path)
                record_metadata.setdefault("modality", inferred)
                record_metadata.setdefault("raw_data", file_path)

            record_metadata.update(
                {
                    "session_id": self.current_session_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=record_metadata,
                auto_classify=True,
            )

            return f"[OK] 記憶已添加 (ID: {memory_id[:8]}...)"
        except Exception as e:
            return f"[ERROR] 新增記憶失敗: {str(e)}"

    def _infer_modality(self, path: str) -> str:
        """Infer modality from file extension."""
        try:
            ext = (path.rsplit(".", 1)[-1] or "").lower()
            if ext in {"png", "jpg", "jpeg", "bmp", "gif", "webp"}:
                return "image"
            if ext in {"mp3", "wav", "flac", "m4a", "ogg"}:
                return "audio"
            return "text"
        except Exception:
            return "text"

    @tool_action("memory_search", "搜尋記憶")
    def _search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_type: str = None,
        min_importance: float = 0.1,
    ) -> str:
        """Search memories."""
        try:
            memory_types = [memory_type] if memory_type else None
            results = self.memory_manager.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=memory_types,
                min_importance=min_importance,
            )

            if not results:
                return f"[INFO] 找不到與 '{query}' 相關的記憶。"

            formatted_results = [f"[INFO] 找到 {len(results)} 條相關記憶"]
            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作記憶",
                    "episodic": "情節記憶",
                    "semantic": "語意記憶",
                    "perceptual": "感知記憶",
                }.get(memory.memory_type, memory.memory_type)
                content_preview = (
                    memory.content[:80] + "..."
                    if len(memory.content) > 80
                    else memory.content
                )
                formatted_results.append(
                    f"{i}. [{memory_type_label}] {content_preview} "
                    f"(重要度 {memory.importance:.2f})"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"[ERROR] 搜尋記憶失敗: {str(e)}"

    @tool_action("memory_summary", "取得記憶摘要")
    def _get_summary(self, limit: int = 10) -> str:
        """Return a memory summary."""
        try:
            stats = self.memory_manager.get_memory_stats()
            summary_parts = [
                "[INFO] 記憶系統摘要",
                f"總記憶數: {stats['total_memories']}",
                f"目前 session: {self.current_session_id or '尚未建立'}",
                f"對話次數: {self.conversation_count}",
            ]

            if stats["memories_by_type"]:
                summary_parts.append("\n各類型記憶:")
                for memory_type, type_stats in stats["memories_by_type"].items():
                    count = type_stats.get("count", 0)
                    avg_importance = type_stats.get("avg_importance", 0)
                    type_label = {
                        "working": "工作記憶",
                        "episodic": "情節記憶",
                        "semantic": "語意記憶",
                        "perceptual": "感知記憶",
                    }.get(memory_type, memory_type)
                    summary_parts.append(
                        f"  - {type_label}: {count} 條 (平均重要度 {avg_importance:.2f})"
                    )

            important_memories = self.memory_manager.retrieve_memories(
                query="",
                memory_types=None,
                limit=limit * 3,
                min_importance=0.5,
            )

            if important_memories:
                seen_ids = set()
                seen_contents = set()
                unique_memories = []
                for memory in important_memories:
                    if memory.id in seen_ids:
                        continue
                    content_key = memory.content.strip().lower()
                    if content_key in seen_contents:
                        continue
                    seen_ids.add(memory.id)
                    seen_contents.add(content_key)
                    unique_memories.append(memory)

                unique_memories.sort(key=lambda x: x.importance, reverse=True)
                summary_parts.append(
                    f"\n重要記憶 (前 {min(limit, len(unique_memories))} 條):"
                )
                for i, memory in enumerate(unique_memories[:limit], 1):
                    content_preview = (
                        memory.content[:60] + "..."
                        if len(memory.content) > 60
                        else memory.content
                    )
                    summary_parts.append(
                        f"  {i}. {content_preview} (重要度 {memory.importance:.2f})"
                    )

            return "\n".join(summary_parts)
        except Exception as e:
            return f"[ERROR] 取得摘要失敗: {str(e)}"

    @tool_action("memory_stats", "取得記憶統計")
    def _get_stats(self) -> str:
        """Return memory stats."""
        try:
            stats = self.memory_manager.get_memory_stats()
            stats_info = [
                "[INFO] 記憶系統統計",
                f"總記憶數: {stats['total_memories']}",
                f"啟用類型: {', '.join(stats['enabled_types'])}",
                f"目前 session: {self.current_session_id or '尚未建立'}",
                f"對話次數: {self.conversation_count}",
            ]
            return "\n".join(stats_info)
        except Exception as e:
            return f"[ERROR] 取得統計失敗: {str(e)}"

    def auto_record_conversation(self, user_input: str, agent_response: str):
        """Automatically record a conversation turn."""
        self.conversation_count += 1

        self._add_memory(
            content=f"使用者輸入: {user_input}",
            memory_type="working",
            importance=0.6,
        )

        self._add_memory(
            content=f"Agent 回覆: {agent_response}",
            memory_type="working",
            importance=0.7,
        )

        if len(agent_response) > 100 or "重要" in user_input or "記住" in user_input:
            interaction_content = (
                f"對話紀錄 - 使用者輸入: {user_input}\nAgent 回覆: {agent_response}"
            )
            self._add_memory(
                content=interaction_content,
                memory_type="episodic",
                importance=0.8,
            )

    @tool_action("memory_update", "更新既有記憶")
    def _update_memory(
        self, memory_id: str, content: str = None, importance: float = None
    ) -> str:
        """Update a memory item."""
        try:
            success = self.memory_manager.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                metadata=None,
            )
            if success:
                return "[OK] 記憶已更新"
            return "[WARN] 找不到要更新的記憶"
        except Exception as e:
            return f"[ERROR] 更新記憶失敗: {str(e)}"

    @tool_action("memory_remove", "刪除指定記憶")
    def _remove_memory(self, memory_id: str) -> str:
        """Remove a memory item."""
        try:
            success = self.memory_manager.remove_memory(memory_id)
            if success:
                return "[OK] 記憶已刪除"
            return "[WARN] 找不到要刪除的記憶"
        except Exception as e:
            return f"[ERROR] 刪除記憶失敗: {str(e)}"

    @tool_action("memory_forget", "依策略遺忘記憶")
    def _forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> str:
        """Forget memories by strategy."""
        try:
            count = self.memory_manager.forget_memories(
                strategy=strategy,
                threshold=threshold,
                max_age_days=max_age_days,
            )
            return f"[OK] 已遺忘 {count} 條記憶 (策略: {strategy})"
        except Exception as e:
            return f"[ERROR] 遺忘記憶失敗: {str(e)}"

    @tool_action("memory_consolidate", "整理與合併記憶")
    def _consolidate(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ) -> str:
        """Consolidate memories across types."""
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold,
            )
            return (
                f"[OK] 已整理 {count} 條記憶 "
                f"({from_type} -> {to_type}, 門檻 {importance_threshold})"
            )
        except Exception as e:
            return f"[ERROR] 整理記憶失敗: {str(e)}"

    @tool_action("memory_clear", "清除全部記憶")
    def _clear_all(self) -> str:
        """Clear all memories."""
        try:
            self.memory_manager.clear_all_memories()
            return "[OK] 已清除全部記憶"
        except Exception as e:
            return f"[ERROR] 清除記憶失敗: {str(e)}"

    def add_knowledge(self, content: str, importance: float = 0.9):
        """Add a knowledge-like memory."""
        return self._add_memory(
            content=content,
            memory_type="working",
            importance=importance,
        )

    def get_context_for_query(self, query: str, limit: int = 3) -> str:
        """Build a compact memory context string for a query."""
        results = self.memory_manager.retrieve_memories(
            query=query,
            limit=limit,
            min_importance=0.3,
        )

        if not results:
            return ""

        context_parts = ["相關記憶:"]
        for memory in results:
            context_parts.append(f"- {memory.content}")
        return "\n".join(context_parts)

    def clear_session(self):
        """Reset current session metadata and clear working memory."""
        self.current_session_id = None
        self.conversation_count = 0

        wm = (
            self.memory_manager.memory_types.get("working")
            if hasattr(self.memory_manager, "memory_types")
            else None
        )
        if wm:
            wm.clear()

    def consolidate_memories(self):
        """Consolidate memories with manager defaults."""
        return self.memory_manager.consolidate_memories()

    def forget_old_memories(self, max_age_days: int = 30):
        """Forget old memories using time-based strategy."""
        return self.memory_manager.forget_memories(
            strategy="time_based",
            max_age_days=max_age_days,
        )
