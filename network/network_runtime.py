from __future__ import annotations

import re
from typing import Any

from memory.lesson_rule import (
    build_retrieval_profile,
    parse_semantic_lesson_memory,
    parse_semantic_lesson_text,
    select_relevant_semantic_lessons,
)


class NetworkRuntime:
    """
    NetworkRuntime
    1. 初始化:
    - tool_manager: 提供共同工具管理介面，包含記憶工具等
    - memory_tool: 提供記憶相關功能的工具，包含記憶檢索、寫入等
    - memory_config: 記憶工具的配置參數
    - shared_memory_user_id: 用於共享記憶的使用者ID，預設為 "network_shared"
    - enable_shared_memory: 是否啟用共享記憶功能，預設為 True
    - memory_mode: 記憶模式設定，預設為 "disabled"
    - evidence_builder: 提供證據構建功能的工具
    - debug_print_stage1_first_round_prompt: 是否在第一階段的第一輪提示中顯示信息，預設為 False
    """

    def __init__(
        self,
        tool_manager: Any,
        memory_tool: Any = None,
        memory_config: Any = None,
        shared_memory_user_id: str = "network_shared",
        enable_shared_memory: bool = True,
        memory_mode: str = "disabled",
        evidence_builder: Any = None,
        debug_print_stage1_first_round_prompt: bool = False,
    ):
        self.tool_manager = tool_manager
        self.memory_tool = memory_tool
        self.memory_config = memory_config
        self.shared_memory_user_id = shared_memory_user_id
        self.enable_shared_memory = enable_shared_memory
        self.memory_mode = memory_mode
        self.evidence_builder = evidence_builder
        self.enable_stage1_tools = False
        self.debug_print_stage1_first_round_prompt = debug_print_stage1_first_round_prompt
        self.last_stage1_first_round_prompt: str | None = None

        self.shared_tool_traces: list[dict[str, Any]] = []
        self.shared_memory_reads: list[dict[str, Any]] = []
        self.shared_memory_writes: list[dict[str, Any]] = []
        self.shared_stage2_search_bundle: dict[str, Any] | None = None

    def record_tool_trace(self, trace: dict[str, Any]) -> None:
        self.shared_tool_traces.append(trace)

    def record_memory_read(self, trace: dict[str, Any]) -> None:
        self.shared_memory_reads.append(trace)

    def record_memory_write(self, trace: dict[str, Any]) -> None:
        self.shared_memory_writes.append(trace)

    def clear_stage2_shared_state(self) -> None:
        self.shared_stage2_search_bundle = None

    def prepare_shared_stage2_search(
        self,
        question: str,
        *,
        router_model_name: str | None = None,
    ) -> dict[str, Any] | None:
        builder = getattr(self, "evidence_builder", None)
        normalized_question = str(question or "").strip()
        if builder is None or not normalized_question:
            self.shared_stage2_search_bundle = None
            return None

        try:
            bundle = builder.build_shared_stage2_search_bundle(
                question=normalized_question,
                agent_id="shared_stage2_search",
                stage="stage2_shared_search",
                router_model_name=router_model_name,
            )
        except Exception as exc:
            print(f"[WARN] shared stage2 search 準備失敗: {exc}")
            self.shared_stage2_search_bundle = None
            return None

        self.shared_stage2_search_bundle = bundle
        if bundle.get("tool_usage"):
            self.record_tool_trace(
                {
                    "agent_id": "shared_stage2_search",
                    "stage": "stage2_shared_search",
                    "question": normalized_question,
                    "tool_usage": bundle.get("tool_usage", []),
                    "routing": bundle.get("routing", {}),
                    "shared_search_id": bundle.get("shared_search_id"),
                    "queries": bundle.get("queries", []),
                }
            )

        return bundle

    def normalize_memory_text(self, text: Any) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    def build_memory_profile(self, question: str) -> dict[str, Any]:
        return build_retrieval_profile(question)

    def memory_matches_error_type(self, content: str, target_error_type: str) -> bool:
        lesson = parse_semantic_lesson_text(content)
        if lesson is not None:
            return lesson.error_type == target_error_type if target_error_type else True
        if not target_error_type:
            return True
        normalized = self.normalize_memory_text(content).lower()
        return (
            f"error type: {target_error_type.lower()}" in normalized
            or f"error_type={target_error_type.lower()}" in normalized
        )

    def summarize_memory_content(self, content: str) -> str:
        lesson = parse_semantic_lesson_text(content)
        if lesson is not None:
            return lesson.to_summary()

        normalized = self.normalize_memory_text(content)
        if not normalized:
            return ""

        lesson_match = re.search(
            r"Lesson:\s*(.+?)(?:\s+Tags:|\s+Applicability:|$)",
            normalized,
            flags=re.IGNORECASE,
        )
        error_type_match = re.search(
            r"Error type:\s*([A-Za-z0-9_\-]+)",
            normalized,
            flags=re.IGNORECASE,
        )
        applicability_match = re.search(
            r"Applicability:\s*(.+?)(?:\s+Observed mismatch:|$)",
            normalized,
            flags=re.IGNORECASE,
        )

        if lesson_match:
            parts = []
            if error_type_match:
                parts.append(f"error_type={error_type_match.group(1).strip()}")
            parts.append(f"lesson={lesson_match.group(1).strip()}")
            if applicability_match:
                parts.append(f"applicability={applicability_match.group(1).strip()}")
            return " | ".join(parts)

        if len(normalized) > 220:
            return normalized[:217] + "..."
        return normalized

    def build_memory_context_for_final_decision(self, question: str, limit: int = 5) -> str:
        memory_manager = getattr(getattr(self, "memory_tool", None), "memory_manager", None)
        normalized_question = str(question or "").strip()
        if memory_manager is None or not normalized_question:
            return ""

        profile = self.build_memory_profile(normalized_question)
        lesson_queries = profile.lesson_queries
        case_queries = profile.case_queries
        lesson_memories = []
        lesson_candidates = []
        other_memories = []
        seen_ids: set[str] = set()

        query_plan = (
            [(query, ["semantic"], max(limit * 2, 4)) for query in lesson_queries]
            + [(query, ["episodic", "working"], max(2, limit)) for query in case_queries]
        )
        for query_text, memory_types, query_limit in query_plan:
            try:
                retrieved = memory_manager.retrieve_memories(
                    query=query_text,
                    memory_types=memory_types,
                    limit=query_limit,
                    min_importance=0.0,
                )
            except Exception as exc:
                print(f"[WARN] final decision memory 檢索失敗: {exc}")
                continue

            for memory in retrieved:
                memory_id = str(getattr(memory, "id", "") or "")
                if memory_id and memory_id in seen_ids:
                    continue
                if memory_id:
                    seen_ids.add(memory_id)
                if (
                    str(getattr(memory, "memory_type", "") or "").strip() == "semantic"
                ):
                    lesson = parse_semantic_lesson_memory(memory)
                    if lesson is None:
                        continue
                    lesson_memories.append(memory)
                    lesson_candidates.append(lesson)
                    continue
                other_memories.append(memory)

        lesson_limit = max(1, limit - 2) if other_memories else max(1, limit)
        selected_lessons = select_relevant_semantic_lessons(
            lessons=lesson_candidates,
            profile=profile,
            min_score=1.5,
            limit=lesson_limit,
        )
        selected_lesson_keys = {
            lesson.selection_key()
            for lesson, _ in selected_lessons
        }
        selected_lesson_memories = []
        for memory in lesson_memories:
            lesson = parse_semantic_lesson_memory(memory)
            if lesson is None:
                continue
            key = lesson.selection_key()
            if key in selected_lesson_keys:
                selected_lesson_memories.append(memory)

        other_memories.sort(
            key=lambda item: (
                1 if str(getattr(item, "memory_type", "") or "").strip() == "episodic" else 0,
                float(getattr(item, "importance", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected_lesson_memories = selected_lesson_memories[:lesson_limit]
        memories = selected_lesson_memories + other_memories[: max(0, limit - len(selected_lesson_memories))]

        lesson_lines: list[str] = []
        episode_lines: list[str] = []
        for idx, memory in enumerate(memories, start=1):
            content = str(getattr(memory, "content", "") or "").strip()
            if not content:
                continue
            memory_type = str(getattr(memory, "memory_type", "memory") or "memory").strip()
            importance = getattr(memory, "importance", None)
            prefix = f"[{idx}] ({memory_type})"
            if importance is not None:
                try:
                    prefix += f" importance={float(importance):.2f}"
                except Exception:
                    pass
            if memory_type == "semantic":
                lesson = parse_semantic_lesson_memory(memory)
                compact = lesson.to_summary() if lesson is not None else self.summarize_memory_content(content)
                lesson_lines.append(f"{prefix} {compact}")
            else:
                compact = self.summarize_memory_content(content)
                episode_lines.append(f"{prefix} {compact}")

        sections: list[str] = []
        if lesson_lines:
            sections.append("Relevant memory lessons:\n" + "\n".join(lesson_lines))
        if episode_lines:
            sections.append("Relevant memory cases:\n" + "\n".join(episode_lines))

        if not sections:
            return ""
        return "\n\n".join(sections)

    def dedupe_memory_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for record in records:
            memory_type = str(record.get("memory_type", "") or "").strip()
            content = str(record.get("content", "") or "").strip()
            if not memory_type or not content:
                continue

            key = (memory_type, content)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)

        return deduped
