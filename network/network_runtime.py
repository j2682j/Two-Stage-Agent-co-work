from __future__ import annotations

import os
from typing import Any

from memory.Graph import InsightGraph, QueryTaskGraph


class NetworkRuntime:
    """Shared runtime state for tools, attachments, and graph memory."""

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
        self.current_context: dict[str, Any] = {}
        self.current_attachment: dict[str, Any] | None = None
        self.shared_attachment_bundle: dict[str, Any] | None = None
        self.enable_stage1_attachment_after_first_round = False

        self.shared_tool_traces: list[dict[str, Any]] = []
        self.shared_memory_reads: list[dict[str, Any]] = []
        self.shared_memory_writes: list[dict[str, Any]] = []
        self.shared_stage2_search_bundle: dict[str, Any] | None = None
        self.current_stage2_stage1_result: str | None = None
        self.current_stage2_top_k_answers: list[str] = []
        self.current_stage2_judge_scores: list[float] = []
        self.query_task_graph: QueryTaskGraph | None = None
        self.insight_graph: InsightGraph | None = None
        self._init_gaia_graph_memory()

    def _init_gaia_graph_memory(self) -> None:
        auto_connect = os.getenv("GAIA_GRAPH_MEMORY_NEO4J", "0") == "1"
        try:
            self.query_task_graph = QueryTaskGraph(auto_connect=auto_connect)
            self.insight_graph = InsightGraph(auto_connect=auto_connect)
        except Exception as exc:
            print(f"[WARN] GAIA graph memory init failed; graph guidance disabled: {exc}")
            self.query_task_graph = None
            self.insight_graph = None

    def record_tool_trace(self, trace: dict[str, Any]) -> None:
        self.shared_tool_traces.append(trace)

    def record_memory_read(self, trace: dict[str, Any]) -> None:
        self.shared_memory_reads.append(trace)

    def record_memory_write(self, trace: dict[str, Any]) -> None:
        self.shared_memory_writes.append(trace)

    def clear_stage2_shared_state(self) -> None:
        self.shared_stage2_search_bundle = None
        self.current_stage2_stage1_result = None
        self.current_stage2_top_k_answers = []
        self.current_stage2_judge_scores = []

    def clear_current_context(self) -> None:
        self.current_context = {}
        self.current_attachment = None
        self.shared_attachment_bundle = None

    def should_include_stage1_attachment(self, is_first_round: bool) -> bool:
        if self.current_attachment is None:
            return False
        if is_first_round:
            return True
        return bool(self.enable_stage1_attachment_after_first_round)

    def prepare_shared_attachment_evidence(
        self,
        question: str,
        *,
        agent_id: str = "shared_attachment_reader",
        stage: str = "attachment_shared",
    ) -> dict[str, Any] | None:
        builder = getattr(self, "evidence_builder", None)
        normalized_question = str(question or "").strip()
        if builder is None or not normalized_question or self.current_attachment is None:
            self.shared_attachment_bundle = None
            return None

        try:
            bundle = builder.build_shared_attachment_bundle(
                question=normalized_question,
                agent_id=agent_id,
                stage=stage,
            )
        except Exception as exc:
            print(f"[WARN] shared attachment evidence failed: {exc}")
            self.shared_attachment_bundle = None
            return None

        self.shared_attachment_bundle = bundle
        if bundle.get("tool_usage"):
            self.record_tool_trace(
                {
                    "agent_id": agent_id,
                    "stage": stage,
                    "question": normalized_question,
                    "tool_usage": bundle.get("tool_usage", []),
                    "metadata": bundle.get("metadata", {}),
                }
            )

        return bundle

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
            print(f"[WARN] shared stage2 search failed: {exc}")
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
