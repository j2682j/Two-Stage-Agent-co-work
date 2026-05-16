from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from memory.graph import NetworkMemory
from ..services.memory_service import MemoryService
from .task_context import TaskContext
from .trace_record import LatencyRecord, TraceRecord, now_ms


class NetworkRuntime:
    """NetworkRuntime 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    def __init__(
        self,
        tool_manager: Any,
        memory_tool: Any = None,
        memory_config: Any = None,
        shared_memory_user_id: str = "network_shared",
        enable_shared_memory: bool = False,
        memory_mode: str = "disabled",
        evidence_builder: Any = None,
        debug_print_stage1_first_round_prompt: bool = False,
    ):
        """初始化 NetworkRuntime 實例。
        
        參數:
            tool_manager: 此流程需要使用的輸入資料。
            memory_tool: 此流程需要使用的輸入資料。
            memory_config: 此流程需要使用的輸入資料。
            shared_memory_user_id: 此流程需要使用的輸入資料。
            enable_shared_memory: 此流程需要使用的輸入資料。
            memory_mode: 此流程需要使用的輸入資料。
            evidence_builder: 此流程需要使用的輸入資料。
            debug_print_stage1_first_round_prompt: 此流程需要使用的輸入資料。
        """
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
        self.current_task_context: TaskContext = TaskContext()
        self.current_attachment: dict[str, Any] | None = None
        self.shared_attachment_bundle: dict[str, Any] | None = None
        self.enable_stage1_attachment_after_first_round = False

        self.shared_tool_traces: list[dict[str, Any]] = []
        self.shared_memory_reads: list[dict[str, Any]] = []
        self.shared_memory_writes: list[dict[str, Any]] = []
        self.shared_token_usage: list[dict[str, Any]] = []
        self.shared_latency_records: list[dict[str, Any]] = []
        self.shared_trace_records: list[dict[str, Any]] = []
        self.shared_stage2_search_bundle: dict[str, Any] | None = None
        self.current_stage2_stage1_result: str | None = None
        self.current_stage2_top_k_answers: list[str] = []
        self.current_stage2_judge_scores: list[float] = []
        self.graph_memory: NetworkMemory | None = None
        self.memory_service: MemoryService | None = None
        self._init_graph_memory()
        self.memory_service = MemoryService(self)

    def set_task_context(self, context: dict[str, Any] | TaskContext | None) -> TaskContext:
        task_context = TaskContext.from_dict(context)
        if not task_context.memory_namespace:
            data = task_context.to_dict()
            data["memory_namespace"] = str(self.shared_memory_user_id or "").strip()
            task_context = TaskContext.from_dict(data)
        self.current_task_context = task_context
        self.current_attachment = task_context.attachment or None
        return task_context

    def get_task_context(self) -> TaskContext:
        if isinstance(getattr(self, "current_task_context", None), TaskContext):
            return self.current_task_context
        task_context = TaskContext()
        self.current_task_context = task_context
        return task_context

    @property
    def current_context(self) -> dict[str, Any]:
        return self.get_task_context().to_dict()

    @current_context.setter
    def current_context(self, context: dict[str, Any] | TaskContext | None) -> None:
        task_context = TaskContext.from_dict(context)
        self.current_task_context = task_context
        self.current_attachment = task_context.attachment or None

    @property
    def query_task_graph(self):
        """處理 query_task_graph 流程並回傳結果。"""
        return self.graph_memory.query_task_graph if self.graph_memory is not None else None

    @property
    def insight_graph(self):
        """處理 insight_graph 流程並回傳結果。"""
        return self.graph_memory.insight_graph if self.graph_memory is not None else None

    def _init_graph_memory(self) -> None:
        """處理 init_graph_memory 流程並回傳結果。
        
        回傳:
            此函式的處理結果。
        """
        auto_connect = os.getenv("GAIA_GRAPH_MEMORY_NEO4J", "0") == "1"
        namespace = str(self.shared_memory_user_id or "system").strip() or "system"
        try:
            self.graph_memory = NetworkMemory(auto_connect=auto_connect, namespace=namespace)
        except Exception as exc:
            print(f"[WARN] graph memory init failed; graph guidance disabled: {exc}")
            self.graph_memory = None

    def record_tool_trace(self, trace: dict[str, Any]) -> None:
        """記錄 record_tool_trace 相關追蹤資料。
        
        參數:
            trace: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        self.shared_tool_traces.append(trace)

    def record_memory_read(self, trace: dict[str, Any]) -> None:
        """記錄 record_memory_read 相關追蹤資料。
        
        參數:
            trace: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        self.shared_memory_reads.append(trace)

    def record_memory_write(self, trace: dict[str, Any]) -> None:
        """寫入或儲存 record_memory_write 相關資料。
        
        參數:
            trace: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        self.shared_memory_writes.append(trace)

    def record_token_usage(self, trace: dict[str, Any]) -> None:
        """記錄 record_token_usage 相關追蹤資料。
        
        參數:
            trace: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        prompt_tokens = int(trace.get("prompt_tokens", 0) or 0)
        completion_tokens = int(trace.get("completion_tokens", 0) or 0)
        self.shared_token_usage.append(
            {
                **trace,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(trace.get("total_tokens", prompt_tokens + completion_tokens) or 0),
            }
        )

    def record_latency(self, record: LatencyRecord | dict[str, Any]) -> None:
        payload = record.to_dict() if isinstance(record, LatencyRecord) else dict(record or {})
        self.shared_latency_records.append(payload)

    def record_trace(self, record: TraceRecord | dict[str, Any]) -> None:
        payload = record.to_dict() if isinstance(record, TraceRecord) else dict(record or {})
        self.shared_trace_records.append(payload)

    @contextmanager
    def measure(
        self,
        name: str,
        *,
        stage: str = "",
        category: str = "",
        event_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        input_summary: str = "",
    ):
        task_context = self.get_task_context()
        latency = LatencyRecord(
            name=name,
            stage=stage,
            category=category,
            start_time=now_ms(),
            metadata={
                "benchmark": task_context.benchmark,
                "task_id": task_context.task_id,
                **dict(metadata or {}),
            },
        )
        try:
            yield latency
        except Exception as exc:
            latency.finish(success=False, error=str(exc))
            self.record_latency(latency)
            self.record_trace(
                TraceRecord(
                    task_id=task_context.task_id,
                    benchmark=task_context.benchmark,
                    stage=stage,
                    event_type=event_type or category,
                    name=name,
                    start_time=latency.start_time,
                    end_time=latency.end_time,
                    duration_ms=latency.duration_ms,
                    success=False,
                    error=str(exc),
                    input_summary=input_summary,
                    output_summary=str(latency.metadata.get("output_summary", "") or ""),
                    token_usage=dict(latency.metadata.get("token_usage", {}) or {}),
                    metadata=dict(latency.metadata),
                )
            )
            raise
        else:
            latency.finish(success=True)
            self.record_latency(latency)
            self.record_trace(
                TraceRecord(
                    task_id=task_context.task_id,
                    benchmark=task_context.benchmark,
                    stage=stage,
                    event_type=event_type or category,
                    name=name,
                    start_time=latency.start_time,
                    end_time=latency.end_time,
                    duration_ms=latency.duration_ms,
                    success=True,
                    input_summary=input_summary,
                    output_summary=str(latency.metadata.get("output_summary", "") or ""),
                    token_usage=dict(latency.metadata.get("token_usage", {}) or {}),
                    metadata=dict(latency.metadata),
                )
            )

    def token_usage_summary(self) -> dict[str, Any]:
        """處理 token_usage_summary 流程並回傳結果。
        
        回傳:
            此函式的處理結果。
        """
        prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in self.shared_token_usage)
        completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in self.shared_token_usage)
        by_stage: dict[str, dict[str, int]] = {}
        by_model: dict[str, dict[str, int]] = {}
        for item in self.shared_token_usage:
            stage = str(item.get("stage", "unknown") or "unknown")
            model = str(item.get("model_name", "unknown") or "unknown")
            for bucket, key in ((by_stage, stage), (by_model, model)):
                stats = bucket.setdefault(key, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0})
                stats["prompt_tokens"] += int(item.get("prompt_tokens", 0) or 0)
                stats["completion_tokens"] += int(item.get("completion_tokens", 0) or 0)
                stats["total_tokens"] += int(item.get("total_tokens", 0) or 0)
                stats["calls"] += 1
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "calls": len(self.shared_token_usage),
            "by_stage": by_stage,
            "by_model": by_model,
            "records": list(self.shared_token_usage),
        }

    def latency_summary(self) -> dict[str, Any]:
        by_category: dict[str, dict[str, Any]] = {}
        total_duration_ms = 0.0
        for record in self.shared_latency_records:
            category = str(record.get("category") or "unknown")
            duration_ms = float(record.get("duration_ms", 0.0) or 0.0)
            total_duration_ms += duration_ms
            stats = by_category.setdefault(
                category,
                {"count": 0, "duration_ms": 0.0, "max_duration_ms": 0.0},
            )
            stats["count"] += 1
            stats["duration_ms"] += duration_ms
            stats["max_duration_ms"] = max(stats["max_duration_ms"], duration_ms)

        for stats in by_category.values():
            count = int(stats.get("count", 0) or 0)
            stats["avg_duration_ms"] = stats["duration_ms"] / count if count else 0.0

        return {
            "count": len(self.shared_latency_records),
            "duration_ms": total_duration_ms,
            "by_category": by_category,
            "records": list(self.shared_latency_records),
        }

    def clear_stage2_shared_state(self) -> None:
        """清除 clear_stage2_shared_state 相關狀態。
        
        回傳:
            此函式的處理結果。
        """
        self.shared_stage2_search_bundle = None
        self.current_stage2_stage1_result = None
        self.current_stage2_top_k_answers = []
        self.current_stage2_judge_scores = []

    def clear_observability_records(self) -> None:
        self.shared_tool_traces.clear()
        self.shared_memory_reads.clear()
        self.shared_memory_writes.clear()
        self.shared_token_usage.clear()
        self.shared_latency_records.clear()
        self.shared_trace_records.clear()

    def clear_current_context(self) -> None:
        """清除 clear_current_context 相關狀態。
        
        回傳:
            此函式的處理結果。
        """
        self.current_task_context = TaskContext()
        self.current_attachment = None
        self.shared_attachment_bundle = None

    def should_include_stage1_attachment(self, is_first_round: bool) -> bool:
        """處理 should_include_stage1_attachment 流程並回傳結果。
        
        參數:
            is_first_round: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
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
        """準備 prepare_shared_attachment_evidence 流程所需的資料。
        
        參數:
            question: 此流程需要使用的輸入資料。
            agent_id: 此流程需要使用的輸入資料。
            stage: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
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
        """準備 prepare_shared_stage2_search 流程所需的資料。
        
        參數:
            question: 此流程需要使用的輸入資料。
            router_model_name: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
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
        """處理 dedupe_memory_records 流程並回傳結果。
        
        參數:
            records: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
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
