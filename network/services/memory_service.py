from __future__ import annotations

import hashlib
from typing import Any

from memory.retrieval_cache import (
    MemoryRetrievalCache,
    MemoryRetrievalCacheKey,
    build_question_hash,
)


class MemoryService:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.retrieval_cache = MemoryRetrievalCache()
        self._cache_task_id: str | None = None

    def retrieve_context(
        self,
        *,
        question: str,
        stage: str,
        injection_target: str,
        source: str | None = None,
        task_id: str | None = None,
        attachment_type: str | None = None,
        limit: int = 3,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        runtime = self.runtime
        graph_memory = getattr(runtime, "graph_memory", None)
        normalized_question = str(question or "").strip()
        if graph_memory is None or not normalized_question:
            return None

        task_context = runtime.get_task_context() if hasattr(runtime, "get_task_context") else None
        resolved_task_id = str(task_id or getattr(task_context, "task_id", "") or "").strip()
        if not resolved_task_id:
            resolved_task_id = self._task_id_from_text(normalized_question)
        resolved_source = str(source or getattr(task_context, "source_label", "") or "system").strip() or "system"
        benchmark = str(getattr(task_context, "benchmark", "") or "").strip()
        resolved_attachment_type = attachment_type or getattr(task_context, "attachment_type", None)
        self._clear_cache_if_task_changed(resolved_task_id)
        cache_key = MemoryRetrievalCacheKey(
            benchmark=benchmark,
            task_id=resolved_task_id,
            stage=str(stage or "").strip(),
            injection_target=str(injection_target or "").strip(),
            source=resolved_source,
            question_hash=build_question_hash(normalized_question),
            attachment_type=str(resolved_attachment_type or "").strip(),
            limit=int(limit or 0),
        )

        result, cache_hit = self.retrieval_cache.get_or_compute(
            cache_key,
            lambda: self._retrieve_from_graph_memory(
                graph_memory=graph_memory,
                resolved_task_id=resolved_task_id,
                normalized_question=normalized_question,
                resolved_source=resolved_source,
                benchmark=benchmark,
                resolved_attachment_type=resolved_attachment_type,
                limit=limit,
                injection_target=injection_target,
                stage=stage,
                agent_id=agent_id,
            ),
        )
        if cache_hit:
            self._record_cache_hit_latency(
                result,
                stage=stage,
                resolved_source=resolved_source,
                injection_target=injection_target,
                agent_id=agent_id,
                normalized_question=normalized_question,
                cache_key=cache_key,
            )

        self._record_memory_read(
            result,
            stage=stage,
            source="graph_memory",
            task_id=result.get("task_id") or resolved_task_id,
            agent_id=agent_id,
            cache_hit=cache_hit,
            cache_key=cache_key.short_id(),
        )
        return result

    def clear_cache(self) -> None:
        self.retrieval_cache.clear()
        self._cache_task_id = None

    def _clear_cache_if_task_changed(self, task_id: str) -> None:
        if self._cache_task_id is None:
            self._cache_task_id = task_id
            return
        if self._cache_task_id != task_id:
            self.retrieval_cache.clear()
            self._cache_task_id = task_id

    def _retrieve_from_graph_memory(
        self,
        *,
        graph_memory: Any,
        resolved_task_id: str,
        normalized_question: str,
        resolved_source: str,
        benchmark: str,
        resolved_attachment_type: str | None,
        limit: int,
        injection_target: str,
        stage: str,
        agent_id: str | None,
    ) -> dict[str, Any]:
        runtime = self.runtime
        with runtime.measure(
            "graph_memory_retrieve_context",
            stage=stage,
            category="memory_retrieval",
            event_type="memory_retrieval",
            metadata={
                "source": resolved_source,
                "injection_target": injection_target,
                "agent_id": agent_id or "",
                "cache_hit": False,
            },
            input_summary=normalized_question[:240],
        ) as latency:
            result = graph_memory.retrieve_context(
                task_id=resolved_task_id,
                input_text=normalized_question,
                source=resolved_source,
                benchmark=benchmark,
                attachment_type=resolved_attachment_type,
                limit=limit,
                injection_target=injection_target,
            )
            latency.metadata["related_task_count"] = len(result.get("related_task_ids", []) or [])
            latency.metadata["insight_count"] = len(result.get("insights", []) or [])
            return result

    def _record_cache_hit_latency(
        self,
        result: dict[str, Any],
        *,
        stage: str,
        resolved_source: str,
        injection_target: str,
        agent_id: str | None,
        normalized_question: str,
        cache_key: MemoryRetrievalCacheKey,
    ) -> None:
        runtime = self.runtime
        if not hasattr(runtime, "measure"):
            return
        with runtime.measure(
            "graph_memory_retrieve_context_cache_hit",
            stage=stage,
            category="memory_retrieval",
            event_type="memory_retrieval",
            metadata={
                "source": resolved_source,
                "injection_target": injection_target,
                "agent_id": agent_id or "",
                "cache_hit": True,
                "cache_key": cache_key.short_id(),
            },
            input_summary=normalized_question[:240],
        ) as latency:
            latency.metadata["related_task_count"] = len(result.get("related_task_ids", []) or [])
            latency.metadata["insight_count"] = len(result.get("insights", []) or [])

    def _record_memory_read(
        self,
        result: dict[str, Any],
        *,
        stage: str,
        source: str,
        task_id: str,
        agent_id: str | None = None,
        cache_hit: bool = False,
        cache_key: str = "",
    ) -> None:
        runtime = self.runtime
        retrieval = result.get("retrieval", {}) or {}
        insights = result.get("insights", []) or []
        trace = {
            "stage": stage,
            "source": source,
            "task_id": task_id,
            "task_type": retrieval.get("task_type"),
            "trigger_terms": retrieval.get("trigger_terms", []),
            "related_task_ids": result.get("related_task_ids", []),
            "insight_ids": [
                item.get("insight_id")
                for item in insights
                if isinstance(item, dict) and item.get("insight_id")
            ],
            "seed_task_hits": result.get("seed_task_hits", []),
            "expanded_task_hits": result.get("expanded_task_hits", []),
            "cache_hit": bool(cache_hit),
        }
        if cache_key:
            trace["cache_key"] = cache_key
        if agent_id:
            trace["agent_id"] = agent_id
        runtime.record_memory_read(trace)

    def _task_id_from_text(self, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"task_{digest}"
