from __future__ import annotations

from typing import Any


class MemoryUsageTracer:
    """Trace whether a new sample retrieves memories written by previous samples."""

    def __init__(self, memory_tool: Any):
        self.memory_tool = memory_tool
        self.memory_manager = getattr(memory_tool, "memory_manager", None)
        self.original_retrieve = None
        self.current_sample = None
        self.previous_markers: list[str] = []
        self.used_previous_memory = False
        self.hit_records: list[dict[str, Any]] = []

    def install(self) -> None:
        if self.memory_manager is None or self.original_retrieve is not None:
            return

        self.original_retrieve = self.memory_manager.retrieve_memories

        def traced_retrieve(*args, **kwargs):
            results = self.original_retrieve(*args, **kwargs)
            if self.current_sample and self.previous_markers:
                for memory in results or []:
                    content = getattr(memory, "content", "") or ""
                    for marker in self.previous_markers:
                        if marker and marker in content:
                            self.used_previous_memory = True
                            self.hit_records.append(
                                {
                                    "query": kwargs.get("query", ""),
                                    "marker": marker,
                                    "memory_type": getattr(memory, "memory_type", ""),
                                    "preview": content[:160],
                                }
                            )
                            break
            return results

        self.memory_manager.retrieve_memories = traced_retrieve

    def start_sample(self, sample: dict[str, Any]) -> None:
        self.current_sample = sample
        self.used_previous_memory = False
        self.hit_records = []

    def finish_sample(self) -> None:
        self.current_sample = None

    def register_feedback_marker(self, sample: dict[str, Any], normalized_question: str) -> None:
        task_id = sample.get("task_id", "")
        question = sample.get("question", "")
        markers = [task_id, normalized_question]
        if question:
            markers.append(question[:80])
        for marker in markers:
            if marker and marker not in self.previous_markers:
                self.previous_markers.append(marker)


def print_memory_stats(memory_tool: Any) -> None:
    memory_manager = getattr(memory_tool, "memory_manager", None)
    if memory_manager is None:
        print("   [MEMORY][stats] unavailable")
        return

    stats = memory_manager.get_memory_stats()
    by_type = stats.get("memories_by_type", {}) or {}
    summary = {
        memory_type: details.get("count", 0)
        for memory_type, details in by_type.items()
    }
    print(f"   [MEMORY][stats] total={stats.get('total_memories', 0)} by_type={summary}")


def print_memory_debug(memory_tool: Any, normalized_question: str) -> None:
    for memory_type in ["working", "semantic", "episodic"]:
        memory_debug = memory_tool.run(
            {
                "action": "search",
                "query": normalized_question,
                "memory_type": memory_type,
                "limit": 5,
            }
        )
        print(f"   [MEMORY][{memory_type}]")
        print(f"   {memory_debug}")


def print_memory_hits(tracer: MemoryUsageTracer | None) -> None:
    if tracer is None:
        return
    print(f"   used_previous_memory={tracer.used_previous_memory}")
    for hit in tracer.hit_records[:5]:
        print(
            "   [MEMORY-HIT] "
            f"type={hit['memory_type']} "
            f"marker={hit['marker']!r} "
            f"query={hit['query']!r}"
        )
        print(f"      preview={hit['preview']}")
