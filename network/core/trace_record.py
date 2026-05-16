from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass(slots=True)
class LatencyRecord:
    name: str
    stage: str = ""
    category: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, *, success: bool = True, error: str = "") -> None:
        self.end_time = now_ms()
        self.duration_ms = max(0.0, self.end_time - self.start_time)
        self.success = success
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "category": self.category,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class TraceRecord:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = ""
    benchmark: str = ""
    stage: str = ""
    event_type: str = ""
    name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error: str = ""
    input_summary: str = ""
    output_summary: str = ""
    token_usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "stage": self.stage,
            "event_type": self.event_type,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "token_usage": dict(self.token_usage),
            "metadata": dict(self.metadata),
        }


def summarize_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
