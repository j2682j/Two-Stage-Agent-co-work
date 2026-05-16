from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class TaskContext:
    benchmark: str = ""
    task_id: str = ""
    category: str = ""
    task_type: str = ""
    source: str = ""
    question: str = ""
    functions: list[dict[str, Any]] = field(default_factory=list)
    attachment: dict[str, Any] = field(default_factory=dict)
    memory_namespace: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "TaskContext" | None) -> "TaskContext":
        if isinstance(data, TaskContext):
            return data
        if not isinstance(data, Mapping):
            return cls()

        metadata = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), Mapping) else {}
        known_keys = {
            "benchmark",
            "task_id",
            "id",
            "sample_id",
            "category",
            "task_type",
            "source",
            "question",
            "functions",
            "function",
            "attachment",
            "memory_namespace",
            "metadata",
        }
        for key, value in data.items():
            if key not in known_keys:
                metadata.setdefault(str(key), value)

        functions = data.get("functions", data.get("function", [])) or []
        if not isinstance(functions, list):
            functions = []
        attachment = data.get("attachment") or {}
        if not isinstance(attachment, Mapping):
            attachment = {}

        return cls(
            benchmark=str(data.get("benchmark", "") or "").strip(),
            task_id=str(data.get("task_id") or data.get("id") or data.get("sample_id") or "").strip(),
            category=str(data.get("category", "") or "").strip(),
            task_type=str(data.get("task_type", "") or "").strip(),
            source=str(data.get("source", "") or "").strip(),
            question=str(data.get("question", "") or ""),
            functions=[dict(item) for item in functions if isinstance(item, Mapping)],
            attachment=dict(attachment),
            memory_namespace=str(data.get("memory_namespace", "") or "").strip(),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.metadata)
        data.update(
            {
                "benchmark": self.benchmark,
                "task_id": self.task_id,
                "category": self.category,
                "task_type": self.task_type,
                "source": self.source,
                "question": self.question,
                "functions": list(self.functions),
                "attachment": dict(self.attachment),
                "memory_namespace": self.memory_namespace,
                "metadata": dict(self.metadata),
            }
        )
        return {key: value for key, value in data.items() if value not in ("", None, [], {})}

    @property
    def benchmark_upper(self) -> str:
        return self.benchmark.upper()

    @property
    def source_label(self) -> str:
        return (self.benchmark or self.source or "system").strip().lower() or "system"

    @property
    def attachment_type(self) -> str | None:
        for key in ("extension", "file_extension", "type"):
            value = str(self.attachment.get(key, "") or "").strip().lower().lstrip(".")
            if value:
                return value
        path = str(self.attachment.get("path", "") or self.attachment.get("file_path", "") or "").strip()
        if "." in path:
            return path.rsplit(".", 1)[-1].lower()
        return None

    def with_question(self, question: str) -> "TaskContext":
        if self.question:
            return self
        data = self.to_dict()
        data["question"] = question
        return TaskContext.from_dict(data)
