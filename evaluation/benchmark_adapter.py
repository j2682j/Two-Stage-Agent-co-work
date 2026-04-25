from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBenchmarkAdapter(ABC):
    """Benchmark 介面轉接器的抽象基底類別。"""

    def __init__(self, agent: Any, name: str | None = None):
        self.agent = agent
        self.name = name or getattr(agent, "name", agent.__class__.__name__)

    @abstractmethod
    def run(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def normalize_question(self, question: str) -> str:
        raise NotImplementedError

    def record_evaluation_feedback(
        self,
        *,
        benchmark: str,
        sample: dict[str, Any],
        sample_result: dict[str, Any],
    ) -> None:
        return None





