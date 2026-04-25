from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDecisionMaker(ABC):
    name: str = "base_decision"

    def __init__(self, max_inner_turns: int = 1):
        self.max_inner_turns = max_inner_turns

    @abstractmethod
    def decide(
        self,
        question: str,
        stage1_result: str | None,
        top_k_outputs: list[dict[str, Any]],
        top_k_indices: list[int],
        importance_scores: list[float] | None = None,
        memory_context: str = "",
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _successful_outputs(
        self,
        top_k_outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [item for item in top_k_outputs if item.get("success") and item.get("answer")]

    def _build_result(
        self,
        *,
        mode: str,
        success: bool,
        final_answer: str = "",
        final_reply: str | None = None,
        selected_agent_idx: int | None = None,
        selected_indices: list[int] | None = None,
        critiques: list[dict[str, Any]] | None = None,
        intermediate_steps: list[dict[str, Any]] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "mode": mode,
            "success": success,
            "final_result": final_answer,
            "final_reply": final_reply,
            "selected_agent_idx": selected_agent_idx,
            "selected_indices": selected_indices or [],
            "critiques": critiques or [],
            "intermediate_steps": intermediate_steps or [],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "error": error,
        }
