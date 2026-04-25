from __future__ import annotations

from typing import Any


class DecisionTraceBuilder:
    def build_critic_round_step(
        self,
        *,
        round_idx: int,
        solver_agent_idx: int | None,
        critiques: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "step": "critic_round",
            "round_idx": round_idx,
            "solver_agent_idx": solver_agent_idx,
            "critiques": critiques,
        }

    def build_solver_revision_step(
        self,
        *,
        round_idx: int,
        solver_agent_idx: int | None,
        revised_reply: str | None,
        revised_answer: str,
    ) -> dict[str, Any]:
        return {
            "step": "solver_revision",
            "round_idx": round_idx,
            "solver_agent_idx": solver_agent_idx,
            "revised_reply": revised_reply,
            "revised_answer": revised_answer,
        }

    def build_critic_fallback(
        self,
        *,
        critic_agent_idx: int | None,
        critique: str,
        revised_answer: str,
    ) -> dict[str, Any]:
        return {
            "critic_agent_idx": critic_agent_idx,
            "agree": False,
            "critique": critique,
            "revised_answer": revised_answer,
        }
