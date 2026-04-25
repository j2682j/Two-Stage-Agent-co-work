from __future__ import annotations

from typing import Any, Mapping

from parser import DecisionParser, StageParser


_DECISION_PARSER = DecisionParser()
_STAGE_PARSER = StageParser()


def should_write_stage1_memory(*args: Any, **kwargs: Any) -> bool:
    return False


def should_write_stage2_memory(candidate: Mapping[str, Any]) -> bool:
    return False


def should_write_final_memory(decision: Mapping[str, Any]) -> bool:
    return False


def build_memory_records(
    *,
    question: str,
    source_stage: str,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if source_stage == "stage2":
        return _build_stage2_memory_records(question=question, payload=payload)
    if source_stage == "final":
        return _build_final_memory_records(question=question, payload=payload)
    return []


def _build_stage2_memory_records(
    *,
    question: str,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    answer = str(payload.get("answer", "") or "").strip()
    reply = str(payload.get("reply", "") or "").strip()
    agent_idx = payload.get("agent_idx")
    judge_score = float(payload.get("stage2_judge_score", 0.0) or 0.0)

    content = (
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
        f"Reasoning summary: {_summarize_text(reply, fallback=answer)}"
    )
    metadata = {
        "source_stage": "stage2",
        "source_agent_idx": agent_idx,
        "question": question,
        "answer": answer,
        "judge_score": judge_score,
        "is_acceptable": bool(payload.get("stage2_judge_is_acceptable", False)),
        "memory_role": "high_confidence_candidate",
    }
    importance = _score_to_importance(judge_score, floor=0.8)

    return [
        {
            "memory_type": "working",
            "content": content,
            "importance": importance,
            "metadata": metadata,
            "auto_classify": False,
        }
    ]


def _build_final_memory_records(
    *,
    question: str,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    final_result = str(payload.get("final_result", "") or "").strip()
    final_reply = str(payload.get("final_reply", "") or "").strip()
    selected_agent_idx = payload.get("selected_agent_idx")

    working_content = (
        f"Question: {question}\n"
        f"Final answer: {final_result}\n"
        f"Decision summary: {_summarize_text(final_reply, fallback=final_result)}"
    )
    semantic_content = (
        f"Verified answer for question '{question}': {final_result}\n"
        f"Evidence summary: {_summarize_text(final_reply, fallback=final_result)}"
    )

    common_metadata = {
        "source_stage": "final",
        "selected_agent_idx": selected_agent_idx,
        "question": question,
        "answer": final_result,
        "memory_role": "final_decision",
        "success": bool(payload.get("success", False)),
    }

    return [
        {
            "memory_type": "working",
            "content": working_content,
            "importance": 1.0,
            "metadata": {
                **common_metadata,
                "memory_kind": "working_summary",
            },
            "auto_classify": False,
        },
        {
            "memory_type": "semantic",
            "content": semantic_content,
            "importance": 0.95,
            "metadata": {
                **common_metadata,
                "memory_kind": "verified_fact",
            },
            "auto_classify": False,
        },
    ]


def _score_to_importance(score: float, floor: float = 0.8) -> float:
    bounded = max(0.0, min(10.0, float(score)))
    return max(floor, min(1.0, floor + (bounded / 10.0) * (1.0 - floor)))


def _summarize_text(text: str, *, fallback: str = "", max_len: int = 240) -> str:
    candidate = (text or "").strip()
    if not candidate:
        candidate = (fallback or "").strip()
    if not candidate:
        return ""

    single_line = " ".join(candidate.split())
    if len(single_line) <= max_len:
        return single_line
    return single_line[: max_len - 3].rstrip() + "..."
