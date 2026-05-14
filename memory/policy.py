from __future__ import annotations

from typing import Any, Mapping

from parser import DecisionParser, StageParser


_DECISION_PARSER = DecisionParser()
_STAGE_PARSER = StageParser()


def should_write_stage1_memory(*args: Any, **kwargs: Any) -> bool:
    """
    負責執行 memory.policy 中的 should_write_stage1_memory 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
    
    Args:
        *args: 記憶系統提供的檢索結果、寫入資料或操作介面。
        **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return False


def should_write_stage2_memory(candidate: Mapping[str, Any]) -> bool:
    """
    負責執行 memory.policy 中的 should_write_stage2_memory 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
    
    Args:
        candidate: 模型、節點或工具產生的候選回覆內容。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return False


def should_write_final_memory(decision: Mapping[str, Any]) -> bool:
    """
    負責執行 memory.policy 中的 should_write_final_memory 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
    
    Args:
        decision: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return False


def build_memory_records(
    *,
    question: str,
    source_stage: str,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    負責執行 memory.policy 中的 build_memory_records 流程，建立記憶圖或任務記錄結構，供後續檢索、寫入與提示注入使用。
    
    Args:
        question: 目前要處理的任務、問題或查詢文字。
        source_stage: 記憶系統提供的檢索結果、寫入資料或操作介面。
        payload: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
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
    """
    負責執行 memory.policy 中的 _build_stage2_memory_records 流程，依照 memory.policy 的流程需求處理 _build_stage2_memory_records 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        question: 目前要處理的任務、問題或查詢文字。
        payload: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
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
    """
    負責執行 memory.policy 中的 _build_final_memory_records 流程，依照 memory.policy 的流程需求處理 _build_final_memory_records 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        question: 目前要處理的任務、問題或查詢文字。
        payload: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
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
    """
    負責執行 memory.policy 中的 _score_to_importance 流程，依照 memory.policy 的流程需求處理 _score_to_importance 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        score: 記憶系統提供的檢索結果、寫入資料或操作介面。
        floor: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 float。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    bounded = max(0.0, min(10.0, float(score)))
    return max(floor, min(1.0, floor + (bounded / 10.0) * (1.0 - floor)))


def _summarize_text(text: str, *, fallback: str = "", max_len: int = 240) -> str:
    """
    負責執行 memory.policy 中的 _summarize_text 流程，依照 memory.policy 的流程需求處理 _summarize_text 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        text: 記憶系統提供的檢索結果、寫入資料或操作介面。
        fallback: 記憶系統提供的檢索結果、寫入資料或操作介面。
        max_len: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    candidate = (text or "").strip()
    if not candidate:
        candidate = (fallback or "").strip()
    if not candidate:
        return ""

    single_line = " ".join(candidate.split())
    if len(single_line) <= max_len:
        return single_line
    return single_line[: max_len - 3].rstrip() + "..."
