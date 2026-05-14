from __future__ import annotations

from typing import Any


class DecisionTraceBuilder:
    """
    負責在 builder.trace.decision_trace_builder 中封裝 DecisionTraceBuilder，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def build_critic_round_step(
        self,
        *,
        round_idx: int,
        solver_agent_idx: int | None,
        critiques: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        負責執行 DecisionTraceBuilder 中的 build_critic_round_step 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            round_idx: 此流程需要使用的輸入資料。
            solver_agent_idx: 此流程需要使用的輸入資料。
            critiques: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 DecisionTraceBuilder 中的 build_solver_revision_step 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            round_idx: 此流程需要使用的輸入資料。
            solver_agent_idx: 此流程需要使用的輸入資料。
            revised_reply: 此流程需要使用的輸入資料。
            revised_answer: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 DecisionTraceBuilder 中的 build_critic_fallback 流程，建立後續流程需要的物件、資料結構或輸出區塊。
        
        Args:
            critic_agent_idx: 此流程需要使用的輸入資料。
            critique: 此流程需要使用的輸入資料。
            revised_answer: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "critic_agent_idx": critic_agent_idx,
            "agree": False,
            "critique": critique,
            "revised_answer": revised_answer,
        }
