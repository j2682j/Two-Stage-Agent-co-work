from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDecisionMaker(ABC):
    """
    負責在 decisionmaker.base_decision_maker 中封裝 BaseDecisionMaker，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        max_inner_turns: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    name: str = "base_decision"

    def __init__(self, max_inner_turns: int = 1):
        """
        負責執行 BaseDecisionMaker 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            max_inner_turns: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        prompt_contract: Any | None = None,
        task_context: Any | None = None,
    ) -> dict[str, Any]:
        """
        負責執行 BaseDecisionMaker 中的 decide 流程，依照 BaseDecisionMaker 的流程需求處理 decide 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            stage1_result: 評估、推理或工具執行後產生的結果與分數資料。
            top_k_outputs: 控制檢索、篩選或輸出數量的數值參數。
            top_k_indices: 控制檢索、篩選或輸出數量的數值參數。
            importance_scores: 此流程需要使用的輸入資料。
            memory_context: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    def _successful_outputs(
        self,
        top_k_outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        負責執行 BaseDecisionMaker 中的 _successful_outputs 流程，依照 BaseDecisionMaker 的流程需求處理 _successful_outputs 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            top_k_outputs: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 BaseDecisionMaker 中的 _build_result 流程，依照 BaseDecisionMaker 的流程需求處理 _build_result 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            mode: 評估、推理或工具執行後產生的結果與分數資料。
            success: 評估、推理或工具執行後產生的結果與分數資料。
            final_answer: 評估、推理或工具執行後產生的結果與分數資料。
            final_reply: 評估、推理或工具執行後產生的結果與分數資料。
            selected_agent_idx: 評估、推理或工具執行後產生的結果與分數資料。
            selected_indices: 評估、推理或工具執行後產生的結果與分數資料。
            critiques: 評估、推理或工具執行後產生的結果與分數資料。
            intermediate_steps: 評估、推理或工具執行後產生的結果與分數資料。
            prompt_tokens: 評估、推理或工具執行後產生的結果與分數資料。
            completion_tokens: 評估、推理或工具執行後產生的結果與分數資料。
            error: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
