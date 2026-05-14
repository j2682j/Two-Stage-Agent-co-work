from __future__ import annotations

import json
from typing import Any

from .builder import PromptBuilder, PromptPacket


class _CriticPromptBuilder(PromptBuilder):
    """
    負責在 prompt.decision_prompt_builder 中封裝 _CriticPromptBuilder，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def gather(self, **kwargs) -> list[PromptPacket]:
        """
        負責執行 _CriticPromptBuilder 中的 gather 流程，依照 _CriticPromptBuilder 的流程需求處理 gather 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [
            PromptPacket(self._normalize_text(kwargs.get("question", "")), "question", priority=10.0),
            PromptPacket(self._normalize_text(kwargs.get("stage1_result", "")), "stage1_result", priority=8.0),
            PromptPacket(self._normalize_text(kwargs.get("solver_answer", "")), "solver_answer", priority=9.0),
            PromptPacket(self._normalize_text(kwargs.get("critic_answer", "")), "critic_answer", priority=9.0),
            PromptPacket(self._normalize_text(kwargs.get("memory_context", "")), "memory_context", priority=7.0),
        ]

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        """
        負責執行 _CriticPromptBuilder 中的 select 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return packets

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        """
        負責執行 _CriticPromptBuilder 中的 structure 流程，依照 _CriticPromptBuilder 的流程需求處理 structure 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        structured = {
            "question": "",
            "stage1_result": "",
            "solver_answer": "",
            "critic_answer": "",
            "memory_context": "",
        }
        for packet in packets:
            structured[packet.packet_type] = packet.content
        return structured

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        負責執行 _CriticPromptBuilder 中的 compress 流程，依照 _CriticPromptBuilder 的流程需求處理 compress 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            structured: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return structured

    def render(self, compressed: dict[str, Any], **kwargs):
        """
        負責執行 _CriticPromptBuilder 中的 render 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            compressed: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        system_prompt = (
            "You are a careful critic reviewing a solver's answer. "
            "Return plain text only using the required key=value format. "
            "Do not use JSON."
        )
        user_prompt = f"""
Question:
{compressed["question"]}

Stage-1 answer:
{compressed["stage1_result"]}

Relevant memory lessons and cases:
{compressed["memory_context"] or "No relevant memory."}

Solver answer:
{compressed["solver_answer"]}

Your own answer:
{compressed["critic_answer"]}

Instructions:
1. Compare the solver answer against your own answer.
2. If the solver answer is already acceptable, set AGREE=true.
3. If not, provide a concise critique and a better revised answer.
4. Use memory as lessons or error checks, not as direct answer lookup.
5. If a relevant lesson applies, treat it as a constraint against repeating the same mistake.
6. If the solver answer conflicts with a relevant lesson, set AGREE=false unless current evidence clearly overrides that lesson.
7. If memory suggests a likely mistake pattern, explicitly account for it in your critique.
8. Return plain text only in exactly this format:

AGREE=<true or false>
CRITIQUE=<brief critique>
REVISED_ANSWER=<better answer or empty>

Rules:
- Do not use JSON.
- Keep CRITIQUE short.
- REVISED_ANSWER may be empty if you agree.
 - Do not copy old answers from memory unless they are independently supported by the current question.
        """.strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class _SolverRevisionPromptBuilder(PromptBuilder):
    """
    負責在 prompt.decision_prompt_builder 中封裝 _SolverRevisionPromptBuilder，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def gather(self, **kwargs) -> list[PromptPacket]:
        """
        負責執行 _SolverRevisionPromptBuilder 中的 gather 流程，依照 _SolverRevisionPromptBuilder 的流程需求處理 gather 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        critiques = kwargs.get("critiques", []) or []
        return [
            PromptPacket(self._normalize_text(kwargs.get("question", "")), "question", priority=10.0),
            PromptPacket(self._normalize_text(kwargs.get("stage1_result", "")), "stage1_result", priority=8.0),
            PromptPacket(self._normalize_text(kwargs.get("solver_answer", "")), "solver_answer", priority=9.0),
            PromptPacket(self._normalize_text(kwargs.get("memory_context", "")), "memory_context", priority=7.0),
            PromptPacket(json.dumps(critiques, ensure_ascii=False, indent=2), "critiques", priority=8.0),
        ]

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        """
        負責執行 _SolverRevisionPromptBuilder 中的 select 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return packets

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        """
        負責執行 _SolverRevisionPromptBuilder 中的 structure 流程，依照 _SolverRevisionPromptBuilder 的流程需求處理 structure 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        structured = {
            "question": "",
            "stage1_result": "",
            "solver_answer": "",
            "memory_context": "",
            "critiques": "",
        }
        for packet in packets:
            structured[packet.packet_type] = packet.content
        return structured

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        負責執行 _SolverRevisionPromptBuilder 中的 compress 流程，依照 _SolverRevisionPromptBuilder 的流程需求處理 compress 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            structured: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        structured["critiques"] = self._compress_multiline_text(
            structured["critiques"],
            max_lines=20,
            max_chars=1800,
        )
        return structured

    def render(self, compressed: dict[str, Any], **kwargs):
        """
        負責執行 _SolverRevisionPromptBuilder 中的 render 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            compressed: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        system_prompt = (
            "You are the final solver. Revise your answer using the critics' feedback. "
            "Return plain text only using the required key=value format. "
            "Do not use JSON."
        )
        user_prompt = f"""
Question:
{compressed["question"]}

Stage-1 answer:
{compressed["stage1_result"]}

Relevant memory lessons and cases:
{compressed["memory_context"] or "No relevant memory."}

Current solver answer:
{compressed["solver_answer"]}

Critiques:
{compressed["critiques"]}

Instructions:
1. Revise the current solver answer using only useful critiques.
2. Ignore critiques that are weak or not actually improvements.
3. Use memory as lessons or error-avoidance rules, not as direct answer lookup.
4. If memory reveals a relevant mistake pattern, make sure your revision addresses that risk explicitly.
5. If a relevant lesson warns against the current answer pattern, revise away from that pattern unless the current evidence clearly supports it.
6. Return plain text only in exactly this format:

REASONING=<brief revision reasoning only>
FINAL_ANSWER=<your final answer>

Rules:
- Do not use JSON.
- Keep REASONING short.
- FINAL_ANSWER must contain only the final answer.
- FINAL_ANSWER should be the last non-empty line if possible.
 - Do not copy a past answer from memory unless it is independently justified for the current question.
        """.strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class DecisionPromptBuilder:
    """
    負責在 prompt.decision_prompt_builder 中封裝 DecisionPromptBuilder，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self):
        """
        負責執行 DecisionPromptBuilder 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.critic_prompt_builder = _CriticPromptBuilder()
        self.solver_revision_prompt_builder = _SolverRevisionPromptBuilder()

    def build_critic_messages(
        self,
        question: str,
        stage1_result: str | None,
        solver_answer: str,
        critic_answer: str,
        memory_context: str = "",
    ) -> list[dict[str, str]]:
        """
        負責執行 DecisionPromptBuilder 中的 build_critic_messages 流程，組裝提示詞內容，將任務、記憶、證據或格式要求整理成模型可讀的輸入。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            stage1_result: 評估、推理或工具執行後產生的結果與分數資料。
            solver_answer: 此流程需要使用的輸入資料。
            critic_answer: 此流程需要使用的輸入資料。
            memory_context: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, str]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.critic_prompt_builder.build(
            question=question,
            stage1_result=stage1_result or "",
            solver_answer=solver_answer,
            critic_answer=critic_answer,
            memory_context=memory_context,
        )

    def build_solver_revision_messages(
        self,
        question: str,
        stage1_result: str | None,
        solver_answer: str,
        critiques: list[dict[str, Any]],
        memory_context: str = "",
    ) -> list[dict[str, str]]:
        """
        負責執行 DecisionPromptBuilder 中的 build_solver_revision_messages 流程，組裝提示詞內容，將任務、記憶、證據或格式要求整理成模型可讀的輸入。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            stage1_result: 評估、推理或工具執行後產生的結果與分數資料。
            solver_answer: 此流程需要使用的輸入資料。
            critiques: 此流程需要使用的輸入資料。
            memory_context: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, str]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.solver_revision_prompt_builder.build(
            question=question,
            stage1_result=stage1_result or "",
            solver_answer=solver_answer,
            critiques=critiques,
            memory_context=memory_context,
        )
