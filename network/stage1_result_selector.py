from __future__ import annotations

import re
from typing import Any

from parser import try_parse_json
from utils.network_utils import answer_equivalence

from .slm_agent import SLM_4b_Agent


class Stage1ResultSelector:
    """
    負責在 network.stage1_result_selector 中封裝 Stage1ResultSelector，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        helper: 評估、推理或工具執行後產生的結果與分數資料。
        judge_model_name: 評估、推理或工具執行後產生的結果與分數資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, helper, judge_model_name: str = "gpt-oss:20b"):
        """
        負責執行 Stage1ResultSelector 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            helper: 評估、推理或工具執行後產生的結果與分數資料。
            judge_model_name: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.helper = helper
        self.judge_model_name = judge_model_name
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def select_stage1_result_with_judge(
        self,
        nodes,
        question: str | None = None,
        fallback_answer: str | None = None,
    ) -> str | None:
        """
        負責執行 Stage1ResultSelector 中的 select_stage1_result_with_judge 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            nodes: 評估、推理或工具執行後產生的結果與分數資料。
            question: 目前要處理的任務、問題或查詢文字。
            fallback_answer: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        candidates = []
        for idx, node in enumerate(nodes):
            answer = str(node.get_answer() or "").strip()
            if not getattr(node, "active", False) or not answer:
                continue

            candidates.append(
                {
                    "node_idx": idx,
                    "answer": answer,
                    "judge_ok": bool(getattr(node, "stage1_judge_is_acceptable", False)),
                    "judge_score": float(getattr(node, "stage1_judge_score", 0.0) or 0.0),
                    "judge_adjusted_score": float(
                        getattr(node, "stage1_judge_adjusted_score", getattr(node, "stage1_judge_score", 0.0)) or 0.0
                    ),
                }
            )

        if not candidates:
            return fallback_answer

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        judge_pre_reason = self._judge_stage1_pre_reason(
            question=question or "",
            candidates=candidates,
        )
        judge_target_answer = self._normalize_text(judge_pre_reason.get("provisional_answer", ""))

        approved = [item for item in candidates if item["judge_ok"]]
        if approved:
            approved_answers = [item["answer"] for item in approved]
            approved_clusters = self.helper.cluster_answers(approved_answers)
            if approved_clusters:
                best_cluster = None
                best_key = None
                for cluster in approved_clusters:
                    members = [approved[pos] for pos in cluster["member_indices"]]
                    cluster_best_score = max(member["judge_adjusted_score"] for member in members)
                    cluster_size = len(members)
                    match_bonus = 0.0
                    if judge_target_answer:
                        for member in members:
                            if answer_equivalence(member["answer"], judge_target_answer):
                                match_bonus = 1.0
                                break
                    key = (match_bonus, cluster_best_score, cluster_size)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_cluster = {
                            "canonical_answer": cluster["canonical_answer"],
                            "members": [member["answer"] for member in members],
                            "member_indices": cluster["member_indices"],
                        }

                if best_cluster is not None:
                    representative = self.helper.select_cluster_representative([best_cluster])
                    if representative:
                        return representative

        all_answers = [item["answer"] for item in candidates]
        clusters = self.helper.cluster_answers(all_answers)
        if judge_target_answer and clusters:
            matching_clusters = []
            for cluster in clusters:
                if answer_equivalence(cluster["canonical_answer"], judge_target_answer):
                    matching_clusters.append(cluster)
            if matching_clusters:
                representative = self.helper.select_cluster_representative(matching_clusters)
                if representative:
                    return representative

        representative = self.helper.select_cluster_representative(clusters)
        return representative or fallback_answer

    def _judge_stage1_pre_reason(
        self,
        question: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        負責執行 Stage1ResultSelector 中的 _judge_stage1_pre_reason 流程，依照 Stage1ResultSelector 的流程需求處理 _judge_stage1_pre_reason 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            candidates: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not question or not candidates:
            return {
                "provisional_answer": "",
                "judge_reasoning": "",
                "used_fallback": True,
            }

        candidate_lines = [f"- Candidate {item['node_idx']}: {item['answer']}" for item in candidates]

        prompt = f"""
You are selecting the best stage-1 candidate answer in a multi-agent reasoning system.

Before selecting, do a very short independent reasoning pass for the question and produce a provisional final answer.
Then use that provisional answer only as a reference for selecting among the candidates.

Rules:
- Keep the reasoning brief.
- The provisional answer should use the exact unit requested by the question.
- Do not return markdown.
- Return valid JSON only.

Return exactly this schema:
{{
  "provisional_answer": "string",
  "judge_reasoning": "string"
}}

Question:
{question}

Candidate answers:
{chr(10).join(candidate_lines)}
        """.strip()

        try:
            judge_agent = SLM_4b_Agent(model_name=self.judge_model_name)
            raw, prompt_tokens, completion_tokens = judge_agent.invoke_with_usage(
                [
                    {"role": "system", "content": "You are a strict JSON-only stage-1 selector judge."},
                    {"role": "user", "content": prompt},
                ]
            )
            self.last_prompt_tokens = prompt_tokens
            self.last_completion_tokens = completion_tokens
            parsed = try_parse_json(raw)
            if isinstance(parsed, dict):
                return {
                    "provisional_answer": self._normalize_text(parsed.get("provisional_answer", "")),
                    "judge_reasoning": self._normalize_text(parsed.get("judge_reasoning", "")),
                    "used_fallback": False,
                }
        except Exception:
            pass

        return {
            "provisional_answer": "",
            "judge_reasoning": "",
            "used_fallback": True,
        }

    def _normalize_text(self, text: Any) -> str:
        """
        負責執行 Stage1ResultSelector 中的 _normalize_text 流程，依照 Stage1ResultSelector 的流程需求處理 _normalize_text 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()
