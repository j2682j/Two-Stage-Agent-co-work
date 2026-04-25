from __future__ import annotations

import re
from typing import Any

from parser import try_parse_json
from utils.network_utils import answer_equivalence

from .slm_agent import SLM_4b_Agent


class Stage1ResultSelector:
    def __init__(self, helper, judge_model_name: str = "gpt-oss:20b"):
        self.helper = helper
        self.judge_model_name = judge_model_name

    def select_stage1_result_with_judge(
        self,
        nodes,
        question: str | None = None,
        fallback_answer: str | None = None,
    ) -> str | None:
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
            raw = judge_agent.invoke(
                [
                    {"role": "system", "content": "You are a strict JSON-only stage-1 selector judge."},
                    {"role": "user", "content": prompt},
                ]
            )
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
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()
