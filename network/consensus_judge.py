from __future__ import annotations

import json
import math
import re
from typing import Any

from parser import try_parse_json
from utils.network_utils import answer_equivalence

from .slm_agent import SLM_4b_Agent


class ConsensusJudge:
    def __init__(self, judge_model_name: str = "gpt-oss:20b"):
        self.judge_model_name = judge_model_name

    def collect_consensus_candidates(self, nodes, idxs: list[int]) -> list[dict[str, Any]]:
        candidates = []
        for idx in idxs:
            node = nodes[idx]
            if not node.active:
                continue

            reply = node.get_reply()
            if not reply:
                continue

            candidates.append(
                {
                    "node_idx": idx,
                    "reply": reply,
                    "final_claim": self.extract_final_claim(reply),
                }
            )

        return candidates

    def extract_final_claim(self, reply: str) -> str:
        if not reply:
            return ""

        text = reply.strip()

        parsed = try_parse_json(text)
        if isinstance(parsed, dict):
            if "final_answer" in parsed:
                return str(parsed["final_answer"]).strip()
            if "answer" in parsed:
                return str(parsed["answer"]).strip()

        json_block_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_block_match:
            parsed = try_parse_json(json_block_match.group(0))
            if isinstance(parsed, dict):
                if "final_answer" in parsed:
                    return str(parsed["final_answer"]).strip()
                if "answer" in parsed:
                    return str(parsed["answer"]).strip()

        patterns = [
            r"final answer\s*[:=]\s*([^\n\r]+)",
            r"answer\s*[:=]\s*([^\n\r]+)",
            r"the answer is\s+([^\n\r]+)",
            r"therefore[, ]+(?:the answer is|the result is)?\s*([^\n\r]+)",
            r"so[, ]+(?:the answer is|the result is)?\s*([^\n\r]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                claim = match.group(1).strip()
                claim = claim.splitlines()[0].strip()
                return claim.strip(" .")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        return lines[-1]

    def judge_consensus(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            return {
                "has_consensus": False,
                "consensus_cluster": None,
                "consensus_answer": None,
                "member_indices": [],
                "clusters": [],
            }

        judge_input = [
            {
                "node_idx": item["node_idx"],
                "final_claim": item["final_claim"],
            }
            for item in candidates
        ]

        judge_prompt = f"""
You are a consensus judge.

Group answers that express the same final conclusion, even if wording differs.
Ignore style, explanation length, and reasoning details.
Focus only on whether the final claims mean the same thing.

Return valid JSON only with this schema:
{{
  "clusters": [
    {{
      "canonical_answer": "string",
      "member_indices": [0, 2]
    }}
  ]
}}

Answers:
{json.dumps(judge_input, ensure_ascii=False, indent=2)}
        """.strip()

        consensus_judge_agent = SLM_4b_Agent(model_name=self.judge_model_name)
        response = consensus_judge_agent.invoke(
            [
                {"role": "system", "content": "You are a strict JSON-only consensus judge."},
                {"role": "user", "content": judge_prompt},
            ]
        )

        parsed = try_parse_json(response)
        if parsed is None:
            return {
                "has_consensus": False,
                "consensus_cluster": None,
                "consensus_answer": None,
                "member_indices": [],
                "clusters": [],
                "raw_response": response,
            }

        clusters = parsed.get("clusters", [])
        if not clusters:
            return {
                "has_consensus": False,
                "consensus_cluster": None,
                "consensus_answer": None,
                "member_indices": [],
                "clusters": [],
            }

        valid_positions = set(range(len(candidates)))
        sanitized_clusters = []

        for cluster in clusters:
            raw_indices = cluster.get("member_indices", [])
            member_indices = [
                idx for idx in raw_indices
                if isinstance(idx, int) and idx in valid_positions
            ]
            member_indices = sorted(set(member_indices))
            if not member_indices:
                continue

            canonical_answer = cluster.get("canonical_answer")
            verified_indices = []
            for idx in member_indices:
                claim = candidates[idx]["final_claim"]
                if canonical_answer and claim and answer_equivalence(claim, canonical_answer):
                    verified_indices.append(idx)

            if not verified_indices and member_indices:
                verified_indices = [member_indices[0]]
                canonical_answer = candidates[member_indices[0]]["final_claim"]

            sanitized_clusters.append(
                {
                    "canonical_answer": canonical_answer,
                    "member_indices": verified_indices,
                }
            )

        if not sanitized_clusters:
            return {
                "has_consensus": False,
                "consensus_cluster": None,
                "consensus_answer": None,
                "member_indices": [],
                "clusters": [],
            }

        largest_cluster = max(sanitized_clusters, key=lambda c: len(c["member_indices"]))
        member_indices = largest_cluster["member_indices"]
        consensus_answer = largest_cluster.get("canonical_answer")

        return {
            "has_consensus": len(member_indices) >= 2,
            "consensus_answer": consensus_answer,
            "member_indices": member_indices,
            "clusters": sanitized_clusters,
        }

    def check_consensus(self, nodes, idxs: list[int], idx_mask: list[int]) -> tuple[bool, str | None]:
        print("開始檢查是否達成共識")
        candidates = self.collect_consensus_candidates(nodes, idxs)
        judge_result = self.judge_consensus(candidates)

        member_indices = judge_result.get("member_indices", [])
        consensus_answer = judge_result.get("consensus_answer")
        required_votes = max(2, math.ceil((2 / 3) * len(idx_mask)))
        has_consensus = len(member_indices) >= required_votes

        print("has_consensus:", has_consensus)
        print("consensus_answer:", consensus_answer)

        if has_consensus:
            return True, consensus_answer

        return False, None
