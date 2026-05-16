from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from parser.bfcl_tool_call_parser import BFCLToolCallParser
from utils.network_utils import (
    extract_choice_answer,
    extract_math_answer,
    normalize_for_exact,
    normalize_text,
)


@dataclass
class EarlyStopDecision:
    should_stop: bool
    reason: str = ""
    benchmark: str = ""
    round_id: int = 0
    representative: str | None = None
    cluster_size: int = 0
    active_answer_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStopConfig:
    enabled: bool = True
    gaia_min_rounds: int = 1
    bfcl_min_rounds: int = 1
    gaia_min_cluster_size: int = 2
    gaia_min_cluster_ratio: float = 0.5
    gaia_embedding_threshold: float = 0.90
    gaia_lexical_threshold: float = 0.92
    gaia_short_answer_max_tokens: int = 4
    bfcl_min_valid_votes: int = 2


class EarlyStopPolicy:
    def __init__(self, config: EarlyStopConfig | None = None):
        self.config = config or EarlyStopConfig()
        self._bfcl_parser = BFCLToolCallParser()
        self._embedder: Any | None = None
        self._embedder_failed = False

    def check(self, *, network: Any, round_id: int, active_indices: list[int], context: dict[str, Any]) -> EarlyStopDecision:
        benchmark = str((context or {}).get("benchmark", "")).upper()
        if not self.config.enabled:
            return EarlyStopDecision(False, reason="disabled", benchmark=benchmark, round_id=round_id)

        if benchmark == "BFCL":
            if round_id < self.config.bfcl_min_rounds:
                return EarlyStopDecision(False, reason="below_min_rounds", benchmark=benchmark, round_id=round_id)
            return self._check_bfcl(network=network, round_id=round_id, active_indices=active_indices, context=context)

        if benchmark == "GAIA":
            if round_id < self.config.gaia_min_rounds:
                return EarlyStopDecision(False, reason="below_min_rounds", benchmark=benchmark, round_id=round_id)
            return self._check_gaia(network=network, round_id=round_id, active_indices=active_indices)

        return EarlyStopDecision(False, reason="unsupported_benchmark", benchmark=benchmark, round_id=round_id)

    def _check_gaia(self, *, network: Any, round_id: int, active_indices: list[int]) -> EarlyStopDecision:
        answers = self._active_answers(network, active_indices)
        if len(answers) < self.config.gaia_min_cluster_size:
            return EarlyStopDecision(
                False,
                reason="not_enough_answers",
                benchmark="GAIA",
                round_id=round_id,
                active_answer_count=len(answers),
            )

        clusters = self._cluster_gaia_answers(answers)
        best = max(clusters, key=lambda item: len(item["members"]), default=None)
        if not best:
            return EarlyStopDecision(False, reason="no_clusters", benchmark="GAIA", round_id=round_id)

        cluster_size = len(best["members"])
        required = max(
            self.config.gaia_min_cluster_size,
            math.ceil(len(answers) * self.config.gaia_min_cluster_ratio),
        )
        should_stop = cluster_size >= required
        return EarlyStopDecision(
            should_stop,
            reason="semantic_consensus" if should_stop else "no_semantic_consensus",
            benchmark="GAIA",
            round_id=round_id,
            representative=best["representative"],
            cluster_size=cluster_size,
            active_answer_count=len(answers),
            details={
                "required_votes": required,
                "clusters": [
                    {
                        "representative": cluster["representative"],
                        "size": len(cluster["members"]),
                        "indices": cluster["indices"],
                        "merge_methods": cluster.get("merge_methods", []),
                    }
                    for cluster in clusters
                ],
            },
        )

    def _check_bfcl(
        self,
        *,
        network: Any,
        round_id: int,
        active_indices: list[int],
        context: dict[str, Any],
    ) -> EarlyStopDecision:
        answers = self._active_answers(network, active_indices)
        functions = (context or {}).get("functions", []) or []
        schema = self._bfcl_schema(functions)
        valid_items: list[dict[str, Any]] = []
        clusters: dict[str, list[dict[str, Any]]] = {}

        for item in answers:
            metadata = self._bfcl_parser.parse_with_metadata(item["answer"])
            calls = metadata.get("calls", [])
            if not calls or not self._bfcl_calls_schema_valid(calls, schema):
                continue
            canonical = json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            valid = {**item, "calls": calls, "canonical": canonical}
            valid_items.append(valid)
            clusters.setdefault(canonical, []).append(valid)

        if len(valid_items) < self.config.bfcl_min_valid_votes:
            return EarlyStopDecision(
                False,
                reason="not_enough_valid_bfcl_calls",
                benchmark="BFCL",
                round_id=round_id,
                active_answer_count=len(answers),
                details={"valid_call_count": len(valid_items)},
            )

        best_key, best_members = max(clusters.items(), key=lambda pair: len(pair[1]))
        should_stop = len(best_members) >= self.config.bfcl_min_valid_votes
        return EarlyStopDecision(
            should_stop,
            reason="bfcl_structural_consensus" if should_stop else "no_bfcl_structural_consensus",
            benchmark="BFCL",
            round_id=round_id,
            representative=best_key,
            cluster_size=len(best_members),
            active_answer_count=len(answers),
            details={
                "valid_call_count": len(valid_items),
                "clusters": [
                    {"canonical": key, "size": len(members), "indices": [m["idx"] for m in members]}
                    for key, members in clusters.items()
                ],
            },
        )

    def _active_answers(self, network: Any, active_indices: list[int]) -> list[dict[str, Any]]:
        answers: list[dict[str, Any]] = []
        for idx in active_indices:
            node = network.nodes[idx]
            answer = str(node.get_answer() or "").strip()
            if answer:
                answers.append({"idx": idx, "answer": answer})
        return answers

    def _cluster_gaia_answers(self, answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for item in answers:
            placed = False
            for cluster in clusters:
                method = self._gaia_equivalence_method(item["answer"], cluster["representative"])
                if method:
                    cluster["members"].append(item["answer"])
                    cluster["indices"].append(item["idx"])
                    cluster.setdefault("merge_methods", []).append(method)
                    placed = True
                    break
            if not placed:
                clusters.append(
                    {
                        "representative": item["answer"],
                        "members": [item["answer"]],
                        "indices": [item["idx"]],
                        "merge_methods": [],
                    }
                )
        return clusters

    def _gaia_equivalence_method(self, left: str, right: str) -> str | None:
        left_key = self._cheap_answer_key(left)
        right_key = self._cheap_answer_key(right)
        if left_key and left_key == right_key:
            return "cheap_key"

        lexical_similarity = SequenceMatcher(None, normalize_for_exact(left), normalize_for_exact(right)).ratio()
        if lexical_similarity >= self.config.gaia_lexical_threshold:
            return "lexical"

        if self._is_short_answer(left) or self._is_short_answer(right):
            return None

        embedding_similarity = self._embedding_similarity(left, right)
        if embedding_similarity is not None and embedding_similarity >= self.config.gaia_embedding_threshold:
            return "embedding"

        return None

    def _cheap_answer_key(self, answer: str) -> str:
        math_answer = extract_math_answer(answer)
        if math_answer is not None:
            return f"math:{math_answer}"
        choice_answer = extract_choice_answer(answer)
        if choice_answer is not None:
            return f"choice:{choice_answer}"
        normalized = normalize_for_exact(answer)
        return f"exact:{normalized}" if normalized else ""

    def _is_short_answer(self, answer: str) -> bool:
        tokens = re.findall(r"\w+", normalize_text(answer))
        return len(tokens) <= self.config.gaia_short_answer_max_tokens

    def _embedding_similarity(self, left: str, right: str) -> float | None:
        if self._embedder_failed:
            return None
        try:
            if self._embedder is None:
                from memory.embedding import get_text_embedder

                self._embedder = get_text_embedder()
            left_vec = self._to_float_vector(self._embedder.encode(left))
            right_vec = self._to_float_vector(self._embedder.encode(right))
            return self._cosine(left_vec, right_vec)
        except Exception:
            self._embedder_failed = True
            return None

    def _to_float_vector(self, value: Any) -> list[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], (list, tuple)):
            value = value[0]
        return [float(item) for item in value]

    def _cosine(self, left: list[float], right: list[float]) -> float | None:
        if not left or not right or len(left) != len(right):
            return None
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return None
        return dot / (left_norm * right_norm)

    def _bfcl_schema(self, functions: list[dict[str, Any]]) -> dict[str, set[str]]:
        schema: dict[str, set[str]] = {}
        for function in functions:
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            parameters = function.get("parameters") or function.get("parameter") or {}
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    parameters = {}
            required = parameters.get("required", []) if isinstance(parameters, dict) else []
            schema[name] = {str(item) for item in required or []}
        return schema

    def _bfcl_calls_schema_valid(self, calls: list[dict[str, Any]], schema: dict[str, set[str]]) -> bool:
        if not schema:
            return bool(calls)
        for call in calls:
            name = str(call.get("name") or "").strip()
            if name not in schema:
                return False
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                return False
            if not schema[name].issubset(set(arguments.keys())):
                return False
        return True
