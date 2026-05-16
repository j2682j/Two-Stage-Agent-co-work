from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class BackwardScorer:
    """BackwardScorer 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    def score(self, network: Any, result: str | None) -> list[float]:
        flag_last = False
        for round_id in range(network.rounds - 1, -1, -1):
            if not flag_last:
                active_indices = self._active_indices_for_round(network, round_id)
                if active_indices:
                    flag_last = True
                else:
                    continue

                self._score_last_active_round(network, round_id, active_indices)
            else:
                self._propagate_importance_to_round(network, round_id)

        refined_result = self._select_stage1_result(network, result)
        network.last_stage1_result = refined_result
        return [node.importance for node in network.nodes]

    def _active_indices_for_round(self, network: Any, round_id: int) -> list[int]:
        return [
            idx
            for idx in range(network.agents * round_id, network.agents * (round_id + 1))
            if network.nodes[idx].active
        ]

    def _score_last_active_round(self, network: Any, round_id: int, active_indices: list[int]) -> None:
        scored: dict[int, float] = {}
        total_score = 0.0
        evaluations = self._evaluate_active_nodes_parallel(network, active_indices)
        for idx in active_indices:
            node = network.nodes[idx]
            evaluation = evaluations.get(idx, {})
            raw_score = max(float(evaluation.get("score", 0.0)), 0.0)
            score = network.stage1_judge.adjust_stage1_importance(evaluation)
            node.stage1_judge_is_acceptable = evaluation.get("is_acceptable", False)
            node.stage1_judge_score = raw_score
            node.stage1_judge_adjusted_score = score
            node.stage1_judge_approved_answer = evaluation.get("approved_answer", "")
            node.stage1_judge_suggested_fix = evaluation.get("suggested_fix", "")
            node.stage1_judge_revised_answer = evaluation.get("revised_answer", "")
            node.stage1_judge_reasoning = evaluation.get("judge_reasoning", "")
            node.stage1_judge_used_fallback = evaluation.get("used_fallback", False)
            self._record_stage1_judge_usage(network, idx, evaluation)
            scored[idx] = score
            total_score += score

        if total_score <= 0 and active_indices:
            uniform_score = 1 / len(active_indices)
            for idx in range(network.agents * round_id, network.agents * (round_id + 1)):
                network.nodes[idx].importance = uniform_score if idx in active_indices else 0
            return

        for idx in range(network.agents * round_id, network.agents * (round_id + 1)):
            if idx in scored and total_score > 0:
                network.nodes[idx].importance = scored[idx] / total_score
            else:
                network.nodes[idx].importance = 0

    def _evaluate_active_nodes_parallel(self, network: Any, active_indices: list[int]) -> dict[int, dict[str, Any]]:
        if not active_indices:
            return {}

        workers = self._backward_judge_worker_count(network, len(active_indices))
        if workers <= 1 or len(active_indices) <= 1:
            return {
                idx: self._evaluate_stage1_node(network, idx)
                for idx in active_indices
            }

        evaluations: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="backward-judge") as executor:
            futures = [
                (idx, executor.submit(self._evaluate_stage1_node, network, idx))
                for idx in active_indices
            ]
            for idx, future in futures:
                try:
                    evaluations[idx] = future.result()
                except BaseException as exc:
                    print(f"[WARN] Stage1 judge failed - idx={idx}: {exc}")
                    evaluations[idx] = {
                        "is_acceptable": False,
                        "score": 0.0,
                        "approved_answer": "",
                        "suggested_fix": f"Stage1 judge failed: {type(exc).__name__}: {exc}",
                        "revised_answer": "",
                        "judge_reasoning": "",
                        "raw_response": None,
                        "used_fallback": True,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }
        return evaluations

    def _backward_judge_worker_count(self, network: Any, task_count: int) -> int:
        configured = getattr(network, "backward_judge_workers", None)
        if configured is None:
            configured = os.getenv("BACKWARD_JUDGE_WORKERS")

        try:
            worker_count = int(configured) if configured not in (None, "") else task_count
        except (TypeError, ValueError):
            worker_count = task_count

        if worker_count <= 0:
            worker_count = task_count
        return max(1, min(task_count, worker_count))

    def _evaluate_stage1_node(self, network: Any, node_idx: int) -> dict[str, Any]:
        node = network.nodes[node_idx]
        runtime = getattr(network, "runtime", None)
        if runtime is not None and hasattr(runtime, "measure"):
            with runtime.measure(
                "stage1_judge_llm",
                stage="stage1_judge",
                category="llm_call",
                event_type="llm_call",
                metadata={
                    "agent_id": f"stage1_judge_node_{node_idx}",
                    "model_name": getattr(network.stage1_judge, "judge_model_name", "unknown"),
                    "node_idx": node_idx,
                },
                input_summary=str(network.current_question or "")[:240],
            ) as latency:
                evaluation = network.stage1_judge.evaluate_stage1_candidate(
                    question=network.current_question or "",
                    reasoning=node.reasoning,
                    final_answer=node.get_answer(),
                )
                prompt_tokens = int(evaluation.get("prompt_tokens", 0) or 0)
                completion_tokens = int(evaluation.get("completion_tokens", 0) or 0)
                latency.metadata["token_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                latency.metadata["output_summary"] = str(evaluation.get("judge_reasoning", "") or "")[:240]
                return evaluation

        return network.stage1_judge.evaluate_stage1_candidate(
            question=network.current_question or "",
            reasoning=node.reasoning,
            final_answer=node.get_answer(),
        )

    def _record_stage1_judge_usage(self, network: Any, node_idx: int, evaluation: dict[str, Any]) -> None:
        runtime = getattr(network, "runtime", None)
        if runtime is None:
            return
        runtime.record_token_usage(
            {
                "stage": "stage1_judge",
                "agent_id": f"stage1_judge_node_{node_idx}",
                "model_name": getattr(network.stage1_judge, "judge_model_name", "unknown"),
                "prompt_tokens": evaluation.get("prompt_tokens", 0),
                "completion_tokens": evaluation.get("completion_tokens", 0),
                "node_idx": node_idx,
            }
        )

    def _propagate_importance_to_round(self, network: Any, round_id: int) -> None:
        for idx in range(network.agents * round_id, network.agents * (round_id + 1)):
            network.nodes[idx].importance = 0
            if network.nodes[idx].active:
                for edge in network.nodes[idx].to_edges:
                    network.nodes[idx].importance += edge.weight * edge.a2.importance

    def _select_stage1_result(self, network: Any, result: str | None) -> str | None:
        runtime = getattr(network, "runtime", None)
        if runtime is not None and hasattr(runtime, "measure"):
            with runtime.measure(
                "stage1_result_selector_llm",
                stage="stage1_result_selector",
                category="llm_call",
                event_type="llm_call",
                metadata={
                    "agent_id": "stage1_result_selector",
                    "model_name": getattr(network.stage1_result_selector, "judge_model_name", "unknown"),
                },
                input_summary=str(network.current_question or "")[:240],
            ) as latency:
                refined_result = network.stage1_result_selector.select_stage1_result_with_judge(
                    network.nodes,
                    question=network.current_question,
                    fallback_answer=result,
                )
                prompt_tokens = int(getattr(network.stage1_result_selector, "last_prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(network.stage1_result_selector, "last_completion_tokens", 0) or 0)
                latency.metadata["token_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                latency.metadata["output_summary"] = str(refined_result or "")[:240]
        else:
            refined_result = network.stage1_result_selector.select_stage1_result_with_judge(
                network.nodes,
                question=network.current_question,
                fallback_answer=result,
            )

        if runtime is not None:
            runtime.record_token_usage(
                {
                    "stage": "stage1_result_selector",
                    "agent_id": "stage1_result_selector",
                    "model_name": getattr(network.stage1_result_selector, "judge_model_name", "unknown"),
                    "prompt_tokens": getattr(network.stage1_result_selector, "last_prompt_tokens", 0),
                    "completion_tokens": getattr(network.stage1_result_selector, "last_completion_tokens", 0),
                }
            )
        return refined_result
