from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass
class Stage1RunResult:
    majority_answer: str | None
    response_count: int
    completions: list[list[str | None]]
    prompt_tokens: int
    completion_tokens: int


class Stage1RoundRunner:
    """Stage1RoundRunner 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    def run(self, network: Any, question: str) -> Stage1RunResult:
        response_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        network.network_helper.set_allnodes_deactivated(network)
        network.last_stage1_activation_trace = []
        if hasattr(network, "early_stop_checker"):
            network.early_stop_checker.reset(network)
        else:
            network.last_early_stop_decision = None
            network.last_early_stop_trace = []
        assert network.rounds > 2

        activated_indices, prompt_tokens, completion_tokens = self._run_full_round(
            network,
            question,
            round_id=0,
        )
        response_count += len(activated_indices)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        network.last_stage1_activation_trace.append(
            self._build_activation_trace_entry(network, 0, activated_indices)
        )

        activated_indices, prompt_tokens, completion_tokens = self._run_full_round(
            network,
            question,
            round_id=1,
        )
        response_count += len(activated_indices)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        network.last_stage1_activation_trace.append(
            self._build_activation_trace_entry(network, 1, activated_indices)
        )
        stage1_stopped = self._check_early_stop(network, 1)

        idx_mask = list(range(network.agents))
        previous_round_indices = list(range(network.agents, network.agents * 2))
        for round_id in range(2, network.rounds):
            if stage1_stopped:
                break
            print("=" * 20)
            print(f"第 {round_id} 輪")
            print(f"前一輪被 activate 的 node: {activated_indices}\n")

            if network.agents > 3:
                rank_result = self._rank_previous_round(network, question, previous_round_indices, round_id)
                idx_mask = rank_result["idx_mask"]
                response_count += network.activation_cost
                total_prompt_tokens += rank_result["prompt_tokens"]
                total_completion_tokens += rank_result["completion_tokens"]

            activated_indices, prompt_tokens, completion_tokens = self._run_masked_round(
                network,
                question,
                round_id=round_id,
                idx_mask=idx_mask,
            )
            response_count += len(activated_indices)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            previous_round_indices = activated_indices
            network.last_stage1_activation_trace.append(
                self._build_activation_trace_entry(network, round_id, activated_indices)
            )
            if self._check_early_stop(network, round_id):
                break

        completions = self._get_completions(network)
        majority_answer = self._select_majority_answer(network)
        return Stage1RunResult(
            majority_answer=majority_answer,
            response_count=response_count,
            completions=completions,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )

    def _check_early_stop(self, network: Any, round_id: int) -> bool:
        if hasattr(network, "early_stop_checker"):
            return network.early_stop_checker.check_stage1_round(network, round_id)
        return network._check_stage1_early_stop(round_id)

    def _run_full_round(
        self,
        network: Any,
        question: str,
        *,
        round_id: int,
    ) -> tuple[list[int], int, int]:
        loop_indices = list(range(network.agents * round_id, network.agents * (round_id + 1)))
        network.rng.shuffle(loop_indices)
        activated_indices: list[int] = []
        for idx, node_idx in enumerate(loop_indices):
            if round_id > 0:
                print("=" * 20)
            print(f"第 {round_id} 輪，第{idx + 1}個Node 開始回覆")
            activated_indices.append(node_idx)

        prompt_tokens, completion_tokens = self._activate_nodes_parallel(
            network,
            question,
            activated_indices,
        )
        return activated_indices, prompt_tokens, completion_tokens

    def _run_masked_round(
        self,
        network: Any,
        question: str,
        *,
        round_id: int,
        idx_mask: list[int],
    ) -> tuple[list[int], int, int]:
        loop_indices = list(range(network.agents * round_id, network.agents * (round_id + 1)))
        network.rng.shuffle(loop_indices)
        activated_indices: list[int] = []
        for idx, node_idx in enumerate(loop_indices):
            if idx in idx_mask:
                print(round_id, idx)
                activated_indices.append(node_idx)

        prompt_tokens, completion_tokens = self._activate_nodes_parallel(
            network,
            question,
            activated_indices,
        )
        return activated_indices, prompt_tokens, completion_tokens

    def _activate_nodes_parallel(
        self,
        network: Any,
        question: str,
        node_indices: list[int],
    ) -> tuple[int, int]:
        if not node_indices:
            return 0, 0

        workers = self._stage1_worker_count(network, len(node_indices))
        if workers <= 1 or len(node_indices) <= 1:
            for node_idx in node_indices:
                network.nodes[node_idx].activate(question)
            return self._sum_token_usage(network, node_indices)

        first_error: BaseException | None = None
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage1-node") as executor:
            futures = [
                (node_idx, executor.submit(network.nodes[node_idx].activate, question))
                for node_idx in node_indices
            ]
            for node_idx, future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    print(f"[WARN] Stage1 node activation failed - idx={node_idx}: {exc}")

        if first_error is not None:
            raise first_error

        return self._sum_token_usage(network, node_indices)

    def _stage1_worker_count(self, network: Any, task_count: int) -> int:
        configured = getattr(network, "stage1_parallel_workers", None)
        if configured is None:
            configured = os.getenv("STAGE1_PARALLEL_WORKERS")

        try:
            worker_count = int(configured) if configured not in (None, "") else task_count
        except (TypeError, ValueError):
            worker_count = task_count

        if worker_count <= 0:
            worker_count = task_count
        return max(1, min(task_count, worker_count))

    def _sum_token_usage(self, network: Any, node_indices: list[int]) -> tuple[int, int]:
        prompt_tokens = 0
        completion_tokens = 0
        for node_idx in node_indices:
            prompt_tokens += int(getattr(network.nodes[node_idx], "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(network.nodes[node_idx], "completion_tokens", 0) or 0)
        return prompt_tokens, completion_tokens

    def _rank_previous_round(
        self,
        network: Any,
        question: str,
        previous_round_indices: list[int],
        round_id: int,
    ) -> dict[str, Any]:
        replies = [network.nodes[idx].get_reply() for idx in previous_round_indices]
        indices = list(range(len(replies)))
        network.rng.shuffle(indices)
        shuffled_replies = [replies[idx] for idx in indices]

        if network.runtime is not None and hasattr(network.runtime, "measure"):
            with network.runtime.measure(
                "stage1_activation_ranker_llm",
                stage="stage1_activation_ranker",
                category="llm_call",
                event_type="llm_call",
                metadata={"agent_id": "activation_ranker", "model_name": "gpt-oss:20b", "round": round_id},
                input_summary=str(question or "")[:240],
            ) as latency:
                tops, prompt_tokens, completion_tokens = network.activation(
                    shuffled_replies,
                    question,
                    "gpt-oss:20b",
                )
                latency.metadata["token_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                latency.metadata["output_summary"] = str(tops)
        else:
            tops, prompt_tokens, completion_tokens = network.activation(
                shuffled_replies,
                question,
                "gpt-oss:20b",
            )

        if network.runtime is not None:
            network.runtime.record_token_usage(
                {
                    "stage": "stage1_activation_ranker",
                    "agent_id": "activation_ranker",
                    "model_name": "gpt-oss:20b",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "round": round_id,
                }
            )

        idx_mask = list(map(lambda x: previous_round_indices[indices[x]] % network.agents, tops))
        return {
            "idx_mask": idx_mask,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _build_activation_trace_entry(self, network: Any, round_id: int, node_indices: list[int]) -> dict[str, Any]:
        return {
            "round": round_id,
            "node_indices": list(node_indices),
            "nodes": [
                {
                    "idx": idx,
                    "model_name": getattr(network.nodes[idx], "model_name", None),
                    "active": bool(getattr(network.nodes[idx], "active", False)),
                    "answer": network.nodes[idx].get_answer(),
                    "reply": network.nodes[idx].get_reply(),
                    "stage1_reflection_context": getattr(network.nodes[idx], "stage1_reflection_context", ""),
                    "stage1_attachment_context": getattr(network.nodes[idx], "stage1_attachment_context", ""),
                }
                for idx in node_indices
            ],
        }

    def _get_completions(self, network: Any) -> list[list[str | None]]:
        completions = [[] for _ in range(network.agents)]
        for round_id in range(network.rounds):
            for idx in range(network.agents * round_id, network.agents * (round_id + 1)):
                if network.nodes[idx].active:
                    completions[idx % network.agents].append(network.nodes[idx].get_reply())
                else:
                    completions[idx % network.agents].append(None)
        return completions

    def _select_majority_answer(self, network: Any) -> str | None:
        active_answer_indices = [
            idx for idx, node in enumerate(network.nodes)
            if getattr(node, "active", False) and str(node.get_answer() or "").strip()
        ]
        if network.last_early_stop_decision and network.last_early_stop_decision.get("representative"):
            return network.last_early_stop_decision.get("representative")

        answers = [network.nodes[idx].get_answer() for idx in active_answer_indices]
        clusters = network.network_helper.cluster_answers(answers)
        return network.network_helper.select_cluster_representative(clusters)
