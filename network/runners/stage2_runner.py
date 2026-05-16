from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from builder.evidence_builder import EvidenceBuilder
from network.core.stage2_evidence_bundle import Stage2EvidenceBundle


class Stage2Runner:
    """Stage2Runner 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    def run(
        self,
        network: Any,
        *,
        question: str,
        top_k_indices: list[int],
        stage1_result: str | None = None,
        importance: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        tool_manager = network.network_helper.ensure_tool_manager(network)
        stage2_outputs: list[dict[str, Any]] = []
        shared_search_bundle = self._prepare_shared_stage2_state(
            network,
            question=question,
            top_k_indices=top_k_indices,
            stage1_result=stage1_result,
        )

        stage2_outputs = self._run_agents_parallel(
            network,
            question=question,
            top_k_indices=top_k_indices,
            tool_manager=tool_manager,
            stage1_result=stage1_result,
            importance=importance,
            shared_search_bundle=shared_search_bundle,
        )

        return stage2_outputs

    def _run_agents_parallel(
        self,
        network: Any,
        *,
        question: str,
        top_k_indices: list[int],
        tool_manager: Any,
        stage1_result: str | None,
        importance: list[float] | None,
        shared_search_bundle: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not top_k_indices:
            return []

        workers = self._stage2_worker_count(network, len(top_k_indices))
        if workers <= 1 or len(top_k_indices) <= 1:
            return [
                self._run_agent(
                    network,
                    idx=idx,
                    question=question,
                    tool_manager=tool_manager,
                    stage1_result=stage1_result,
                    importance=importance[idx] if importance is not None else None,
                    shared_search_bundle=shared_search_bundle,
                )
                for idx in top_k_indices
            ]

        first_error: BaseException | None = None
        outputs: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage2-agent") as executor:
            futures = [
                (
                    idx,
                    executor.submit(
                        self._run_agent,
                        network,
                        idx=idx,
                        question=question,
                        tool_manager=tool_manager,
                        stage1_result=stage1_result,
                        importance=importance[idx] if importance is not None else None,
                        shared_search_bundle=shared_search_bundle,
                    ),
                )
                for idx in top_k_indices
            ]
            for idx, future in futures:
                try:
                    outputs.append(future.result())
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    print(f"[WARN] Stage2 agent failed - idx={idx}: {exc}")
                    outputs.append(self._build_stage2_error_output(network, idx, str(exc)))

        if first_error is not None:
            print(f"[WARN] Stage2 parallel execution completed with errors: {first_error}")

        return outputs

    def _stage2_worker_count(self, network: Any, task_count: int) -> int:
        configured = getattr(network, "stage2_parallel_workers", None)
        if configured is None:
            configured = os.getenv("STAGE2_PARALLEL_WORKERS")

        try:
            worker_count = int(configured) if configured not in (None, "") else task_count
        except (TypeError, ValueError):
            worker_count = task_count

        if worker_count <= 0:
            worker_count = task_count
        return max(1, min(task_count, worker_count))

    def _prepare_shared_stage2_state(
        self,
        network: Any,
        *,
        question: str,
        top_k_indices: list[int],
        stage1_result: str | None,
    ) -> dict[str, Any] | None:
        runtime = getattr(network, "runtime", None)
        if runtime is None:
            return None

        runtime.clear_stage2_shared_state()
        runtime.current_stage2_stage1_result = stage1_result
        runtime.current_stage2_top_k_answers = [
            str(network.nodes[idx].get_answer() or "").strip()
            for idx in top_k_indices
            if str(network.nodes[idx].get_answer() or "").strip()
        ]
        runtime.current_stage2_judge_scores = [
            float(getattr(network.nodes[idx], "stage1_judge_score", 0.0) or 0.0)
            for idx in top_k_indices
        ]
        shared_search_bundle = runtime.prepare_shared_stage2_search(
            question=question,
            router_model_name=None,
        )
        if shared_search_bundle is not None:
            print(
                "[SHARED-SEARCH] "
                f"enabled={bool(shared_search_bundle.get('enabled'))} "
                f"queries={shared_search_bundle.get('queries', [])}"
            )
        return shared_search_bundle

    def _run_agent(
        self,
        network: Any,
        *,
        idx: int,
        question: str,
        tool_manager: Any,
        stage1_result: str | None,
        importance: float | None,
        shared_search_bundle: dict[str, Any] | None,
    ) -> dict[str, Any]:
        node = network.nodes[idx]
        print("=" * 20)
        print(f"Stage2 Agent Start - idx={idx}, model={getattr(node, 'model_name', None)}")
        print("=" * 20)

        try:
            evidence_bundle = self._build_agent_evidence_bundle(
                network,
                idx=idx,
                question=question,
                tool_manager=tool_manager,
                shared_search_bundle=shared_search_bundle,
            )
            result = node.activate_stage2(
                question=question,
                stage1_result=stage1_result,
                importance=importance,
                evidence_bundle=evidence_bundle,
            )
            output = self._build_stage2_output(network, idx, result)
        except Exception as exc:
            result = {
                "answer": None,
                "reply": None,
                "tool_usage": [],
                "success": False,
                "error": str(exc),
            }
            output = self._build_stage2_error_output(network, idx, str(exc))

        print("=" * 20)
        print(f"Stage2 Agent End - idx={idx}, success={result.get('success', False)}")
        print("=" * 20)
        return output

    def _build_agent_evidence_bundle(
        self,
        network: Any,
        *,
        idx: int,
        question: str,
        tool_manager: Any,
        shared_search_bundle: dict[str, Any] | None,
    ) -> Stage2EvidenceBundle:
        node = network.nodes[idx]
        agent_id = str(getattr(node, "model_name", "unknown_agent") or "unknown_agent")
        runtime = getattr(network, "runtime", None)
        builder = getattr(runtime, "evidence_builder", None)
        if builder is None:
            builder = EvidenceBuilder(
                tool_manager=tool_manager,
                memory_tool=getattr(runtime, "memory_tool", None),
                runtime=runtime,
            )

        try:
            evidence = builder.build(
                question=question,
                agent_id=agent_id,
                stage="stage2",
                router_model_name=agent_id,
                shared_search_bundle=shared_search_bundle,
            )
        except Exception as exc:
            return Stage2EvidenceBundle.failed(agent_id=agent_id, error=str(exc))

        if runtime is not None:
            runtime.record_tool_trace(
                {
                    "agent_id": agent_id,
                    "stage": "stage2",
                    "question": question,
                    "tool_usage": evidence.get("tool_usage", []),
                    "routing": evidence.get("routing", {}),
                }
            )

        return Stage2EvidenceBundle.from_evidence(agent_id=agent_id, evidence=evidence)

    def _build_stage2_output(self, network: Any, idx: int, result: dict[str, Any]) -> dict[str, Any]:
        node = network.nodes[idx]
        return {
            "agent_idx": idx,
            "model_name": getattr(node, "model_name", None),
            "answer": result.get("answer"),
            "reply": result.get("reply"),
            "tool_usage": result.get("tool_usage", []),
            "routing": result.get("routing", {}),
            "attachment_context": getattr(node, "stage2_attachment_context", ""),
            "search_context": getattr(node, "stage2_search_context", ""),
            "solver_context": getattr(node, "stage2_solver_context", ""),
            "memory_context": getattr(node, "stage2_memory_context", ""),
            "rag_context": getattr(node, "stage2_rag_context", ""),
            "success": result.get("success", True),
            "error": result.get("error"),
        }

    def _build_stage2_error_output(self, network: Any, idx: int, error: str) -> dict[str, Any]:
        node = network.nodes[idx]
        return {
            "agent_idx": idx,
            "model_name": getattr(node, "model_name", None),
            "answer": None,
            "reply": None,
            "tool_usage": [],
            "routing": {},
            "attachment_context": getattr(node, "stage2_attachment_context", ""),
            "search_context": getattr(node, "stage2_search_context", ""),
            "solver_context": getattr(node, "stage2_solver_context", ""),
            "memory_context": getattr(node, "stage2_memory_context", ""),
            "rag_context": getattr(node, "stage2_rag_context", ""),
            "success": False,
            "error": error,
        }
