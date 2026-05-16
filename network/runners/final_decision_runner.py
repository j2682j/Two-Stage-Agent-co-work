from __future__ import annotations

import re
from typing import Any

from prompt.contracts import resolve_prompt_contract

from ..core.task_context import TaskContext


class FinalDecisionRunner:
    """FinalDecisionRunner 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    def finalize(self, network: Any, stage2_traces: list[dict[str, Any]], helper: Any | None = None):
        runtime = getattr(network, "runtime", None)
        task_context = (
            runtime.get_task_context()
            if runtime is not None and hasattr(runtime, "get_task_context")
            else None
        )
        prompt_contract = resolve_prompt_contract(
            task_context,
            question=network.current_question or "",
        )
        memory_context = self._build_final_decision_memory_context(network)
        decision = self._decide(
            network,
            stage2_traces=stage2_traces,
            runtime=runtime,
            memory_context=memory_context,
            prompt_contract=prompt_contract,
            task_context=task_context,
        )

        network.last_final_decision = decision
        self._record_final_decision_usage(network, decision)
        self._write_shared_memory_records(
            network,
            helper=helper,
            stage2_traces=stage2_traces,
            decision=decision,
        )

        if not decision.get("success"):
            return None
        return decision.get("final_result")

    def _decide(
        self,
        network: Any,
        *,
        stage2_traces: list[dict[str, Any]],
        runtime: Any,
        memory_context: str,
        prompt_contract: Any,
        task_context: Any,
    ) -> dict[str, Any]:
        if runtime is not None and hasattr(runtime, "measure"):
            with runtime.measure(
                "final_decision_llm",
                stage="final_decision",
                category="llm_call",
                event_type="llm_call",
                metadata={
                    "agent_id": "final_decision_maker",
                    "model_name": getattr(network.final_decision_maker, "fallback_model_name", "unknown"),
                },
                input_summary=str(network.current_question or "")[:240],
            ) as latency:
                decision = self._call_decision_maker(
                    network,
                    stage2_traces=stage2_traces,
                    memory_context=memory_context,
                    prompt_contract=prompt_contract,
                    task_context=task_context,
                )
                prompt_tokens = int(decision.get("prompt_tokens", 0) or 0)
                completion_tokens = int(decision.get("completion_tokens", 0) or 0)
                latency.metadata["token_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                latency.metadata["output_summary"] = str(decision.get("final_result", "") or "")[:240]
                return decision

        return self._call_decision_maker(
            network,
            stage2_traces=stage2_traces,
            memory_context=memory_context,
            prompt_contract=prompt_contract,
            task_context=task_context,
        )

    def _call_decision_maker(
        self,
        network: Any,
        *,
        stage2_traces: list[dict[str, Any]],
        memory_context: str,
        prompt_contract: Any,
        task_context: Any,
    ) -> dict[str, Any]:
        return network.final_decision_maker.decide(
            question=network.current_question or "",
            stage1_result=network.last_stage1_result,
            top_k_outputs=stage2_traces,
            top_k_indices=network.last_top_k_indices,
            importance_scores=network.last_importance,
            memory_context=memory_context,
            prompt_contract=prompt_contract,
            task_context=task_context,
        )

    def _record_final_decision_usage(self, network: Any, decision: dict[str, Any]) -> None:
        runtime = getattr(network, "runtime", None)
        if runtime is None:
            return
        runtime.record_token_usage(
            {
                "stage": "final_decision",
                "agent_id": "final_decision_maker",
                "model_name": getattr(network.final_decision_maker, "fallback_model_name", "unknown"),
                "prompt_tokens": decision.get("prompt_tokens", 0),
                "completion_tokens": decision.get("completion_tokens", 0),
                "mode": decision.get("mode", ""),
            }
        )

    def _build_final_decision_memory_context(self, network: Any) -> str:
        runtime = getattr(network, "runtime", None)
        memory_service = getattr(runtime, "memory_service", None)
        question = self._normalize_text(getattr(network, "current_question", "") or "")
        if runtime is None or memory_service is None or not question:
            return ""

        try:
            task_context = (
                runtime.get_task_context()
                if hasattr(runtime, "get_task_context")
                else TaskContext.from_dict(getattr(runtime, "current_context", {}) or {})
            )
            source = task_context.source_label
            task_id = task_context.task_id or None
            result = memory_service.retrieve_context(
                question=question,
                stage="final_decision",
                injection_target="final_decision",
                source=f"{source}_final_decision",
                task_id=task_id,
                limit=3,
            )
            if result is None:
                return ""
            return str(result.get("guidance", "") or "")
        except Exception as exc:
            print(f"[WARN] final decision graph memory guidance failed: {exc}")
            return ""

    def _write_shared_memory_records(
        self,
        network: Any,
        *,
        helper: Any | None,
        stage2_traces: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> None:
        if helper is not None and hasattr(helper, "_write_shared_memory_records"):
            helper._write_shared_memory_records(
                network,
                stage2_traces=stage2_traces,
                decision=decision,
            )

    def _normalize_text(self, text: Any) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()
