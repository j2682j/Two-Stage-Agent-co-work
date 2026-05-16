from __future__ import annotations

import re
from typing import Any, Optional

from memory.base import MemoryConfig
from ..runners.final_decision_runner import FinalDecisionRunner
from prompt.contracts import resolve_prompt_contract
from utils.network_utils import answer_equivalence
from .task_context import TaskContext


class AgentNetworkHelper:
    """AgentNetworkHelper 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """
    def zero_grad(self, network) -> None:
        """處理 zero_grad 流程並回傳結果。
        
        參數:
            network: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        for edge in network.edges:
            edge.zero_weight()

    def set_allnodes_deactivated(self, network) -> None:
        """設定 set_allnodes_deactivated 對應的資料。
        
        參數:
            network: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        for node in network.nodes:
            node.deactivate()
        
    
    def clone_memory_config(self, memory_config: Optional[MemoryConfig]) -> MemoryConfig:
        """處理 clone_memory_config 流程並回傳結果。
        
        參數:
            memory_config: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        if memory_config is None:
            return MemoryConfig()

        if hasattr(memory_config, "model_copy"):
            return memory_config.model_copy(deep=True)
        if hasattr(memory_config, "copy"):
            try:
                return memory_config.copy(deep=True)
            except TypeError:
                return memory_config.copy()
        return memory_config


    def ensure_tool_manager(self, network):
        """處理 ensure_tool_manager 流程並回傳結果。
        
        參數:
            network: 此流程需要使用的輸入資料。
        """
        return network.tool_manager

    def sample_model_name_for_round(self, network, n: int) -> list[str]:
        """處理 sample_model_name_for_round 流程並回傳結果。
        
        參數:
            network: 此流程需要使用的輸入資料。
            n: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        if n > len(network.model_pool):
            raise ValueError("Not enough models for one round")

        models = list(network.model_pool)
        network.rng.shuffle(models)
        return models[:n]


    def cluster_answers(self, answers: list[str]) -> list[dict]:
        """處理 cluster_answers 流程並回傳結果。
        
        參數:
            answers: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        clusters = []

        for idx, ans in enumerate(answers):
            placed = False
            for cluster in clusters:
                if answer_equivalence(ans, cluster["canonical_answer"]):
                    cluster["member_indices"].append(idx)
                    cluster["members"].append(ans)
                    placed = True
                    break
            if not placed:
                clusters.append(
                    {
                        "canonical_answer": ans,
                        "member_indices": [idx],
                        "members": [ans],
                    }
                )

        return clusters

    def select_cluster_representative(self, clusters: list[dict]) -> str | None:
        """處理 select_cluster_representative 流程並回傳結果。
        
        參數:
            clusters: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        if not clusters:
            return None

        best_cluster = max(clusters, key=lambda c: len(c["member_indices"]))
        candidates = [m for m in best_cluster["members"] if m is not None and str(m).strip()]
        if not candidates:
            return None

        return min(candidates, key=lambda c: len(str(c).strip()))

    def select_diverse_top_k(
        self,
        nodes,
        active_indices: list[int],
        importance: list[float],
        top_k: int,
    ) -> list[int]:
        """處理 select_diverse_top_k 流程並回傳結果。
        
        參數:
            nodes: 此流程需要使用的輸入資料。
            active_indices: 此流程需要使用的輸入資料。
            importance: 此流程需要使用的輸入資料。
            top_k: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        ranked = sorted(
            active_indices,
            key=lambda idx: importance[idx],
            reverse=True,
        )

        selected = []
        selected_answers = []

        for idx in ranked:
            answer = self._normalize_text(nodes[idx].get_answer())
            if not selected:
                selected.append(idx)
                selected_answers.append(answer)
                if len(selected) >= top_k:
                    return selected
                continue

            is_diverse = True
            if answer:
                for existing in selected_answers:
                    if existing and answer_equivalence(answer, existing):
                        is_diverse = False
                        break

            if is_diverse:
                selected.append(idx)
                selected_answers.append(answer)
                if len(selected) >= top_k:
                    return selected

        for idx in ranked:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= top_k:
                    break

        return selected

    def _normalize_text(self, text) -> str:
        """處理 normalize_text 流程並回傳結果。
        
        參數:
            text: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    def select_top_k_agents(self, network, top_k: int, importance: list[float]) -> list[int]:
        """處理 select_top_k_agents 流程並回傳結果。
        
        參數:
            network: 此流程需要使用的輸入資料。
            top_k: 此流程需要使用的輸入資料。
            importance: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        active_indices = [
            idx for idx, node in enumerate(network.nodes)
            if getattr(node, "active", False)
        ]

        return self.select_diverse_top_k(
            nodes=network.nodes,
            active_indices=active_indices,
            importance=importance,
            top_k=top_k,
        )

    def finalize_stage2_results(self, network, stage2_traces):
        """處理 finalize_stage2_results 流程並回傳結果。
        
        參數:
            network: 此流程需要使用的輸入資料。
            stage2_traces: 此流程需要使用的輸入資料。
        """
        return FinalDecisionRunner().finalize(network, stage2_traces, helper=self)

        runtime = getattr(network, "runtime", None)
        task_context = runtime.get_task_context() if runtime is not None and hasattr(runtime, "get_task_context") else None
        prompt_contract = resolve_prompt_contract(task_context, question=network.current_question or "")
        memory_context = self._build_final_decision_memory_context(network)
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
                decision = network.final_decision_maker.decide(
                    question=network.current_question or "",
                    stage1_result=network.last_stage1_result,
                    top_k_outputs=stage2_traces,
                    top_k_indices=network.last_top_k_indices,
                    importance_scores=network.last_importance,
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
        else:
            decision = network.final_decision_maker.decide(
                question=network.current_question or "",
                stage1_result=network.last_stage1_result,
                top_k_outputs=stage2_traces,
                top_k_indices=network.last_top_k_indices,
                importance_scores=network.last_importance,
                memory_context=memory_context,
                prompt_contract=prompt_contract,
                task_context=task_context,
            )

        network.last_final_decision = decision
        if runtime is not None:
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
        self._write_shared_memory_records(
            network,
            stage2_traces=stage2_traces,
            decision=decision,
        )

        if not decision.get("success"):
            return None

        return decision.get("final_result")

    def _build_final_decision_memory_context(self, network) -> str:
        """建立 build_final_decision_memory_context 所需的資料或輸出。
        
        參數:
            network: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        runtime = getattr(network, "runtime", None)
        memory_service = getattr(runtime, "memory_service", None)
        question = self._normalize_text(getattr(network, "current_question", "") or "")
        if runtime is None or memory_service is None or not question:
            return ""

        try:
            task_context = runtime.get_task_context() if hasattr(runtime, "get_task_context") else TaskContext.from_dict(getattr(runtime, "current_context", {}) or {})
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
            guidance = str(result.get("guidance", "") or "")
            return guidance
        except Exception as exc:
            print(f"[WARN] final decision graph memory guidance failed: {exc}")
            return ""

    def _write_shared_memory_records(
        self,
        network,
        *,
        stage2_traces: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> None:
        """寫入或儲存 write_shared_memory_records 相關資料。
        
        參數:
            network: 此流程需要使用的輸入資料。
            stage2_traces: 此流程需要使用的輸入資料。
            decision: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        return
    def _extract_judged_stage2_candidates(
        self,
        stage2_traces: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """處理 extract_judged_stage2_candidates 流程並回傳結果。
        
        參數:
            stage2_traces: 此流程需要使用的輸入資料。
            decision: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        judged_by_idx: dict[int, dict[str, Any]] = {}
        for step in decision.get("intermediate_steps", []) or []:
            if step.get("step") != "stage2_judge_rerank":
                continue
            for candidate in step.get("candidates", []) or []:
                agent_idx = candidate.get("agent_idx")
                if isinstance(agent_idx, int):
                    judged_by_idx[agent_idx] = candidate
            break

        merged: list[dict[str, Any]] = []
        for trace in stage2_traces or []:
            item = dict(trace)
            judged = judged_by_idx.get(item.get("agent_idx"), {})
            item["success"] = bool(item.get("success", item.get("stage2_success", False)))
            item["stage2_judge_is_acceptable"] = bool(
                judged.get("is_acceptable", item.get("stage2_judge_is_acceptable", False))
            )
            item["stage2_judge_score"] = judged.get(
                "judge_score",
                item.get("stage2_judge_score", 0.0),
            )
            item["stage2_judge_revised_answer"] = judged.get(
                "revised_answer",
                item.get("stage2_judge_revised_answer", ""),
            )
            merged.append(item)
        return merged
