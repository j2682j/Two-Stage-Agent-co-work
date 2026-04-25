from __future__ import annotations

import re
from typing import Any, Optional

from memory import MemoryConfig
from memory.policy import (
    build_memory_records,
    should_write_final_memory,
    should_write_stage2_memory,
)
from utils.network_utils import answer_equivalence


class AgentNetworkHelper:
    """
    AgentNetworkHelper 提供了 AgentNetwork 相關的工具方法，包括記憶體配置複製、運行時初始化、神經網絡構建、答案聚類與選擇，以及共享記憶體寫入等功能。
    1. init_nn: 根據抽取的模型名稱構建神經網絡，設置節點和邊，並定義激活函數和成本。
    2. init_runtime: 初始化 NetworkRuntime，設置工具管理器、記憶體工具和相關配置。
    
    """
    def zero_grad(self, network) -> None:
        for edge in network.edges:
            edge.zero_weight()

    def set_allnodes_deactivated(self, network) -> None:
        for node in network.nodes:
            node.deactivate()
        
    
    def clone_memory_config(self, memory_config: Optional[MemoryConfig]) -> MemoryConfig:
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
        return network.tool_manager

    def sample_model_name_for_round(self, network, n: int) -> list[str]:
        if n > len(network.model_pool):
            raise ValueError("Not enough models for one round")

        models = list(network.model_pool)
        network.rng.shuffle(models)
        return models[:n]


    def cluster_answers(self, answers: list[str]) -> list[dict]:
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
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    def select_top_k_agents(self, network, top_k: int, importance: list[float]) -> list[int]:
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
        memory_context = ""
        runtime = getattr(network, "runtime", None)
        memory_mode = getattr(runtime, "memory_mode", "disabled")
        if memory_mode == "final_decision":
            memory_context = runtime.build_memory_context_for_final_decision(
                getattr(network, "current_question", "") or ""
            )
        decision = network.final_decision_maker.decide(
            question=network.current_question or "",
            stage1_result=network.last_stage1_result,
            top_k_outputs=stage2_traces,
            top_k_indices=network.last_top_k_indices,
            importance_scores=network.last_importance,
            memory_context=memory_context,
        )

        network.last_final_decision = decision
        self._write_shared_memory_records(
            network,
            stage2_traces=stage2_traces,
            decision=decision,
        )

        if not decision.get("success"):
            return None

        return decision.get("final_result")

    def _write_shared_memory_records(
        self,
        network,
        *,
        stage2_traces: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> None:
        memory_tool = getattr(network.runtime, "memory_tool", None) or getattr(network, "memory_tool", None)
        memory_manager = getattr(memory_tool, "memory_manager", None)
        if memory_manager is None:
            return

        records: list[dict[str, Any]] = []
        for candidate in self._extract_judged_stage2_candidates(stage2_traces, decision):
            if should_write_stage2_memory(candidate):
                records.extend(
                    build_memory_records(
                        question=network.current_question or "",
                        source_stage="stage2",
                        payload=candidate,
                    )
                )

        if should_write_final_memory(decision):
            records.extend(
                build_memory_records(
                    question=network.current_question or "",
                    source_stage="final",
                    payload=decision,
                )
            )

        for record in network.runtime.dedupe_memory_records(records):
            memory_type = str(record.get("memory_type", "") or "").strip()
            if memory_type not in getattr(memory_manager, "memory_types", {}):
                continue
            try:
                memory_id = memory_manager.add_memory(
                    content=record["content"],
                    memory_type=memory_type,
                    importance=record.get("importance"),
                    metadata=record.get("metadata"),
                    auto_classify=bool(record.get("auto_classify", False)),
                )
            except Exception as e:
                print(f"[WARN] shared memory 寫入失敗: {e}")
                continue

            if network.runtime is not None:
                network.runtime.record_memory_write(
                    {
                        "memory_id": memory_id,
                        "memory_type": memory_type,
                        "source_stage": record.get("metadata", {}).get("source_stage"),
                        "answer": record.get("metadata", {}).get("answer"),
                    }
                )

    def _extract_judged_stage2_candidates(
        self,
        stage2_traces: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
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
