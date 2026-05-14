from __future__ import annotations

import re
from typing import Any, Optional

from memory.base import MemoryConfig
from utils.network_utils import answer_equivalence


class AgentNetworkHelper:
    """
    鞎痊??network.agentnetwork_helper 銝剖?鋆?AgentNetworkHelper嚗恣???嗅??遙???炎蝝Ｙ???頝其遙??撽????????
    
    Args:
        ?⊥?蝣箏遣瑽??賂??航?? dataclass 甈???閮剖澆遣蝡隞嗚?
    
    Returns:
        憿?祈澈銝?亙??喳潘?撱箇?撖虫?敺???嗆瘜?雿???瘚???
    
    ??雿:
        ?寞??航?湔?折???撖急?獢?怠??冽????Ｙ??亥?嚗?靘蝙?冽?憓Ⅱ隤?
    """
    def zero_grad(self, network) -> None:
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? zero_grad 瘚?嚗???AgentNetworkHelper ??蝔?瘙???zero_grad 撠?????????雿?蝯??Ｙ???
        
        Args:
            network: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 None??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        for edge in network.edges:
            edge.zero_weight()

    def set_allnodes_deactivated(self, network) -> None:
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? set_allnodes_deactivated 瘚?嚗??啁?頛詨鞈??蔥?啁?隞嗥???瘚?蝝?葉??
        
        Args:
            network: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 None??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        for node in network.nodes:
            node.deactivate()
        
    
    def clone_memory_config(self, memory_config: Optional[MemoryConfig]) -> MemoryConfig:
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? clone_memory_config 瘚?嚗???AgentNetworkHelper ??蝔?瘙???clone_memory_config 撠?????????雿?蝯??Ｙ???
        
        Args:
            memory_config: 閮蝟餌絞???炎蝝Ｙ??神?亥?????隞??
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 MemoryConfig??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
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
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? ensure_tool_manager 瘚?嚗???AgentNetworkHelper ??蝔?瘙???ensure_tool_manager 撠?????????雿?蝯??Ｙ???
        
        Args:
            network: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 ?芣?閮颯?
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        return network.tool_manager

    def sample_model_name_for_round(self, network, n: int) -> list[str]:
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? sample_model_name_for_round 瘚?嚗???AgentNetworkHelper ??蝔?瘙???sample_model_name_for_round 撠?????????雿?蝯??Ｙ???
        
        Args:
            network: ?其??澆璅∪????冽???璅∪??迂?恥?嗥垢??身摰?
            n: ?其??澆璅∪????冽???璅∪??迂?恥?嗥垢??身摰?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 list[str]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if n > len(network.model_pool):
            raise ValueError("Not enough models for one round")

        models = list(network.model_pool)
        network.rng.shuffle(models)
        return models[:n]


    def cluster_answers(self, answers: list[str]) -> list[dict]:
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? cluster_answers 瘚?嚗???AgentNetworkHelper ??蝔?瘙???cluster_answers 撠?????????雿?蝯??Ｙ???
        
        Args:
            answers: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 list[dict]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
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
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? select_cluster_representative 瘚?嚗?遙?敺萸蝑????????蝥?暺極?瑟?瘚????
        
        Args:
            clusters: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str | None??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
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
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? select_diverse_top_k 瘚?嚗?遙?敺萸蝑????????蝥?暺極?瑟?瘚????
        
        Args:
            nodes: ?批瑼Ｙ揣?祟?豢?頛詨?賊???澆??詻?
            active_indices: ?批瑼Ｙ揣?祟?豢?頛詨?賊???澆??詻?
            importance: ?批瑼Ｙ揣?祟?豢?頛詨?賊???澆??詻?
            top_k: ?批瑼Ｙ揣?祟?豢?頛詨?賊???澆??詻?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 list[int]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
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
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? _normalize_text 瘚?嚗???AgentNetworkHelper ??蝔?瘙???_normalize_text 撠?????????雿?蝯??Ｙ???
        
        Args:
            text: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    def select_top_k_agents(self, network, top_k: int, importance: list[float]) -> list[int]:
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? select_top_k_agents 瘚?嚗?遙?敺萸蝑????????蝥?暺極?瑟?瘚????
        
        Args:
            network: ?批瑼Ｙ揣?祟?豢?頛詨?賊???澆??詻?
            top_k: ?批瑼Ｙ揣?祟?豢?頛詨?賊???澆??詻?
            importance: ?批瑼Ｙ揣?祟?豢?頛詨?賊???澆??詻?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 list[int]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
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
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? finalize_stage2_results 瘚?嚗???AgentNetworkHelper ??蝔?瘙???finalize_stage2_results 撠?????????雿?蝯??Ｙ???
        
        Args:
            network: 甇斗?蝔?閬蝙?函?頛詨鞈???
            stage2_traces: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 ?芣?閮颯?
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        runtime = getattr(network, "runtime", None)
        memory_context = self._build_final_decision_memory_context(network)
        decision = network.final_decision_maker.decide(
            question=network.current_question or "",
            stage1_result=network.last_stage1_result,
            top_k_outputs=stage2_traces,
            top_k_indices=network.last_top_k_indices,
            importance_scores=network.last_importance,
            memory_context=memory_context,
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
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? _build_final_decision_memory_context 瘚?嚗???AgentNetworkHelper ??蝔?瘙???_build_final_decision_memory_context 撠?????????雿?蝯??Ｙ???
        
        Args:
            network: 閮蝟餌絞???炎蝝Ｙ??神?亥?????隞??
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        runtime = getattr(network, "runtime", None)
        graph_memory = getattr(runtime, "graph_memory", None)
        question = self._normalize_text(getattr(network, "current_question", "") or "")
        if runtime is None or graph_memory is None or not question:
            return ""

        try:
            context = getattr(runtime, "current_context", {}) or {}
            source = str(context.get("benchmark") or context.get("source") or "system").strip().lower() or "system"
            task_id = str(context.get("task_id") or context.get("id") or context.get("sample_id") or "").strip() or None
            attachment = context.get("attachment") or {}
            attachment_type = str(attachment.get("extension", "") or "").strip().lower().lstrip(".") or None
            result = graph_memory.retrieve_context(
                task_id=task_id,
                input_text=question,
                source=f"{source}_final_decision",
                attachment_type=attachment_type,
                limit=3,
                injection_target="final_decision",
            )
            guidance = str(result.get("guidance", "") or "")
            runtime.record_memory_read(
                {
                    "stage": "final_decision",
                    "source": "graph_memory",
                    "task_id": result.get("task_id") or task_id,
                    "task_type": (result.get("retrieval") or {}).get("task_type"),
                    "related_task_ids": result.get("related_task_ids", []),
                    "insight_ids": [
                        item.get("insight_id")
                        for item in result.get("insights", [])
                        if isinstance(item, dict) and item.get("insight_id")
                    ],
                    "seed_task_hits": result.get("seed_task_hits", []),
                    "expanded_task_hits": result.get("expanded_task_hits", []),
                }
            )
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
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? _write_shared_memory_records 瘚?嚗???AgentNetworkHelper ??蝔?瘙???_write_shared_memory_records 撠?????????雿?蝯??Ｙ???
        
        Args:
            network: 閮蝟餌絞???炎蝝Ｙ??神?亥?????隞??
            stage2_traces: 閮蝟餌絞???炎蝝Ｙ??神?亥?????隞??
            decision: 閮蝟餌絞???炎蝝Ｙ??神?亥?????隞??
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 None??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        return
    def _extract_judged_stage2_candidates(
        self,
        stage2_traces: list[dict[str, Any]],
        decision: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        鞎痊?瑁? AgentNetworkHelper 銝剔? _extract_judged_stage2_candidates 瘚?嚗???AgentNetworkHelper ??蝔?瘙???_extract_judged_stage2_candidates 撠?????????雿?蝯??Ｙ???
        
        Args:
            stage2_traces: 甇斗?蝔?閬蝙?函?頛詨鞈???
            decision: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 list[dict[str, Any]]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
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
