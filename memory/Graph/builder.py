from __future__ import annotations

from typing import Any

from .interaction_graph import AgentMessage, InteractionGraph


class InteractionGraphBuilder:
    """
    負責在 memory.graph.builder 中封裝 InteractionGraphBuilder，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def build_from_network(
        self,
        *,
        network: Any,
        sample: dict[str, Any] | None = None,
        sample_result: dict[str, Any] | None = None,
    ) -> InteractionGraph:
        """
        負責執行 InteractionGraphBuilder 中的 build_from_network 流程，建立記憶圖或任務記錄結構，供後續檢索、寫入與提示注入使用。
        
        Args:
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
            sample: 記憶系統提供的檢索結果、寫入資料或操作介面。
            sample_result: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InteractionGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        sample = sample or {}
        sample_result = sample_result or {}
        task_id = self._task_id(sample, network)
        question = self._question(sample, network)
        label = self._label(sample_result)
        expected = self._expected(sample, sample_result)
        final_decision = getattr(network, "last_final_decision", None) or {}
        final_result = str(final_decision.get("final_result", getattr(network, "last_stage1_result", "")) or "")

        trace = InteractionGraph(
            task_id=task_id,
            task_main=str(sample.get("benchmark", "GAIA") or "GAIA"),
            task_description=question,
            label=label,
            extra_fields={
                "benchmark": str(sample.get("benchmark", "GAIA") or "GAIA"),
                "level": sample.get("level"),
                "expected": expected,
                "stage1_result": getattr(network, "last_stage1_result", None),
                "final_result": final_result,
                "predicted": sample_result.get("predicted"),
                "exact_match": bool(sample_result.get("exact_match", False)),
                "partial_match": bool(sample_result.get("partial_match", False)),
                "score": sample_result.get("score"),
                "top_k_indices": list(getattr(network, "last_top_k_indices", []) or []),
                "selected_agent_idx": final_decision.get("selected_agent_idx"),
            },
        )

        node_id_by_idx: dict[int, str] = {}
        self._add_stage1_round_states(trace, network, node_id_by_idx)
        self._add_stage1_judge_state(trace, network, node_id_by_idx)
        stage2_node_ids = self._add_stage2_state(trace, network, node_id_by_idx)
        self._add_final_decision_state(trace, network, sample_result, stage2_node_ids)
        return trace

    def _add_stage1_round_states(
        self,
        trace: InteractionGraph,
        network: Any,
        node_id_by_idx: dict[int, str],
    ) -> None:
        """
        負責執行 InteractionGraphBuilder 中的 _add_stage1_round_states 流程，依照 InteractionGraphBuilder 的流程需求處理 _add_stage1_round_states 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            trace: 記憶系統提供的檢索結果、寫入資料或操作介面。
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
            node_id_by_idx: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        activation_trace = getattr(network, "last_stage1_activation_trace", []) or []
        network_nodes = list(getattr(network, "nodes", []) or [])
        index_by_obj = {id(node): idx for idx, node in enumerate(network_nodes)}

        for entry in activation_trace:
            round_id = int(entry.get("round", 0) or 0)
            node_indices = list(entry.get("node_indices", []) or [])
            state_id = f"{trace.task_id}:stage1_round{round_id}"
            trace.start_state(
                state_id=state_id,
                state_type="stage1_round",
                stage="stage1",
                round_id=round_id,
                action=f"activate stage1 round {round_id}",
                observation=f"{len(node_indices)} active agent replies",
                reward=None,
                extra_fields={"active_node_indices": node_indices},
            )
            for node_idx in node_indices:
                if not isinstance(node_idx, int) or node_idx < 0 or node_idx >= len(network_nodes):
                    continue
                node = network_nodes[node_idx]
                upstream_refs = []
                same_state_upstream_ids = []
                for edge in getattr(node, "from_edges", []) or []:
                    src_idx = index_by_obj.get(id(getattr(edge, "a1", None)))
                    if src_idx is None:
                        continue
                    src_node_id = node_id_by_idx.get(src_idx)
                    ref = {
                        "state_id": self._state_id_for_node_idx(src_idx, network),
                        "node_id": src_node_id or f"node{src_idx}",
                        "network_node_idx": src_idx,
                        "edge_type": "spatial",
                        "weight": float(getattr(edge, "weight", 0.0) or 0.0),
                    }
                    if src_node_id and src_node_id.startswith(state_id + ":"):
                        same_state_upstream_ids.append(src_node_id)
                    else:
                        upstream_refs.append(ref)

                message = AgentMessage(
                    agent_name=str(getattr(node, "model_name", f"node{node_idx}") or f"node{node_idx}"),
                    message=str(getattr(node, "reply", "") or ""),
                    extra_fields={
                        "node_type": "stage1_agent",
                        "task_id": trace.task_id,
                        "stage": "stage1",
                        "round": round_id,
                        "network_node_idx": node_idx,
                        "model_name": getattr(node, "model_name", None),
                        "active": bool(getattr(node, "active", False)),
                        "reasoning": getattr(node, "reasoning", ""),
                        "final_answer": getattr(node, "answer", ""),
                        "importance": getattr(node, "importance", 0.0),
                        "prompt_tokens": getattr(node, "prompt_tokens", 0),
                        "completion_tokens": getattr(node, "completion_tokens", 0),
                        "stage1_reflection_context": getattr(node, "stage1_reflection_context", ""),
                        "stage1_attachment_context": getattr(node, "stage1_attachment_context", ""),
                        "upstream_refs": upstream_refs,
                    },
                )
                node_id = f"{state_id}:node{node_idx}"
                node_id_by_idx[node_idx] = node_id
                trace.add_message_to_current_state(
                    message,
                    same_state_upstream_ids,
                    node_id=node_id,
                    edge_type="spatial",
                )
            trace.end_state()

    def _add_stage1_judge_state(
        self,
        trace: InteractionGraph,
        network: Any,
        node_id_by_idx: dict[int, str],
    ) -> None:
        """
        負責執行 InteractionGraphBuilder 中的 _add_stage1_judge_state 流程，依照 InteractionGraphBuilder 的流程需求處理 _add_stage1_judge_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            trace: 記憶系統提供的檢索結果、寫入資料或操作介面。
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
            node_id_by_idx: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        network_nodes = list(getattr(network, "nodes", []) or [])
        judged_indices = [
            idx
            for idx, node in enumerate(network_nodes)
            if getattr(node, "active", False)
            and (
                getattr(node, "stage1_judge_score", 0.0)
                or getattr(node, "stage1_judge_reasoning", "")
                or getattr(node, "importance", 0.0)
            )
        ]
        if not judged_indices:
            return

        state_id = f"{trace.task_id}:stage1_judge"
        trace.start_state(
            state_id=state_id,
            state_type="stage1_judge",
            stage="stage1_judge",
            action="evaluate stage1 candidates and select stage1 result",
            observation=str(getattr(network, "last_stage1_result", "") or ""),
            reward=None,
            extra_fields={"judged_node_indices": judged_indices},
        )
        for idx in judged_indices:
            node = network_nodes[idx]
            upstream_node_id = node_id_by_idx.get(idx)
            message = AgentMessage(
                agent_name="stage1_judge",
                message=str(getattr(node, "stage1_judge_reasoning", "") or ""),
                extra_fields={
                    "node_type": "stage1_judge",
                    "task_id": trace.task_id,
                    "target_node_idx": idx,
                    "target_node_id": upstream_node_id,
                    "is_acceptable": bool(getattr(node, "stage1_judge_is_acceptable", False)),
                    "score": float(getattr(node, "stage1_judge_score", 0.0) or 0.0),
                    "adjusted_score": float(getattr(node, "stage1_judge_adjusted_score", 0.0) or 0.0),
                    "approved_answer": getattr(node, "stage1_judge_approved_answer", ""),
                    "suggested_fix": getattr(node, "stage1_judge_suggested_fix", ""),
                    "revised_answer": getattr(node, "stage1_judge_revised_answer", ""),
                    "used_fallback": bool(getattr(node, "stage1_judge_used_fallback", False)),
                    "upstream_refs": [
                        {
                            "node_id": upstream_node_id or f"node{idx}",
                            "network_node_idx": idx,
                            "edge_type": "evaluates",
                        }
                    ],
                },
            )
            trace.add_message_to_current_state(message, node_id=f"{state_id}:judge_node{idx}")

        selector_message = AgentMessage(
            agent_name="stage1_result_selector",
            message=str(getattr(network, "last_stage1_result", "") or ""),
            extra_fields={
                "node_type": "stage1_result_selector",
                "stage1_result": getattr(network, "last_stage1_result", None),
                "importance": list(getattr(network, "last_importance", []) or []),
            },
        )
        trace.add_message_to_current_state(selector_message, node_id=f"{state_id}:selector")
        trace.end_state()

    def _add_stage2_state(
        self,
        trace: InteractionGraph,
        network: Any,
        node_id_by_idx: dict[int, str],
    ) -> dict[int, str]:
        """
        負責執行 InteractionGraphBuilder 中的 _add_stage2_state 流程，依照 InteractionGraphBuilder 的流程需求處理 _add_stage2_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            trace: 記憶系統提供的檢索結果、寫入資料或操作介面。
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
            node_id_by_idx: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[int, str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        stage2_outputs = list(getattr(network, "last_stage2_outputs", []) or [])
        if not stage2_outputs:
            return {}

        state_id = f"{trace.task_id}:stage2"
        trace.start_state(
            state_id=state_id,
            state_type="stage2",
            stage="stage2",
            action="run top_k stage2 agents with routed evidence",
            observation=f"{len(stage2_outputs)} stage2 candidate replies",
            reward=None,
            extra_fields={"top_k_indices": list(getattr(network, "last_top_k_indices", []) or [])},
        )
        stage2_node_ids: dict[int, str] = {}
        for item in stage2_outputs:
            agent_idx = item.get("agent_idx")
            if not isinstance(agent_idx, int):
                continue
            upstream_node_id = node_id_by_idx.get(agent_idx)
            message = AgentMessage(
                agent_name=str(item.get("model_name") or f"stage2_node{agent_idx}"),
                message=str(item.get("reply", "") or ""),
                extra_fields={
                    "node_type": "stage2_agent",
                    "task_id": trace.task_id,
                    "stage": "stage2",
                    "agent_idx": agent_idx,
                    "model_name": item.get("model_name"),
                    "answer": item.get("answer"),
                    "success": bool(item.get("success", False)),
                    "error": item.get("error"),
                    "tool_usage": item.get("tool_usage", []),
                    "routing": item.get("routing", {}),
                    "attachment_context": item.get("attachment_context", ""),
                    "search_context": item.get("search_context", ""),
                    "solver_context": item.get("solver_context", ""),
                    "memory_context": item.get("memory_context", ""),
                    "rag_context": item.get("rag_context", ""),
                    "upstream_refs": [
                        {
                            "node_id": upstream_node_id or f"node{agent_idx}",
                            "network_node_idx": agent_idx,
                            "edge_type": "repairs",
                        }
                    ],
                },
            )
            node_id = f"{state_id}:node{agent_idx}"
            stage2_node_ids[agent_idx] = node_id
            trace.add_message_to_current_state(message, node_id=node_id)
        trace.end_state()
        return stage2_node_ids

    def _add_final_decision_state(
        self,
        trace: InteractionGraph,
        network: Any,
        sample_result: dict[str, Any],
        stage2_node_ids: dict[int, str],
    ) -> None:
        """
        負責執行 InteractionGraphBuilder 中的 _add_final_decision_state 流程，依照 InteractionGraphBuilder 的流程需求處理 _add_final_decision_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            trace: 記憶系統提供的檢索結果、寫入資料或操作介面。
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
            sample_result: 記憶系統提供的檢索結果、寫入資料或操作介面。
            stage2_node_ids: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        decision = getattr(network, "last_final_decision", None) or {}
        final_result = decision.get("final_result", getattr(network, "last_stage1_result", None))
        reward = self._reward(sample_result)
        selected_agent_idx = decision.get("selected_agent_idx")
        upstream_refs = []
        for agent_idx, node_id in stage2_node_ids.items():
            upstream_refs.append(
                {
                    "node_id": node_id,
                    "agent_idx": agent_idx,
                    "edge_type": "candidate",
                    "selected": agent_idx == selected_agent_idx,
                }
            )

        state_id = f"{trace.task_id}:final_decision"
        trace.start_state(
            state_id=state_id,
            state_type="final_decision",
            stage="final_decision",
            action="select final answer",
            observation=str(final_result or ""),
            reward=reward,
            extra_fields={
                "selected_agent_idx": selected_agent_idx,
                "exact_match": bool(sample_result.get("exact_match", False)),
                "partial_match": bool(sample_result.get("partial_match", False)),
            },
        )
        message = AgentMessage(
            agent_name="final_decision",
            message=str(final_result or ""),
            extra_fields={
                "node_type": "final_decision",
                "task_id": trace.task_id,
                "selected_agent_idx": selected_agent_idx,
                "final_result": final_result,
                "success": bool(decision.get("success", False)),
                "critiques": decision.get("critiques", []),
                "intermediate_steps": decision.get("intermediate_steps", []),
                "upstream_refs": upstream_refs,
            },
        )
        trace.add_message_to_current_state(message, node_id=f"{state_id}:final")
        trace.end_state()

    def _state_id_for_node_idx(self, node_idx: int, network: Any) -> str | None:
        """
        負責執行 InteractionGraphBuilder 中的 _state_id_for_node_idx 流程，依照 InteractionGraphBuilder 的流程需求處理 _state_id_for_node_idx 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            node_idx: 記憶系統提供的檢索結果、寫入資料或操作介面。
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for entry in getattr(network, "last_stage1_activation_trace", []) or []:
            if node_idx in (entry.get("node_indices", []) or []):
                return f"{self._task_id({}, network)}:stage1_round{entry.get('round', 0)}"
        return None

    def _task_id(self, sample: dict[str, Any], network: Any) -> str:
        """
        負責執行 InteractionGraphBuilder 中的 _task_id 流程，依照 InteractionGraphBuilder 的流程需求處理 _task_id 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sample: 記憶系統提供的檢索結果、寫入資料或操作介面。
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        task_id = str(sample.get("task_id", "") or "").strip()
        if task_id:
            return task_id
        runtime = getattr(network, "runtime", None)
        if hasattr(runtime, "get_task_context"):
            task_context = runtime.get_task_context()
            if task_context.task_id:
                return task_context.task_id
        context = getattr(runtime, "current_context", {}) or {}
        return str(context.get("task_id", "") or context.get("id", "") or "unknown_task")

    def _question(self, sample: dict[str, Any], network: Any) -> str:
        """
        負責執行 InteractionGraphBuilder 中的 _question 流程，依照 InteractionGraphBuilder 的流程需求處理 _question 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sample: 記憶系統提供的檢索結果、寫入資料或操作介面。
            network: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return str(sample.get("question", "") or getattr(network, "current_question", "") or "")

    def _expected(self, sample: dict[str, Any], sample_result: dict[str, Any]) -> str:
        """
        負責執行 InteractionGraphBuilder 中的 _expected 流程，依照 InteractionGraphBuilder 的流程需求處理 _expected 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sample: 記憶系統提供的檢索結果、寫入資料或操作介面。
            sample_result: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return str(
            sample_result.get("expected")
            or sample_result.get("expected_answer")
            or sample.get("final_answer")
            or ""
        )

    def _label(self, sample_result: dict[str, Any]) -> str:
        """
        負責執行 InteractionGraphBuilder 中的 _label 流程，依照 InteractionGraphBuilder 的流程需求處理 _label 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sample_result: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if sample_result.get("exact_match", False):
            return "success"
        if sample_result.get("partial_match", False):
            return "partial"
        return "failure"

    def _reward(self, sample_result: dict[str, Any]) -> float:
        """
        負責執行 InteractionGraphBuilder 中的 _reward 流程，依照 InteractionGraphBuilder 的流程需求處理 _reward 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sample_result: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if sample_result.get("exact_match", False):
            return 1.0
        if sample_result.get("partial_match", False):
            return 0.35
        return -1.0
