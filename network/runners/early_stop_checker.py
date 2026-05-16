from __future__ import annotations

from typing import Any

from ..policies.early_stop import EarlyStopPolicy


class EarlyStopChecker:
    """EarlyStopChecker 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    def __init__(self, policy: EarlyStopPolicy | None = None):
        self.policy = policy or EarlyStopPolicy()

    def reset(self, network: Any) -> None:
        network.last_early_stop_decision = None
        network.last_early_stop_trace = []

    def check_stage1_round(self, network: Any, round_id: int) -> bool:
        active_indices = [
            idx for idx, node in enumerate(network.nodes)
            if getattr(node, "active", False)
        ]
        context = (
            network.get_task_context().to_dict()
            if hasattr(network, "get_task_context")
            else dict(getattr(network, "current_context", {}) or {})
        )
        decision = self.policy.check(
            network=network,
            round_id=round_id,
            active_indices=active_indices,
            context=context,
        )
        network.last_early_stop_trace.append(decision.__dict__)
        if decision.should_stop:
            network.last_early_stop_decision = decision.__dict__
            print(
                "[EARLY-STOP] "
                f"benchmark={decision.benchmark} round={decision.round_id} "
                f"reason={decision.reason} cluster={decision.cluster_size}/"
                f"{decision.active_answer_count}"
            )
            return True
        return False
