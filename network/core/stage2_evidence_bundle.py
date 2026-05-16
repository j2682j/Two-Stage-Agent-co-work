from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Stage2EvidenceBundle:
    """Stage2EvidenceBundle 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    agent_id: str
    tool_usage: list[dict[str, Any]] = field(default_factory=list)
    tool_context: str = ""
    attachment_context: str = ""
    search_context: str = ""
    solver_context: str = ""
    memory_context: str = ""
    rag_context: str = ""
    routing: dict[str, Any] = field(default_factory=dict)
    used_attachment: bool = False
    used_search: bool = False
    used_memory: bool = False
    used_calculator: bool = False
    used_python_solver: bool = False
    used_rag: bool = False
    error: str | None = None

    @classmethod
    def from_evidence(cls, agent_id: str, evidence: dict[str, Any]) -> "Stage2EvidenceBundle":
        return cls(
            agent_id=agent_id,
            tool_usage=list(evidence.get("tool_usage") or []),
            tool_context=str(evidence.get("tool_context", "") or ""),
            attachment_context=str(evidence.get("attachment_context", "") or ""),
            search_context=str(evidence.get("search_context", "") or ""),
            solver_context=str(evidence.get("solver_context", "") or ""),
            memory_context=str(evidence.get("memory_context", "") or ""),
            rag_context=str(evidence.get("rag_context", "") or ""),
            routing=dict(evidence.get("routing") or {}),
            used_attachment=bool(evidence.get("used_attachment")),
            used_search=bool(evidence.get("used_search")),
            used_memory=bool(evidence.get("used_memory")),
            used_calculator=bool(evidence.get("used_calculator")),
            used_python_solver=bool(evidence.get("used_python_solver")),
            used_rag=bool(evidence.get("used_rag")),
        )

    @classmethod
    def failed(cls, agent_id: str, error: str) -> "Stage2EvidenceBundle":
        return cls(agent_id=agent_id, error=error)

    def tool_context_or_empty(self) -> str:
        context = str(self.tool_context or "").strip()
        return context if context else "No tool result available."
