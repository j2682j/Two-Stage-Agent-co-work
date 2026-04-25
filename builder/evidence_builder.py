from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from network.slm_agent import SLM_4b_Agent
from parser import try_parse_json
from utils.network_utils import should_use_calculator, should_use_search

if TYPE_CHECKING:
    from .search_evidence_builder import SearchEvidenceBuilder
    from .search_query_planner import SearchQueryPlanner


class EvidenceBuilder:
    def __init__(
        self,
        tool_manager,
        memory_tool=None,
        runtime=None,
        *,
        search_query_planner: SearchQueryPlanner | None = None,
        search_evidence_builder: SearchEvidenceBuilder | None = None,
        initialize_search_helpers: bool = True,
    ):
        self.tool_manager = tool_manager
        self.memory_tool = memory_tool
        self.runtime = runtime
        self.search_query_planner = search_query_planner
        self.search_evidence_builder = search_evidence_builder

        if initialize_search_helpers:
            if self.search_query_planner is None:
                from .search_query_planner import SearchQueryPlanner

                self.search_query_planner = SearchQueryPlanner()

            if self.search_evidence_builder is None:
                from .search_evidence_builder import SearchEvidenceBuilder

                self.search_evidence_builder = SearchEvidenceBuilder(
                    tool_manager=tool_manager,
                    memory_tool=memory_tool,
                    runtime=runtime,
                    search_query_planner=self.search_query_planner,
                )

    def build(
        self,
        question: str,
        agent_id: str = "unknown_agent",
        stage: str = "stage2",
        router_model_name: str | None = None,
        shared_search_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if shared_search_bundle is None and stage == "stage2" and self.runtime is not None:
            runtime_bundle = getattr(self.runtime, "shared_stage2_search_bundle", None)
            if isinstance(runtime_bundle, dict):
                shared_search_bundle = runtime_bundle

        tool_usage = []
        routing = self._route_tools(question, router_model_name)
        if shared_search_bundle is not None:
            routing["use_search"] = bool(shared_search_bundle.get("enabled"))

        calc = self._build_calculator_evidence(question, agent_id, stage) if routing["use_calculator"] else self._empty_tool_result()
        if routing["use_search"]:
            if shared_search_bundle is not None:
                search = self._reuse_shared_search_evidence(shared_search_bundle)
            else:
                search = self._build_search_evidence(question, agent_id, stage)
        else:
            search = self._empty_tool_result()
        memory = self._build_memory_evidence(question) if routing["use_memory"] else self._empty_context_result()
        rag = self._build_rag_evidence(question) if routing["use_rag"] else self._empty_context_result()

        tool_usage.extend(calc["tool_usage"])
        tool_usage.extend(search["tool_usage"])

        # Memory is retrieved separately but is not injected into prompt tool evidence.
        tool_context = self._join_contexts(
            [
                calc["context"],
                search["context"],
                rag["context"],
            ]
        )

        return {
            "tool_usage": tool_usage,
            "tool_context": tool_context,
            "calculator_context": calc["context"],
            "search_context": search["context"],
            "memory_context": memory["context"],
            "rag_context": rag["context"],
            "used_calculator": calc["used"],
            "used_search": search["used"],
            "used_memory": memory["used"],
            "used_rag": rag["used"],
            "routing": routing,
        }

    def build_shared_stage2_search_bundle(
        self,
        *,
        question: str,
        agent_id: str = "shared_stage2_search",
        stage: str = "stage2_shared_search",
        router_model_name: str | None = None,
    ) -> dict[str, Any]:
        routing = self._route_tools(question, router_model_name)
        bundle = {
            "enabled": False,
            "used": False,
            "search_context": "",
            "tool_usage": [],
            "queries": [],
            "query_plan": None,
            "search_runs": [],
            "routing": routing,
            "shared_search_id": self._make_shared_search_id(question),
        }

        if self.tool_manager is None or not routing.get("use_search"):
            return bundle

        search = self._build_search_evidence(question, agent_id, stage)
        bundle.update(
            {
                "enabled": bool(search["used"]),
                "used": bool(search["used"]),
                "search_context": search["context"],
                "tool_usage": search["tool_usage"],
                "queries": list(search.get("queries") or []),
                "query_plan": search.get("query_plan"),
                "search_runs": list(search.get("search_runs") or []),
            }
        )
        return bundle

    def _empty_tool_result(self) -> dict[str, Any]:
        return {"tool_usage": [], "context": "", "used": False}

    def _empty_context_result(self) -> dict[str, Any]:
        return {"context": "", "used": False}

    def _route_tools(self, question: str, router_model_name: str | None) -> dict[str, bool]:
        fallback = {
            "use_calculator": should_use_calculator(question),
            "use_search": should_use_search(question),
            "use_memory": False,
            "use_rag": False,
        }

        if not router_model_name:
            return fallback

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a tool routing assistant. "
                    "Decide which tools are worth calling for the given question. "
                    "Return JSON only with keys: "
                    '{"use_calculator": true/false, "use_search": true/false, '
                    '"use_memory": true/false, "use_rag": true/false}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{question}\n\n"
                    "Tool descriptions:\n"
                    "- calculator: for explicit arithmetic or unit conversion.\n"
                    "- search: for facts that may require web lookup.\n"
                    "- memory: for prior conversation/task memory.\n"
                    "- rag: for local knowledge base retrieval.\n\n"
                    "Only enable a tool if it is likely useful."
                ),
            },
        ]

        try:
            router = SLM_4b_Agent(model_name=router_model_name)
            raw = router.invoke(messages)
            parsed = try_parse_json(raw) or {}

            return {
                "use_calculator": bool(parsed.get("use_calculator", fallback["use_calculator"])),
                "use_search": bool(parsed.get("use_search", fallback["use_search"])),
                "use_memory": bool(parsed.get("use_memory", fallback["use_memory"])),
                "use_rag": bool(parsed.get("use_rag", fallback["use_rag"])),
            }
        except Exception:
            return fallback

    def _build_calculator_evidence(self, question: str, agent_id: str, stage: str) -> dict[str, Any]:
        if self.tool_manager is None:
            return self._empty_tool_result()

        try:
            result = self.tool_manager.execute_tool(
                "python_calculator",
                {"expression": question},
                agent_id=agent_id,
                stage=stage,
            )
        except Exception as exc:
            result = {
                "ok": False,
                "tool_name": "python_calculator",
                "output_text": "",
                "raw_result": None,
                "error": str(exc),
            }

        context = ""
        if result.get("ok"):
            context = (
                "Calculator evidence:\n"
                f"Tool used: {result['tool_name']}\n"
                f"Tool output: {result['output_text']}\n"
            )

        return {
            "tool_usage": [result],
            "context": context,
            "used": bool(context),
        }

    def _build_search_evidence(self, question: str, agent_id: str, stage: str) -> dict[str, Any]:
        if self.tool_manager is None:
            return self._empty_tool_result()

        planner = self.search_query_planner
        if planner is None:
            from .search_query_planner import SearchQueryPlanner

            planner = SearchQueryPlanner()
            self.search_query_planner = planner

        evidence_formatter = self.search_evidence_builder
        if evidence_formatter is None:
            from .search_evidence_builder import SearchEvidenceBuilder

            evidence_formatter = SearchEvidenceBuilder(
                tool_manager=self.tool_manager,
                memory_tool=self.memory_tool,
                runtime=self.runtime,
                search_query_planner=planner,
            )
            self.search_evidence_builder = evidence_formatter

        plan = planner.plan(question=question, max_queries=3)
        planned_queries = plan.get("queries") or [question]
        precision_needed = bool(plan.get("precision_needed"))

        search_runs: list[dict[str, Any]] = []
        tool_usage: list[dict[str, Any]] = []
        for query in planned_queries:
            try:
                result = self.tool_manager.execute_tool(
                    "search",
                    {
                        "input": query,
                        "mode": "structured",
                        "conditional_fetch": precision_needed,
                        "max_full_page_results": 2,
                    },
                    agent_id=agent_id,
                    stage=stage,
                )
            except Exception as exc:
                result = {
                    "ok": False,
                    "tool_name": "search",
                    "output_text": "",
                    "raw_result": None,
                    "error": str(exc),
                }

            result["planned_query"] = query
            result["query_plan"] = plan
            tool_usage.append(result)
            search_runs.append({"query": query, "result": result})

        context = evidence_formatter.build_planned_search_evidence_block(
            search_runs=search_runs,
            question=question,
        )

        return {
            "tool_usage": tool_usage,
            "context": context,
            "used": bool(context),
            "queries": list(planned_queries),
            "query_plan": plan,
            "search_runs": search_runs,
        }

    def _reuse_shared_search_evidence(self, shared_search_bundle: dict[str, Any]) -> dict[str, Any]:
        context = str(shared_search_bundle.get("search_context", "") or "")
        if not context.strip():
            return self._empty_tool_result()

        return {
            "tool_usage": [self._build_shared_search_reference(shared_search_bundle)],
            "context": context,
            "used": True,
            "queries": list(shared_search_bundle.get("queries") or []),
            "query_plan": shared_search_bundle.get("query_plan"),
            "search_runs": list(shared_search_bundle.get("search_runs") or []),
        }

    def _build_shared_search_reference(self, shared_search_bundle: dict[str, Any]) -> dict[str, Any]:
        queries = [str(query).strip() for query in (shared_search_bundle.get("queries") or []) if str(query).strip()]
        shared_search_id = str(shared_search_bundle.get("shared_search_id", "") or "")
        query_plan = shared_search_bundle.get("query_plan")
        output_lines = ["[shared stage2 search reused]"]
        if shared_search_id:
            output_lines.append(f"shared_search_id={shared_search_id}")
        if queries:
            output_lines.append("queries=" + " | ".join(queries))

        return {
            "ok": True,
            "tool_name": "search",
            "output_text": "\n".join(output_lines),
            "raw_result": {
                "shared": True,
                "shared_search_id": shared_search_id,
                "queries": queries,
                "query_plan": query_plan,
                "source_stage": "stage2_shared_search",
            },
            "error": None,
            "shared": True,
            "shared_search_id": shared_search_id,
            "planned_queries": queries,
            "query_plan": query_plan,
        }

    def _make_shared_search_id(self, question: str) -> str:
        normalized = str(question or "").strip().encode("utf-8", errors="ignore")
        digest = hashlib.md5(normalized).hexdigest()[:12]
        return f"stage2-search-{digest}"

    def _build_memory_evidence(self, question: str) -> dict[str, Any]:
        memory_tool = (
            self.memory_tool
            or getattr(self.runtime, "memory_tool", None)
            or getattr(self.tool_manager, "memory_tool", None)
        )
        memory_manager = getattr(memory_tool, "memory_manager", None)
        if memory_manager is not None:
            try:
                memories = memory_manager.retrieve_memories(
                    query=question,
                    memory_types=["working", "semantic", "episodic"],
                    limit=3,
                    min_importance=0.0,
                )
            except Exception:
                memories = []
        else:
            memories = []

        context = self._format_memory_items(memories)
        if context.strip():
            if self.runtime is not None:
                self.runtime.record_memory_read(
                    {
                        "question": question,
                        "memory_types": ["working", "semantic", "episodic"],
                        "count": len(memories),
                    }
                )
            return {"context": f"Memory evidence:\n{context}", "used": True}

        return self._empty_context_result()

    def _build_rag_evidence(self, question: str) -> dict[str, Any]:
        rag_tool = getattr(self.tool_manager, "rag_tool", None)
        if rag_tool is None:
            return self._empty_context_result()

        try:
            context = rag_tool.get_relevant_context(query=question, limit=3) or ""
        except Exception:
            context = ""

        if context.strip():
            context = f"RAG evidence:\n{context}"

        return {"context": context, "used": bool(context.strip())}

    def _join_contexts(self, contexts: list[str]) -> str:
        valid = [ctx for ctx in contexts if ctx and ctx.strip()]
        return "\n\n".join(valid).strip()

    def _format_memory_items(self, memories: list[Any]) -> str:
        lines: list[str] = []
        for idx, memory in enumerate(memories, start=1):
            content = str(getattr(memory, "content", "") or "").strip()
            if not content:
                continue

            memory_type = str(getattr(memory, "memory_type", "memory") or "memory").strip()
            importance = getattr(memory, "importance", None)
            prefix = f"[{idx}] ({memory_type})"
            if importance is not None:
                try:
                    prefix += f" importance={float(importance):.2f}"
                except Exception:
                    pass
            lines.append(f"{prefix} {content}")

        return "\n".join(lines).strip()
