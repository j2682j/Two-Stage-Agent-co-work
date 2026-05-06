from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from network.slm_agent import SLM_4b_Agent
from parser import try_parse_json
from utils.network_utils import should_use_calculator, should_use_search
from .attachment import AttachmentEvidenceBuilder
from .stage2_tool_router import Stage2ToolRouter, Stage2ToolRoutingInput

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
        self.attachment_evidence_builder = AttachmentEvidenceBuilder()
        self.stage2_tool_router = Stage2ToolRouter()

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
        include_routed_tools: bool = True,
        include_attachment: bool = True,
    ) -> dict[str, Any]:
        if shared_search_bundle is None and stage == "stage2" and self.runtime is not None:
            runtime_bundle = getattr(self.runtime, "shared_stage2_search_bundle", None)
            if isinstance(runtime_bundle, dict):
                shared_search_bundle = runtime_bundle

        tool_usage = []
        attachment = self._build_attachment_evidence(question) if include_attachment else self._empty_tool_result()
        routing = self._route_tools(question, router_model_name, stage=stage) if include_routed_tools else {
            "use_calculator": False,
            "use_search": False,
            "use_python_solver": False,
            "use_memory": False,
            "use_rag": False,
        }
        if attachment["used"] and not self._question_requires_web(question):
            routing["use_search"] = False
        if shared_search_bundle is not None:
            routing["use_search"] = bool(shared_search_bundle.get("enabled"))

        calc = (
            self._build_calculator_evidence(
                question,
                agent_id,
                stage,
                expression=routing.get("calculator_expression"),
            )
            if routing["use_calculator"]
            else self._empty_tool_result()
        )
        if routing["use_search"]:
            if shared_search_bundle is not None:
                search = self._reuse_shared_search_evidence(shared_search_bundle)
            else:
                search = self._build_search_evidence(question, agent_id, stage)
        else:
            search = self._empty_tool_result()
        memory = self._build_memory_evidence(question) if routing["use_memory"] else self._empty_context_result()
        rag = self._build_rag_evidence(question) if routing["use_rag"] else self._empty_context_result()
        solver = (
            self._build_python_solver_guidance(question, routing)
            if routing.get("use_python_solver")
            else self._empty_context_result()
        )

        tool_usage.extend(calc["tool_usage"])
        tool_usage.extend(solver.get("tool_usage", []))
        tool_usage.extend(search["tool_usage"])

        # Memory is retrieved separately but is not injected into prompt tool evidence.
        tool_context = self._join_contexts(
            [
                attachment["context"],
                calc["context"],
                solver["context"],
                search["context"],
                rag["context"],
            ]
        )

        tool_usage.extend(attachment["tool_usage"])
        return {
            "tool_usage": tool_usage,
            "tool_context": tool_context,
            "attachment_context": attachment["context"],
            "calculator_context": calc["context"],
            "search_context": search["context"],
            "solver_context": solver["context"],
            "memory_context": memory["context"],
            "rag_context": rag["context"],
            "used_attachment": attachment["used"],
            "used_calculator": calc["used"],
            "used_search": search["used"],
            "used_python_solver": solver["used"],
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
        routing = self._route_tools(question, router_model_name, stage=stage)
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
        if getattr(self.runtime, "current_attachment", None) and not self._question_requires_web(question):
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

    def _route_tools(
        self,
        question: str,
        router_model_name: str | None,
        *,
        stage: str = "stage2",
    ) -> dict[str, Any]:
        if stage.startswith("stage2"):
            graph_decision = self._route_stage2_with_graph(question)
            if graph_decision is not None:
                return graph_decision

        fallback = {
            "use_calculator": should_use_calculator(question),
            "use_search": should_use_search(question),
            "use_python_solver": False,
            "use_memory": False,
            "use_rag": False,
            "calculator_expression": None,
            "task_type": "legacy_keyword",
            "trigger_terms": [],
            "tool_policy": {},
            "routing_reasons": ["legacy keyword fallback"],
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
                "use_python_solver": bool(parsed.get("use_python_solver", fallback["use_python_solver"])),
                "use_memory": bool(parsed.get("use_memory", fallback["use_memory"])),
                "use_rag": bool(parsed.get("use_rag", fallback["use_rag"])),
                "calculator_expression": parsed.get("calculator_expression"),
                "task_type": parsed.get("task_type", fallback["task_type"]),
                "trigger_terms": parsed.get("trigger_terms", []),
                "tool_policy": parsed.get("tool_policy", {}),
                "routing_reasons": parsed.get("reasons", fallback["routing_reasons"]),
            }
        except Exception:
            return fallback

    def _route_stage2_with_graph(self, question: str) -> dict[str, Any] | None:
        runtime = getattr(self, "runtime", None)
        query_graph = getattr(runtime, "query_task_graph", None)
        if query_graph is None:
            return None

        try:
            task_id = self._resolve_task_id(question)
            attachment_type = self._resolve_attachment_type()
            query_graph.register_task(
                task_id,
                question,
                metadata={
                    "source": "stage2_tool_router",
                    "attachment_type": attachment_type,
                },
            )
            classification = query_graph.classify_task(question, attachment_type=attachment_type)
            query_graph.link_task_signals(task_id, classification)
            retrieval = query_graph.retrieve_for_stage1_round0(task_id, question, limit=3)
            routing_input = Stage2ToolRoutingInput(
                question=question,
                task_type=str(retrieval.get("task_type") or "general_reasoning"),
                trigger_terms=list(retrieval.get("trigger_terms") or []),
                tool_policy=dict(retrieval.get("tool_policy") or {}),
                stage1_result=str(getattr(runtime, "current_stage2_stage1_result", "") or "") or None,
                top_k_answers=list(getattr(runtime, "current_stage2_top_k_answers", []) or []),
                judge_scores=list(getattr(runtime, "current_stage2_judge_scores", []) or []),
                has_attachment=bool(getattr(runtime, "current_attachment", None)),
            )
            decision = self.stage2_tool_router.route(routing_input).to_dict()
            decision["use_memory"] = False
            decision["use_rag"] = False
            decision["routing_reasons"] = decision.pop("reasons", [])
            decision["task_id"] = task_id
            return decision
        except Exception as exc:
            print(f"[WARN] stage2 graph tool routing failed: {exc}")
            return None

    def _resolve_task_id(self, question: str) -> str:
        runtime = getattr(self, "runtime", None)
        context = getattr(runtime, "current_context", {}) or {}
        for key in ("task_id", "id", "sample_id"):
            value = str(context.get(key, "") or "").strip()
            if value:
                return value
        normalized = str(question or "").strip().encode("utf-8", errors="ignore")
        digest = hashlib.sha1(normalized).hexdigest()[:12]
        return f"gaia_task_{digest}"

    def _resolve_attachment_type(self) -> str | None:
        runtime = getattr(self, "runtime", None)
        attachment = getattr(runtime, "current_attachment", None) or {}
        for key in ("extension", "file_extension", "type"):
            value = str(attachment.get(key, "") or "").strip().lower().lstrip(".")
            if value:
                return value
        path = str(attachment.get("path", "") or attachment.get("file_path", "") or "").strip()
        if "." in path:
            return path.rsplit(".", 1)[-1].lower()
        return None

    def _build_calculator_evidence(
        self,
        question: str,
        agent_id: str,
        stage: str,
        expression: str | None = None,
    ) -> dict[str, Any]:
        if self.tool_manager is None:
            return self._empty_tool_result()

        calculator_expression = expression or self.stage2_tool_router.extract_calculator_expression(question)
        if not calculator_expression:
            return {
                "tool_usage": [
                    {
                        "ok": False,
                        "tool_name": "python_calculator",
                        "output_text": "",
                        "raw_result": None,
                        "error": "No safe calculator expression extracted; raw natural-language calculator call skipped.",
                    }
                ],
                "context": "",
                "used": False,
            }

        try:
            result = self.tool_manager.execute_tool(
                "python_calculator",
                {"expression": calculator_expression},
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
        result["calculator_expression"] = calculator_expression

        context = ""
        if result.get("ok"):
            context = (
                "Calculator evidence:\n"
                f"Tool used: {result['tool_name']}\n"
                f"Expression: {calculator_expression}\n"
                f"Tool output: {result['output_text']}\n"
            )

        return {
            "tool_usage": [result],
            "context": context,
            "used": bool(context),
        }

    def _build_python_solver_guidance(self, question: str, routing: dict[str, Any]) -> dict[str, Any]:
        task_type = str(routing.get("task_type", "general_reasoning") or "general_reasoning")
        reasons = routing.get("routing_reasons") or routing.get("reasons") or []
        reason_text = "; ".join(str(item) for item in reasons[:4] if str(item).strip())
        guidance = (
            "Python solver guidance:\n"
            f"Task type: {task_type}\n"
            "Recommended approach: construct a small explicit model, table, simulation, dynamic program, "
            "or data-processing plan before choosing the final answer.\n"
        )
        if reason_text:
            guidance += f"Routing reasons: {reason_text}\n"
        usage = {
            "ok": True,
            "tool_name": "python_solver",
            "output_text": "python_solver recommended as structured solver guidance; no executable solver tool is registered.",
            "raw_result": {
                "executed": False,
                "task_type": task_type,
                "routing_reasons": list(reasons),
            },
            "error": None,
        }
        return {"context": guidance, "used": True, "tool_usage": [usage]}

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

    def build_shared_attachment_bundle(
        self,
        *,
        question: str,
        agent_id: str = "shared_attachment_reader",
        stage: str = "attachment_shared",
    ) -> dict[str, Any]:
        attachment = getattr(self.runtime, "current_attachment", None)
        bundle = {
            "enabled": False,
            "used": False,
            "attachment_context": "",
            "tool_usage": [],
            "metadata": {},
            "agent_id": agent_id,
            "stage": stage,
        }
        if not attachment:
            return bundle

        result = self.attachment_evidence_builder.build(question, attachment)
        bundle.update(
            {
                "enabled": bool(result["used"]),
                "used": bool(result["used"]),
                "attachment_context": result["context"],
                "tool_usage": result["tool_usage"],
                "metadata": result.get("metadata", {}),
            }
        )
        return bundle

    def _build_attachment_evidence(self, question: str) -> dict[str, Any]:
        runtime_bundle = getattr(self.runtime, "shared_attachment_bundle", None)
        if isinstance(runtime_bundle, dict) and runtime_bundle.get("used"):
            return {
                "tool_usage": [self._build_shared_attachment_reference(runtime_bundle)],
                "context": str(runtime_bundle.get("attachment_context", "") or ""),
                "used": True,
            }

        attachment = getattr(self.runtime, "current_attachment", None)
        if not attachment:
            return self._empty_tool_result()

        result = self.attachment_evidence_builder.build(question, attachment)
        return {
            "tool_usage": result["tool_usage"],
            "context": result["context"],
            "used": result["used"],
        }

    def _build_shared_attachment_reference(self, shared_attachment_bundle: dict[str, Any]) -> dict[str, Any]:
        metadata = shared_attachment_bundle.get("metadata") or {}
        file_path = str(metadata.get("file_path", "") or "")
        file_type = str(metadata.get("file_type", "") or "")
        reader = str(metadata.get("reader", "") or "")
        output_lines = ["[shared attachment evidence reused]"]
        if file_path:
            output_lines.append(f"file_path={file_path}")
        if file_type:
            output_lines.append(f"file_type={file_type}")
        if reader:
            output_lines.append(f"reader={reader}")

        return {
            "ok": True,
            "tool_name": "attachment_reader",
            "output_text": "\n".join(output_lines),
            "raw_result": {
                "shared": True,
                "metadata": metadata,
                "source_stage": shared_attachment_bundle.get("stage", "attachment_shared"),
            },
            "error": None,
            "shared": True,
        }

    def _make_shared_search_id(self, question: str) -> str:
        normalized = str(question or "").strip().encode("utf-8", errors="ignore")
        digest = hashlib.md5(normalized).hexdigest()[:12]
        return f"stage2-search-{digest}"

    def _question_requires_web(self, question: str) -> bool:
        normalized = str(question or "").lower()
        web_markers = [
            "website",
            "web site",
            "webpage",
            "url",
            "http://",
            "https://",
            "internet",
            "search",
            "google",
            "wikipedia",
            "latest",
            "current",
            "today",
            "online",
        ]
        return any(marker in normalized for marker in web_markers)

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
