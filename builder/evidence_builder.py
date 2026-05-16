from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from network.slm_agent import SLM_Agent
from parser import try_parse_json
from utils.network_utils import should_use_calculator, should_use_search
from .attachment import AttachmentEvidenceBuilder
from .stage2_tool_router import Stage2ToolRouter, Stage2ToolRoutingInput

if TYPE_CHECKING:
    from .search_evidence_builder import SearchEvidenceBuilder
    from .search_query_planner import SearchQueryPlanner


class EvidenceBuilder:
    """整合工具、記憶、附件、搜尋與 RAG 證據的建構器。

    負責依照題目、stage 與 routing 結果，產生 agent prompt 可直接使用的 evidence context，
    並回傳工具使用紀錄與各類 evidence 的啟用狀態。

    Args:
        tool_manager: 工具管理器，用於執行 calculator、search、rag 等工具。
        memory_tool: 舊式記憶工具介面，保留相容用；目前主要透過 runtime.memory_service 使用 graph memory。
        runtime: NetworkRuntime，提供 TaskContext、共享附件/搜尋狀態、trace 與 memory service。
        search_query_planner: 搜尋查詢規劃器；未提供時可自動初始化。
        search_evidence_builder: 搜尋證據格式化器；未提供時可自動初始化。
        initialize_search_helpers: 是否在初始化時建立搜尋輔助物件。
    """
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
        """初始化 EvidenceBuilder 與可選的搜尋輔助元件。

        Args:
            tool_manager: 工具管理器。
            memory_tool: 舊式記憶工具介面，保留相容用。
            runtime: NetworkRuntime 或相容的 runtime 物件。
            search_query_planner: 外部注入的搜尋查詢規劃器。
            search_evidence_builder: 外部注入的搜尋證據建構器。
            initialize_search_helpers: 是否自動建立缺少的搜尋輔助元件。
        """
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
        """依題目與 stage 建立完整 evidence bundle。

        會依 routing 結果組合 attachment、memory、calculator、python solver、search 與 RAG context，
        並回傳工具使用紀錄與各 evidence 類型的使用狀態。

        Args:
            question: 使用者問題或 benchmark 題目。
            agent_id: 目前呼叫 evidence 的 agent 識別。
            stage: evidence 所屬流程階段，例如 stage1、stage2、final_decision。
            router_model_name: 可選的 LLM router 模型名稱。
            shared_search_bundle: Stage 2 共用搜尋結果，提供時會避免重複搜尋。
            include_routed_tools: 是否執行 tool routing 與 routed evidence。
            include_attachment: 是否包含附件證據。

        Returns:
            包含 tool_context、各類 evidence context、tool_usage、routing 與 used flags 的 dict。
        """
        if shared_search_bundle is None and stage == "stage2" and self.runtime is not None:
            runtime_bundle = getattr(self.runtime, "shared_stage2_search_bundle", None)
            if isinstance(runtime_bundle, dict):
                shared_search_bundle = runtime_bundle

        tool_usage = []
        attachment = self._build_attachment_evidence(question) if include_attachment else self._empty_tool_result()
        shared_routing = self._get_shared_stage2_routing(shared_search_bundle, stage=stage)
        if shared_routing is not None:
            routing = shared_routing
        elif include_routed_tools:
            if self.runtime is not None and hasattr(self.runtime, "measure"):
                with self.runtime.measure(
                    "route_tools",
                    stage=stage,
                    category="tool_routing",
                    event_type="tool_routing",
                    metadata={"router_model_name": router_model_name or "", "agent_id": agent_id},
                    input_summary=str(question or "")[:240],
                ) as latency:
                    routing = self._route_tools(question, router_model_name, stage=stage)
                    latency.metadata["routing"] = dict(routing or {})
            else:
                routing = self._route_tools(question, router_model_name, stage=stage)
        else:
            routing = {
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
        memory = (
            self._build_memory_evidence(question, agent_id=agent_id, stage=stage)
            if stage.startswith("stage2") or routing["use_memory"]
            else self._empty_context_result()
        )
        rag = self._build_rag_evidence(question) if routing["use_rag"] else self._empty_context_result()
        solver = (
            self._build_python_solver_guidance(question, routing)
            if routing.get("use_python_solver")
            else self._empty_context_result()
        )

        tool_usage.extend(calc["tool_usage"])
        tool_usage.extend(solver.get("tool_usage", []))
        tool_usage.extend(search["tool_usage"])

        tool_context = self._join_contexts(
            [
                attachment["context"],
                memory["context"],
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
        """建立 Stage 2 top-k agents 可共用的搜尋證據 bundle。

        此方法會先做 tool routing；若判斷需要搜尋，則執行一次搜尋並把結果包成可重用的 bundle，
        供多個 Stage 2 agent 共用，避免重複查詢。

        Args:
            question: 使用者問題或 benchmark 題目。
            agent_id: 共享搜尋執行者識別。
            stage: trace 中使用的 stage 名稱。
            router_model_name: 可選的 router 模型名稱。

        Returns:
            包含 enabled、search_context、tool_usage、queries、query_plan、search_runs 與 routing 的 dict。
        """
        if self.runtime is not None and hasattr(self.runtime, "measure"):
            with self.runtime.measure(
                "route_tools",
                stage=stage,
                category="tool_routing",
                event_type="tool_routing",
                metadata={"router_model_name": router_model_name or "", "agent_id": agent_id},
                input_summary=str(question or "")[:240],
            ) as latency:
                routing = self._route_tools(question, router_model_name, stage=stage)
                latency.metadata["routing"] = dict(routing or {})
        else:
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
        """回傳未使用工具時的標準空結果。"""
        return {"tool_usage": [], "context": "", "used": False}

    def _empty_context_result(self) -> dict[str, Any]:
        """回傳未產生 context 時的標準空結果。"""
        return {"context": "", "used": False}

    def _get_shared_stage2_routing(
        self,
        shared_search_bundle: dict[str, Any] | None,
        *,
        stage: str,
    ) -> dict[str, Any] | None:
        if not stage.startswith("stage2") or not isinstance(shared_search_bundle, dict):
            return None
        routing = shared_search_bundle.get("routing")
        if not isinstance(routing, dict) or not routing:
            return None
        reused = dict(routing)
        reused["shared_stage2_routing"] = True
        return reused

    def _route_tools(
        self,
        question: str,
        router_model_name: str | None,
        *,
        stage: str = "stage2",
    ) -> dict[str, Any]:
        """判斷目前題目與 stage 應使用哪些工具。

        Stage 2 會優先使用 graph memory 的 task type 與 policy 進行 routing；
        若無法取得 graph routing，則退回 keyword 與可選 LLM router。

        Args:
            question: 使用者問題或 benchmark 題目。
            router_model_name: 可選的 LLM router 模型名稱。
            stage: 目前流程階段。

        Returns:
            包含 use_calculator、use_search、use_python_solver、use_memory、use_rag 與 routing metadata 的 dict。
        """
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
                    '"use_python_solver": true/false, "use_rag": true/false}.'
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
                    "- rag: for local knowledge base retrieval.\n\n"
                    "Only enable a tool if it is likely useful."
                ),
            },
        ]

        try:
            router = SLM_Agent(model_name=router_model_name)
            if runtime := getattr(self, "runtime", None):
                if hasattr(runtime, "measure"):
                    with runtime.measure(
                        "tool_router_llm",
                        stage=stage,
                        category="llm_call",
                        event_type="llm_call",
                        metadata={"agent_id": "tool_router", "model_name": router_model_name or ""},
                        input_summary=str(question or "")[:240],
                    ) as latency:
                        raw = router.invoke(messages)
                        latency.metadata["output_summary"] = str(raw or "")[:240]
                else:
                    raw = router.invoke(messages)
            else:
                raw = router.invoke(messages)
            parsed = try_parse_json(raw) or {}

            return {
                "use_calculator": bool(parsed.get("use_calculator", fallback["use_calculator"])),
                "use_search": bool(parsed.get("use_search", fallback["use_search"])),
                "use_python_solver": bool(parsed.get("use_python_solver", fallback["use_python_solver"])),
                "use_memory": False,
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
        """使用 graph memory retrieval 結果建立 Stage 2 tool routing 決策。

        Args:
            question: 使用者問題或 benchmark 題目。

        Returns:
            routing dict；若 graph memory 不可用或失敗則回傳 None。
        """
        runtime = getattr(self, "runtime", None)
        memory_service = getattr(runtime, "memory_service", None)
        if memory_service is None:
            return None

        try:
            task_id = self._resolve_task_id(question)
            result = memory_service.retrieve_context(
                question=question,
                stage="stage2_tool_router",
                injection_target="stage2_tool_router",
                source="stage2_tool_router",
                task_id=task_id,
                limit=3,
            )
            if result is None:
                return None
            retrieval = result.get("retrieval", {}) or {}
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
        """從 runtime 的 TaskContext 取得 task_id；若缺少則依題目內容產生穩定 ID。"""
        runtime = getattr(self, "runtime", None)
        if hasattr(runtime, "get_task_context"):
            task_context = runtime.get_task_context()
            if task_context.task_id:
                return task_context.task_id
        else:
            context = getattr(runtime, "current_context", {}) or {}
            for key in ("task_id", "id", "sample_id"):
                value = str(context.get(key, "") or "").strip()
                if value:
                    return value
        normalized = str(question or "").strip().encode("utf-8", errors="ignore")
        digest = hashlib.sha1(normalized).hexdigest()[:12]
        return f"gaia_task_{digest}"

    def _resolve_attachment_type(self) -> str | None:
        """從 TaskContext 或目前附件 metadata 推斷附件副檔名/類型。"""
        runtime = getattr(self, "runtime", None)
        if hasattr(runtime, "get_task_context"):
            task_context = runtime.get_task_context()
            if task_context.attachment_type:
                return task_context.attachment_type
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
        """建構 calculator evidence。

        會先使用 routing 提供的 expression，若沒有則嘗試從題目抽取安全算式；
        無法抽取時不執行自然語言 calculator call。

        Args:
            question: 使用者問題或 benchmark 題目。
            agent_id: 呼叫工具的 agent 識別。
            stage: 目前流程階段。
            expression: 已由 routing 決定的 calculator expression。

        Returns:
            包含 tool_usage、context 與 used flag 的 dict。
        """
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
        """依 routing 結果建立 Python solver guidance。

        目前不執行真實 solver，只提供結構化建模與檢查建議給後續 agent。
        """
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
        """規劃並執行搜尋工具，回傳整理後的搜尋證據。

        Args:
            question: 使用者問題或 benchmark 題目。
            agent_id: 呼叫搜尋工具的 agent 識別。
            stage: 目前流程階段。

        Returns:
            包含搜尋 tool_usage、context、queries、query_plan 與 search_runs 的 dict。
        """
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
        """重用 Stage 2 shared search bundle，避免單一 agent 重複搜尋。"""
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
        """建立指向 shared search bundle 的工具使用紀錄。"""
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
        """建立可供多個 agent 共用的附件證據 bundle。

        Args:
            question: 使用者問題或 benchmark 題目。
            agent_id: 共享附件讀取者識別。
            stage: trace 中使用的 stage 名稱。

        Returns:
            包含 attachment_context、tool_usage、metadata 與 used flags 的 dict。
        """
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
        """建構附件證據；若已有 shared attachment bundle 則直接重用。"""
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
        """建立指向 shared attachment bundle 的工具使用紀錄。"""
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
        """依題目內容產生 shared search bundle 的穩定識別碼。"""
        normalized = str(question or "").strip().encode("utf-8", errors="ignore")
        digest = hashlib.md5(normalized).hexdigest()[:12]
        return f"stage2-search-{digest}"

    def _question_requires_web(self, question: str) -> bool:
        """以簡單關鍵字判斷題目是否明確需要網路搜尋。"""
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

    def _build_memory_evidence(
        self,
        question: str,
        *,
        agent_id: str = "unknown_agent",
        stage: str = "stage2",
    ) -> dict[str, Any]:
        """建構 Stage 2 memory evidence。

        目前只使用 graph memory；若沒有可用 guidance，回傳 `Memory evidence: None`。
        """
        graph_result = self._build_graph_memory_evidence(question, agent_id=agent_id, stage=stage)
        if graph_result["used"]:
            return graph_result

        return {"context": "Memory evidence:\nNone", "used": False}

    def _build_graph_memory_evidence(
        self,
        question: str,
        *,
        agent_id: str,
        stage: str,
    ) -> dict[str, Any]:
        """透過 MemoryService 取得 graph memory guidance 並包成 prompt context。"""
        runtime = getattr(self, "runtime", None)
        memory_service = getattr(runtime, "memory_service", None)
        if runtime is None or memory_service is None:
            return self._empty_context_result()

        try:
            task_id = self._resolve_task_id(question)
            result = memory_service.retrieve_context(
                question=question,
                stage=stage,
                injection_target="stage2_top_k",
                source=f"{stage}_top_k_agent",
                task_id=task_id,
                limit=3,
                agent_id=agent_id,
            )
            if result is None:
                return self._empty_context_result()
        except Exception as exc:
            print(f"[WARN] {stage} graph memory evidence failed: {exc}")
            return self._empty_context_result()

        guidance = str(result.get("guidance", "") or "").strip()
        if not guidance:
            return self._empty_context_result()

        return {"context": f"Memory evidence:\n{guidance}", "used": True}

    def _build_rag_evidence(self, question: str) -> dict[str, Any]:
        """從 RAG tool 取得本地知識庫相關 context。"""
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
        """合併多段非空 evidence context。"""
        valid = [ctx for ctx in contexts if ctx and ctx.strip()]
        return "\n\n".join(valid).strip()

    def _format_memory_items(self, memories: list[Any]) -> str:
        """將舊式 memory item 清單格式化成文字。

        此方法保留給相容路徑使用；新的 graph memory 注入不依賴它。
        """
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
