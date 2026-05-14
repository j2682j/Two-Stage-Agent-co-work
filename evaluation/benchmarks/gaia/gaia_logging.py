from __future__ import annotations

from typing import Any

from evaluation.benchmark_logger import (
    BenchmarkLogger,
    TeeStream,
    open_utf8_text_log,
    restore_stdio,
    setup_utf8_log,
)


class GaiaBenchmarkLogger(BenchmarkLogger):
    """
    負責提供 GAIA benchmark 專用 logger 的共用繼承點。

    Args:
        無。

    Returns:
        GaiaBenchmarkLogger 實例，可使用 BenchmarkLogger 的共用 logging 方法。

    限制或副作用:
        目前 GAIA 的 compact log 仍保留在模組函式中；此類別保留給後續把 GAIA logger 轉成物件式實作。
    """

    pass


def print_stage2_and_final(network: Any) -> None:
    """
    負責將目前 AgentNetwork 的 stage1、stage2 與 final decision 摘要印到 full log。

    Args:
        network: GAIA 評估使用的 AgentNetwork。

    Returns:
        無。

    限制或副作用:
        只讀取 network 上的 last_* 狀態並 print，不會修改網路狀態。
    """
    print(f"   [STAGE1] result={network.last_stage1_result!r}")
    print(f"   [STAGE1] top_k_indices={network.last_top_k_indices}")

    stage2_outputs = network.last_stage2_outputs or []
    if not stage2_outputs:
        print("   [STAGE2] no outputs")
    else:
        for item in stage2_outputs:
            tool_names = [
                tool.get("tool_name")
                for tool in item.get("tool_usage", []) or []
                if isinstance(tool, dict) and tool.get("tool_name")
            ]
            print(
                "   [STAGE2] "
                f"idx={item.get('agent_idx')} "
                f"model={item.get('model_name')} "
                f"success={item.get('success', False)} "
                f"answer={item.get('answer')!r} "
                f"tools={tool_names} "
                f"routing={item.get('routing', {})} "
                f"error={item.get('error')!r}"
            )

    decision = network.last_final_decision or {}
    if not decision:
        print("   [FINAL] no decision")
        return

    step_names = [
        step.get("step")
        for step in decision.get("intermediate_steps", []) or []
        if isinstance(step, dict)
    ]
    print(
        "   [FINAL] "
        f"success={decision.get('success', False)} "
        f"selected_agent_idx={decision.get('selected_agent_idx')} "
        f"final_result={decision.get('final_result')!r}"
    )
    print(
        "   [FINAL] "
        f"critiques={len(decision.get('critiques', []) or [])} "
        f"steps={step_names}"
    )


def _write_indented_block(handle: Any, text: Any, indent: str = "    ") -> None:
    """
    負責將多行文字以固定縮排寫入 compact log。

    Args:
        handle: 已開啟的文字檔 handle。
        text: 要寫入的文字內容。
        indent: 每一行前方要加入的縮排。

    Returns:
        無。

    限制或副作用:
        空內容會寫入 `(none)`；此函式會直接寫入 handle。
    """
    BenchmarkLogger.write_indented_block(handle, text, indent=indent)


def _write_graph_memory_trace(handle: Any, runtime: Any, task_id: str | None = None) -> None:
    """
    負責將 GraphMemory 的讀取、寫入與 interaction state chain 寫入 compact log。

    Args:
        handle: 已開啟的文字檔 handle。
        runtime: AgentNetwork.runtime，用於讀取 shared_memory_reads/shared_memory_writes。
        task_id: 若提供，僅輸出該 task_id 對應的 trace。

    Returns:
        無。

    限制或副作用:
        只讀取 runtime trace，不會觸發新的記憶檢索或寫入。
    """

    def writeln(text: str = "") -> None:
        handle.write(f"{text}\n")

    reads = [
        item
        for item in list(getattr(runtime, "shared_memory_reads", []) or [])
        if isinstance(item, dict)
        and item.get("source") == "graph_memory"
        and (not task_id or str(item.get("task_id", "")) == str(task_id))
    ] if runtime is not None else []
    writes = list(getattr(runtime, "shared_memory_writes", []) or []) if runtime is not None else []
    graph_writes = [
        item
        for item in writes
        if isinstance(item, dict)
        and item.get("memory_type") == "graph_memory"
        and (not task_id or str(item.get("task_id", "")) == str(task_id))
    ]
    if not reads and not graph_writes:
        writeln("- graph_memory: (none)")
        return

    if reads:
        writeln("  reads:")
        for read in reads:
            seed_hits = read.get("seed_task_hits", []) or []
            expanded_hits = read.get("expanded_task_hits", []) or []
            writeln(
                "  - "
                f"stage={read.get('stage')} "
                f"task_type={read.get('task_type')} "
                f"related_task_ids={read.get('related_task_ids', [])} "
                f"insight_ids={read.get('insight_ids', [])} "
                f"qdrant_hit_count={len(seed_hits)} "
                f"expanded_hit_count={len(expanded_hits)}"
            )
            for hit in seed_hits[:5]:
                writeln(
                    "    "
                    f"qdrant_hit task_id={hit.get('task_id')} "
                    f"weight={hit.get('weight')} "
                    f"source={hit.get('source')} "
                    f"question={str(hit.get('question', '') or '')[:180]!r}"
                )
            for hit in expanded_hits[:5]:
                writeln(
                    "    "
                    f"expanded_hit task_id={hit.get('task_id')} "
                    f"weight={hit.get('weight')} "
                    f"source={hit.get('source')}"
                )

    if not graph_writes:
        writeln("  writes: (none)")
        return

    writeln("  writes:")
    for write in graph_writes[-2:]:
        writeln(
            "  - "
            f"task_id={write.get('task_id')} "
            f"source={write.get('source')} "
            f"label={write.get('label')} "
            f"task_type={write.get('task_type')} "
            f"related_task_ids={write.get('related_task_ids', [])} "
            f"insight_ids={write.get('insight_ids', [])}"
        )
        if write.get("candidate_updates"):
            writeln(f"    candidate_updates={write.get('candidate_updates')}")

        task_record = write.get("task_record") or {}
        state_chain = task_record.get("state_chain") or []
        writeln(f"    state_count={len(state_chain)}")
        for state_idx, state in enumerate(state_chain):
            graph = state.get("graph", {}) if isinstance(state, dict) else {}
            nodes = state.get("nodes", []) if isinstance(state, dict) else []
            edges = state.get("edges", []) if isinstance(state, dict) else []
            writeln(
                "    "
                f"[state {state_idx}] "
                f"id={graph.get('state_id')} "
                f"type={graph.get('state_type')} "
                f"stage={graph.get('stage')} "
                f"action={graph.get('action')!r} "
                f"observation={graph.get('observation')!r} "
                f"reward={graph.get('reward')}"
            )
            for node in nodes:
                extra = node.get("extra_fields", {}) or {}
                message = str(node.get("message", "") or "").strip().replace("\n", " ")
                if len(message) > 360:
                    message = message[:360] + "..."
                writeln(
                    "      "
                    f"node_id={node.get('id')} "
                    f"agent={node.get('agent_name')} "
                    f"node_type={extra.get('node_type')} "
                    f"answer={extra.get('final_answer', extra.get('answer', ''))!r} "
                    f"message={message!r}"
                )
                upstream_refs = extra.get("upstream_refs") or []
                if upstream_refs:
                    writeln(f"        upstream_refs={upstream_refs}")
            if edges:
                writeln(f"      edges={edges}")


def _write_token_usage_trace(handle: Any, runtime: Any) -> None:
    """
    負責將本題 LLM token usage 摘要寫入 compact log。

    Args:
        handle: 已開啟的文字檔 handle。
        runtime: AgentNetwork.runtime，用於取得 token_usage_summary()。

    Returns:
        無。

    限制或副作用:
        只讀取 token usage，不會修改 runtime 狀態。
    """

    def writeln(text: str = "") -> None:
        handle.write(f"{text}\n")

    if runtime is None:
        writeln("- token_usage: (none)")
        return
    summary = runtime.token_usage_summary()
    writeln(
        "- "
        f"calls={summary.get('calls', 0)} "
        f"prompt_tokens={summary.get('prompt_tokens', 0)} "
        f"completion_tokens={summary.get('completion_tokens', 0)} "
        f"total_tokens={summary.get('total_tokens', 0)}"
    )
    by_stage = summary.get("by_stage", {}) or {}
    if by_stage:
        writeln("  by_stage:")
        for stage, stats in sorted(by_stage.items()):
            writeln(
                "  - "
                f"{stage}: calls={stats.get('calls', 0)} "
                f"prompt={stats.get('prompt_tokens', 0)} "
                f"completion={stats.get('completion_tokens', 0)} "
                f"total={stats.get('total_tokens', 0)}"
            )
    by_model = summary.get("by_model", {}) or {}
    if by_model:
        writeln("  by_model:")
        for model, stats in sorted(by_model.items()):
            writeln(
                "  - "
                f"{model}: calls={stats.get('calls', 0)} "
                f"prompt={stats.get('prompt_tokens', 0)} "
                f"completion={stats.get('completion_tokens', 0)} "
                f"total={stats.get('total_tokens', 0)}"
            )


def write_compact_sample_log(handle: Any, sample_index: int, sample: dict[str, Any], network: Any, sample_result: dict[str, Any]) -> None:
    """
    負責將單題 GAIA 執行摘要寫入 compact log。

    Args:
        handle: 已開啟的 compact log handle。
        sample_index: 目前題目的 1-based 序號。
        sample: GAIA dataset 的單題資料。
        network: GAIA 評估使用的 AgentNetwork。
        sample_result: GAIA evaluator 產生的單題評估結果。

    Returns:
        無。

    限制或副作用:
        會直接寫入 handle；內容包含 stage1/stage2、attachment、search/RAG、GraphMemory 與 token usage trace。
    """

    def writeln(text: str = "") -> None:
        handle.write(f"{text}\n")

    decision = network.last_final_decision or {}
    activation_trace = getattr(network, "last_stage1_activation_trace", []) or []
    stage2_outputs = network.last_stage2_outputs or []
    runtime = getattr(network, "runtime", None)
    shared_search_bundle = getattr(runtime, "shared_stage2_search_bundle", None) if runtime is not None else None
    shared_attachment_bundle = getattr(runtime, "shared_attachment_bundle", None) if runtime is not None else None

    writeln(f"========= 第 {sample_index} 題 ==========")
    writeln(f"task_id: {sample.get('task_id', '')}")

    writeln("啟用節點:")
    if activation_trace:
        for entry in activation_trace:
            writeln(f"- round{entry.get('round')}: {entry.get('node_indices', [])}")
    else:
        writeln("- []")

    writeln("第一階段各節點狀態與回答:")
    if activation_trace:
        for entry in activation_trace:
            writeln(f"- round{entry.get('round')}:")
            for node in entry.get("nodes", []) or []:
                writeln(
                    "  "
                    f"node={node.get('idx')} "
                    f"model={node.get('model_name')} "
                    f"active={node.get('active', False)} "
                    f"answer={node.get('answer')!r}"
                )
    else:
        writeln("- (none)")

    writeln("第一輪注入的記憶:")
    first_round_nodes = activation_trace[0].get("nodes", []) if activation_trace else []
    if first_round_nodes:
        for node in first_round_nodes:
            writeln(f"- node={node.get('idx')} model={node.get('model_name')}")
            _write_indented_block(handle, node.get("stage1_reflection_context", ""))
    else:
        writeln("- (none)")

    writeln("附件證據:")
    if isinstance(shared_attachment_bundle, dict):
        metadata = shared_attachment_bundle.get("metadata", {}) or {}
        writeln(f"- shared_attachment_used: {bool(shared_attachment_bundle.get('used'))}")
        writeln(f"- file_path: {metadata.get('file_path', '')}")
        writeln(f"- file_type: {metadata.get('file_type', '')}")
        writeln(f"- reader: {metadata.get('reader', '')}")
        writeln("- shared_attachment_result:")
        _write_indented_block(handle, shared_attachment_bundle.get("attachment_context", ""))
    else:
        writeln("- shared_attachment: (none)")

    if activation_trace:
        for entry in activation_trace:
            writeln(f"- stage1_round{entry.get('round')}:")
            for node in entry.get("nodes", []) or []:
                context = node.get("stage1_attachment_context", "")
                if context:
                    writeln(f"  node={node.get('idx')} model={node.get('model_name')}")
                    _write_indented_block(handle, context, indent="    ")

    writeln("第二階段共用 Search/RAG 結果:")
    if isinstance(shared_search_bundle, dict):
        writeln(f"- shared_search_enabled: {bool(shared_search_bundle.get('enabled'))}")
        writeln(f"- shared_search_queries: {shared_search_bundle.get('queries', [])}")
        writeln("- shared_search_result:")
        _write_indented_block(handle, shared_search_bundle.get("search_context", ""))
    else:
        writeln("- shared_search: (none)")

    writeln("第二階段各節點工具使用與回答:")
    if stage2_outputs:
        for item in stage2_outputs:
            tool_names = [
                tool.get("tool_name")
                for tool in item.get("tool_usage", []) or []
                if isinstance(tool, dict) and tool.get("tool_name")
            ]
            writeln(
                "- "
                f"node={item.get('agent_idx')} "
                f"model={item.get('model_name')} "
                f"success={item.get('success', False)} "
                f"answer={item.get('answer')!r} "
                f"tools={tool_names} "
                f"routing={item.get('routing', {})}"
            )
            if item.get("attachment_context"):
                writeln("  attachment:")
                _write_indented_block(handle, item.get("attachment_context", ""), indent="    ")
            if item.get("search_context"):
                writeln("  search:")
                _write_indented_block(handle, item.get("search_context", ""), indent="    ")
            if item.get("solver_context"):
                writeln("  python_solver:")
                _write_indented_block(handle, item.get("solver_context", ""), indent="    ")
            if item.get("rag_context"):
                writeln("  rag:")
                _write_indented_block(handle, item.get("rag_context", ""), indent="    ")
            if (
                not item.get("attachment_context")
                and not item.get("search_context")
                and not item.get("solver_context")
                and not item.get("rag_context")
            ):
                writeln("  attachment/search/rag: (none)")
    else:
        writeln("- stage2: (none)")

    writeln("最終決策摘要:")
    writeln(f"- stage1_result: {network.last_stage1_result!r}")
    writeln(f"- final_result: {decision.get('final_result', network.last_stage1_result)!r}")
    writeln(f"- selected_agent_idx: {decision.get('selected_agent_idx')}")

    writeln("GAIA 評估結果:")
    writeln(f"- predicted: {sample_result.get('predicted', '')!r}")
    writeln(f"- exact_match: {sample_result.get('exact_match', False)}")
    writeln(f"- partial_match: {sample_result.get('partial_match', False)}")
    writeln(f"- score: {sample_result.get('score')}")

    writeln("Graph Memory trace:")
    _write_graph_memory_trace(handle, runtime, task_id=str(sample.get("task_id", "") or ""))

    writeln("Token usage:")
    _write_token_usage_trace(handle, runtime)
    writeln()
