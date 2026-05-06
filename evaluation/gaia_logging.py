from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class TeeStream:
    """Mirror stdout/stderr to a UTF-8 log file without relying on shell redirection."""

    def __init__(self, primary, mirror):
        self.primary = primary
        self.mirror = mirror

    @property
    def encoding(self):
        return getattr(self.primary, "encoding", "utf-8")

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        self.primary.write(data)
        self.mirror.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.mirror.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()

    def fileno(self):
        return self.primary.fileno()

    def __getattr__(self, name):
        return getattr(self.primary, name)


def setup_utf8_log(log_file_path: Path):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file_path.open("w", encoding="utf-8-sig", buffering=1, newline="\n")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    return log_handle, original_stdout, original_stderr


def restore_stdio(*, original_stdout, original_stderr) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr


def open_utf8_text_log(log_file_path: Path):
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    return log_file_path.open("w", encoding="utf-8-sig", buffering=1, newline="\n")


def print_stage2_and_final(network: Any) -> None:
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


def _write_indented_block(handle, text, indent="    "):
    content = str(text or "").strip()
    if not content:
        handle.write(f"{indent}(none)\n")
        return

    for line in content.splitlines():
        stripped = line.rstrip()
        if stripped:
            handle.write(f"{indent}{stripped}\n")
        else:
            handle.write(f"{indent}\n")


def write_compact_sample_log(handle, sample_index: int, sample: dict[str, Any], network: Any, sample_result: dict[str, Any]):
    def writeln(text=""):
        handle.write(f"{text}\n")

    decision = network.last_final_decision or {}
    activation_trace = getattr(network, "last_stage1_activation_trace", []) or []
    stage2_outputs = network.last_stage2_outputs or []
    runtime = getattr(network, "runtime", None)
    shared_search_bundle = getattr(runtime, "shared_stage2_search_bundle", None) if runtime is not None else None
    shared_attachment_bundle = getattr(runtime, "shared_attachment_bundle", None) if runtime is not None else None

    writeln(f"========= 第 {sample_index} 題 ============")
    writeln(f"task_id: {sample.get('task_id', '')}")

    writeln("Activate Node :")
    if activation_trace:
        for entry in activation_trace:
            writeln(f"- round{entry.get('round')}: {entry.get('node_indices', [])}")
    else:
        writeln("- []")

    writeln("各別使用的 model、各別的回答:")
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

    writeln("第一輪使用的 memory:")
    first_round_nodes = activation_trace[0].get("nodes", []) if activation_trace else []
    if first_round_nodes:
        for node in first_round_nodes:
            writeln(f"- node={node.get('idx')} model={node.get('model_name')}")
            _write_indented_block(handle, node.get("stage1_reflection_context", ""))
    else:
        writeln("- (none)")

    writeln("Attachment evidence:")
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

    writeln("Stage2 使用的 Search 或 RAG 結果:")
    if isinstance(shared_search_bundle, dict):
        writeln(f"- shared_search_enabled: {bool(shared_search_bundle.get('enabled'))}")
        writeln(f"- shared_search_queries: {shared_search_bundle.get('queries', [])}")
        writeln("- shared_search_result:")
        _write_indented_block(handle, shared_search_bundle.get("search_context", ""))
    else:
        writeln("- shared_search: (none)")

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

    writeln("系統最後的回答:")
    writeln(f"- stage1_result: {network.last_stage1_result!r}")
    writeln(f"- final_result: {decision.get('final_result', network.last_stage1_result)!r}")
    writeln(f"- selected_agent_idx: {decision.get('selected_agent_idx')}")

    writeln("GAIA 最後的匹配結果:")
    writeln(f"- predicted: {sample_result.get('predicted', '')!r}")
    writeln(f"- exact_match: {sample_result.get('exact_match', False)}")
    writeln(f"- partial_match: {sample_result.get('partial_match', False)}")
    writeln(f"- score: {sample_result.get('score')}")
    writeln()
