import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from evaluation.benchmarks.gaia import GAIADataset, GAIAEvaluator
from evaluation.gaia_adapter import GAIAAdapter
from network.agent_network import AgentNetwork
from utils.project_paths import ensure_runtime_dirs, get_eval_output_path, get_log_file_path


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small GAIA memory smoke test.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument(
        "--log-file",
        default=str(get_log_file_path("test_gaia_latest.log")),
        help="UTF-8 log file path written directly by Python.",
    )
    parser.add_argument(
        "--compact-log-file",
        default=str(get_log_file_path("test_gaia_compact.log")),
        help="Compact UTF-8 summary log file path written directly by Python.",
    )
    return parser.parse_args()


def setup_utf8_log(log_file_path: Path):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    # Use UTF-8 with BOM so Windows PowerShell reads the log correctly via Get-Content.
    log_handle = log_file_path.open("w", encoding="utf-8-sig", buffering=1, newline="\n")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    return log_handle, original_stdout, original_stderr


def open_utf8_text_log(log_file_path: Path):
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    return log_file_path.open("w", encoding="utf-8-sig", buffering=1, newline="\n")


class MemoryUsageTracer:
    """Trace whether a new sample retrieves memories written by previous samples."""

    def __init__(self, memory_tool):
        self.memory_tool = memory_tool
        self.memory_manager = getattr(memory_tool, "memory_manager", None)
        self.original_retrieve = None
        self.current_sample = None
        self.previous_markers = []
        self.used_previous_memory = False
        self.hit_records = []

    def install(self):
        if self.memory_manager is None or self.original_retrieve is not None:
            return

        self.original_retrieve = self.memory_manager.retrieve_memories

        def traced_retrieve(*args, **kwargs):
            results = self.original_retrieve(*args, **kwargs)
            if self.current_sample and self.previous_markers:
                for memory in results or []:
                    content = getattr(memory, "content", "") or ""
                    for marker in self.previous_markers:
                        if marker and marker in content:
                            self.used_previous_memory = True
                            self.hit_records.append(
                                {
                                    "query": kwargs.get("query", ""),
                                    "marker": marker,
                                    "memory_type": getattr(memory, "memory_type", ""),
                                    "preview": content[:160],
                                }
                            )
                            break
            return results

        self.memory_manager.retrieve_memories = traced_retrieve

    def start_sample(self, sample):
        self.current_sample = sample
        self.used_previous_memory = False
        self.hit_records = []

    def finish_sample(self):
        self.current_sample = None

    def register_feedback_marker(self, sample, normalized_question):
        task_id = sample.get("task_id", "")
        question = sample.get("question", "")
        markers = [task_id, normalized_question]
        if question:
            markers.append(question[:80])
        for marker in markers:
            if marker and marker not in self.previous_markers:
                self.previous_markers.append(marker)


def print_memory_stats(memory_tool):
    memory_manager = getattr(memory_tool, "memory_manager", None)
    if memory_manager is None:
        print("   [MEMORY][stats] unavailable")
        return

    stats = memory_manager.get_memory_stats()
    by_type = stats.get("memories_by_type", {}) or {}
    summary = {
        memory_type: details.get("count", 0)
        for memory_type, details in by_type.items()
    }
    print(f"   [MEMORY][stats] total={stats.get('total_memories', 0)} by_type={summary}")


def print_stage2_and_final(network):
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


def _compact_single_line(value, default="(none)"):
    text = str(value or "").strip()
    if not text:
        return default
    return " ".join(text.split())


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


def write_compact_sample_log(handle, sample_index, sample, network, sample_result):
    def writeln(text=""):
        handle.write(f"{text}\n")

    decision = network.last_final_decision or {}
    activation_trace = getattr(network, "last_stage1_activation_trace", []) or []
    stage2_outputs = network.last_stage2_outputs or []
    runtime = getattr(network, "runtime", None)
    shared_search_bundle = getattr(runtime, "shared_stage2_search_bundle", None) if runtime is not None else None

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

    writeln("第一輪使用的memory:")
    first_round_nodes = activation_trace[0].get("nodes", []) if activation_trace else []
    if first_round_nodes:
        for node in first_round_nodes:
            writeln(f"- node={node.get('idx')} model={node.get('model_name')}")
            _write_indented_block(handle, node.get("stage1_reflection_context", ""))
    else:
        writeln("- (none)")

    writeln("Stage2 使用的Search或RAG結果:")
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
                f"tools={tool_names}"
            )
            if item.get("search_context"):
                writeln("  search:")
                _write_indented_block(handle, item.get("search_context", ""), indent="    ")
            if item.get("rag_context"):
                writeln("  rag:")
                _write_indented_block(handle, item.get("rag_context", ""), indent="    ")
            if not item.get("search_context") and not item.get("rag_context"):
                writeln("  search/rag: (none)")
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


def main():
    args = parse_args()
    ensure_runtime_dirs()
    log_file_path = Path(args.log_file).resolve()
    compact_log_file_path = Path(args.compact_log_file).resolve()
    log_handle, original_stdout, original_stderr = setup_utf8_log(log_file_path)
    compact_log_handle = None

    try:
        compact_log_handle = open_utf8_text_log(compact_log_file_path)
        print(f"[INFO] UTF-8 log file: {log_file_path}")
        print(f"[INFO] compact UTF-8 log file: {compact_log_file_path}")

        current_timeout = int(os.getenv("OLLAMA_TIMEOUT", "60") or "60")
        if current_timeout < 180:
            os.environ["OLLAMA_TIMEOUT"] = "180"

        local_dir = ROOT / "test" / "data" / "gaia"

        dataset = GAIADataset(local_data_dir=local_dir, level=args.level)
        items = dataset.load()
        print(f"Loaded {len(items)} GAIA level {args.level} samples.")

        evaluator = GAIAEvaluator(dataset=dataset, level=args.level)
        memory_namespace = f"gaia_stage1_reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        network = AgentNetwork(
            agents=3,
            rounds=3,
            seed=2026,
            shared_memory_user_id=memory_namespace,
            memory_mode="stage1_first_round_only",
        )
        agent = GAIAAdapter(network, use_two_stage=True, include_reasoning=False)

        tracer = MemoryUsageTracer(network.memory_tool)
        tracer.install()

        max_samples = max(0, args.max_samples)
        samples = items[:max_samples]
        print(f"[INFO] memory_namespace={memory_namespace}")
        print(f"[INFO] OLLAMA_TIMEOUT={os.getenv('OLLAMA_TIMEOUT')}")
        print(f"[INFO] Running {len(samples)} samples for stage1 reflection memory test")

        detailed_results = []
        level_stats = {args.level: {"total": 0, "correct": 0, "partial": 0}}

        for i, sample in enumerate(samples, 1):
            print(f"\n========== 第 {i} 題 / 共 {len(samples)} 題 ==========")
            print(f"[INFO] task_id={sample.get('task_id', '')}")
            if getattr(network, "runtime", None) is not None:
                network.runtime.shared_tool_traces.clear()
                network.runtime.shared_memory_reads.clear()
                network.runtime.shared_memory_writes.clear()

            tracer.start_sample(sample)
            sample_result = evaluator.evaluate_sample(agent, sample)
            tracer.finish_sample()
            detailed_results.append(sample_result)

            normalized_question = agent.normalize_question(sample.get("question", ""))

            try:
                agent.record_evaluation_feedback(
                    benchmark="GAIA",
                    sample=sample,
                    sample_result=sample_result,
                )
            except Exception as feedback_error:
                print(f"[WARN] sample {i} feedback write failed: {feedback_error}")

            tracer.register_feedback_marker(sample, normalized_question)
            print_memory_stats(network.memory_tool)

            try:
                for memory_type in ["working", "semantic", "episodic"]:
                    memory_debug = network.memory_tool.run(
                        {
                            "action": "search",
                            "query": normalized_question,
                            "memory_type": memory_type,
                            "limit": 5,
                        }
                    )
                    print(f"   [MEMORY][{memory_type}]")
                    print(f"   {memory_debug}")
            except Exception as memory_debug_error:
                print(f"[WARN] sample {i} memory debug failed: {memory_debug_error}")

            level = sample.get("level", args.level)
            if level not in level_stats:
                level_stats[level] = {"total": 0, "correct": 0, "partial": 0}
            level_stats[level]["total"] += 1
            if sample_result.get("exact_match", False):
                level_stats[level]["correct"] += 1
            if sample_result.get("partial_match", False):
                level_stats[level]["partial"] += 1

            print(
                f"exact_match={sample_result.get('exact_match', False)} "
                f"partial_match={sample_result.get('partial_match', False)} "
                f"predicted={sample_result.get('predicted', '')!r}"
            )
            print_stage2_and_final(network)
            if compact_log_handle is not None:
                write_compact_sample_log(
                    compact_log_handle,
                    sample_index=i,
                    sample=sample,
                    network=network,
                    sample_result=sample_result,
                )
            print(f"   used_previous_memory={tracer.used_previous_memory}")
            if tracer.hit_records:
                for hit in tracer.hit_records[:5]:
                    print(
                        "   [MEMORY-HIT] "
                        f"type={hit['memory_type']} "
                        f"marker={hit['marker']!r} "
                        f"query={hit['query']!r}"
                    )
                    print(f"      preview={hit['preview']}")

        total_samples = len(detailed_results)
        exact_matches = sum(1 for r in detailed_results if r.get("exact_match", False))
        partial_matches = sum(1 for r in detailed_results if r.get("partial_match", False))

        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0.0
        partial_match_rate = partial_matches / total_samples if total_samples > 0 else 0.0

        level_metrics = {}
        for level, stats in level_stats.items():
            if stats["total"] > 0:
                level_metrics[f"Level_{level}"] = {
                    "total": stats["total"],
                    "exact_matches": stats["correct"],
                    "partial_matches": stats["partial"],
                    "exact_match_rate": stats["correct"] / stats["total"],
                    "partial_match_rate": stats["partial"] / stats["total"],
                }

        results = {
            "benchmark": "GAIA",
            "agent_name": getattr(agent, "name", "Unknown"),
            "strict_mode": evaluator.strict_mode,
            "level_filter": evaluator.level,
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "level_metrics": level_metrics,
            "detailed_results": detailed_results,
        }

        print("\n[OK] GAIA test finished")
        print(f"   exact_match_rate={exact_match_rate:.2%}")
        print(f"   partial_match_rate={partial_match_rate:.2%}")

        evaluator.export_to_gaia_format(
            results,
            get_eval_output_path("gaia_evaluation_results.json"),
            include_reasoning=True,
        )
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if compact_log_handle is not None:
            compact_log_handle.close()
        log_handle.close()


if __name__ == "__main__":
    main()
