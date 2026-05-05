from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.benchmarks.gaia import GAIADataset, GAIAEvaluator
from evaluation.gaia_adapter import GAIAAdapter
from evaluation.gaia_analysis_report import build_sample_analysis_record, write_gaia_analysis_report
from evaluation.gaia_logging import (
    open_utf8_text_log,
    print_stage2_and_final,
    restore_stdio,
    setup_utf8_log,
    write_compact_sample_log,
)
from evaluation.gaia_memory_trace import (
    MemoryUsageTracer,
    print_memory_debug,
    print_memory_hits,
    print_memory_stats,
)
from network.agent_network import AgentNetwork
from utils.project_paths import ensure_runtime_dirs, get_eval_output_path


ROOT = Path(__file__).resolve().parents[1]


def create_gaia_network(args, *, memory_namespace: str) -> AgentNetwork:
    return AgentNetwork(
        agents=3,
        rounds=3,
        seed=2026,
        enable_shared_memory=True,
        shared_memory_user_id=memory_namespace,
        memory_mode="stage1_first_round_only",
        enable_stage1_attachment_after_first_round=args.enable_stage1_attachment_after_first_round,
    )


def update_level_stats(level_stats: dict[Any, dict[str, int]], level: Any, sample_result: dict[str, Any]) -> None:
    if level not in level_stats:
        level_stats[level] = {"total": 0, "correct": 0, "partial": 0}
    level_stats[level]["total"] += 1
    if sample_result.get("exact_match", False):
        level_stats[level]["correct"] += 1
    if sample_result.get("partial_match", False):
        level_stats[level]["partial"] += 1


def build_level_metrics(level_stats: dict[Any, dict[str, int]]) -> dict[str, Any]:
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
    return level_metrics


def build_results(
    *,
    agent,
    evaluator,
    detailed_results: list[dict[str, Any]],
    level_metrics: dict[str, Any],
) -> dict[str, Any]:
    total_samples = len(detailed_results)
    exact_matches = sum(1 for result in detailed_results if result.get("exact_match", False))
    partial_matches = sum(1 for result in detailed_results if result.get("partial_match", False))

    return {
        "benchmark": "GAIA",
        "agent_name": getattr(agent, "name", "Unknown"),
        "strict_mode": evaluator.strict_mode,
        "level_filter": evaluator.level,
        "total_samples": total_samples,
        "exact_matches": exact_matches,
        "partial_matches": partial_matches,
        "exact_match_rate": exact_matches / total_samples if total_samples > 0 else 0.0,
        "partial_match_rate": partial_matches / total_samples if total_samples > 0 else 0.0,
        "level_metrics": level_metrics,
        "detailed_results": detailed_results,
    }


def clear_runtime_sample_state(network: AgentNetwork) -> None:
    runtime = getattr(network, "runtime", None)
    if runtime is None:
        return
    runtime.shared_tool_traces.clear()
    runtime.shared_memory_reads.clear()
    runtime.shared_memory_writes.clear()


def run_gaia_evaluation(args) -> dict[str, Any]:
    ensure_runtime_dirs()
    log_file_path = Path(args.log_file).resolve()
    compact_log_file_path = Path(args.compact_log_file).resolve()
    analysis_file_path = Path(args.analysis_file).resolve()
    log_handle, original_stdout, original_stderr = setup_utf8_log(log_file_path)
    compact_log_handle = None

    try:
        compact_log_handle = open_utf8_text_log(compact_log_file_path)
        print(f"[INFO] UTF-8 log file: {log_file_path}")
        print(f"[INFO] compact UTF-8 log file: {compact_log_file_path}")
        print(f"[INFO] analysis file: {analysis_file_path}")

        current_timeout = int(os.getenv("OLLAMA_TIMEOUT", "60") or "60")
        if current_timeout < 180:
            os.environ["OLLAMA_TIMEOUT"] = "180"

        local_dir = ROOT / "test" / "data" / "gaia"
        dataset = GAIADataset(local_data_dir=local_dir, level=args.level)
        items = dataset.load()
        print(f"Loaded {len(items)} GAIA level {args.level} samples.")

        evaluator = GAIAEvaluator(dataset=dataset, level=args.level)
        memory_namespace = f"gaia_stage1_reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        network = create_gaia_network(args, memory_namespace=memory_namespace)
        agent = GAIAAdapter(network, use_two_stage=True, include_reasoning=False)

        memory_enabled = network.memory_tool is not None
        tracer = MemoryUsageTracer(network.memory_tool) if memory_enabled else None
        if tracer is not None:
            tracer.install()

        max_samples = max(0, args.max_samples)
        samples = items[:max_samples]
        print(f"[INFO] memory_namespace={memory_namespace}")
        print(f"[INFO] memory_enabled={memory_enabled}")
        print(
            "[INFO] enable_stage1_attachment_after_first_round="
            f"{args.enable_stage1_attachment_after_first_round}"
        )
        print(f"[INFO] OLLAMA_TIMEOUT={os.getenv('OLLAMA_TIMEOUT')}")
        print(f"[INFO] Running {len(samples)} samples for GAIA evaluation")

        detailed_results: list[dict[str, Any]] = []
        analysis_records: list[dict[str, Any]] = []
        level_stats = {args.level: {"total": 0, "correct": 0, "partial": 0}}

        for sample_index, sample in enumerate(samples, 1):
            print(f"\n========== 第 {sample_index} 題 / 共 {len(samples)} 題 ==========")
            print(f"[INFO] task_id={sample.get('task_id', '')}")
            clear_runtime_sample_state(network)

            if tracer is not None:
                tracer.start_sample(sample)
            sample_result = evaluator.evaluate_sample(agent, sample)
            if tracer is not None:
                tracer.finish_sample()

            detailed_results.append(sample_result)
            analysis_records.append(
                build_sample_analysis_record(
                    sample_index=sample_index,
                    sample=sample,
                    network=network,
                    sample_result=sample_result,
                    evaluator=evaluator,
                    memory_used=tracer.used_previous_memory if tracer is not None else False,
                    memory_hits=list(tracer.hit_records) if tracer is not None else [],
                )
            )

            normalized_question = agent.normalize_question(sample.get("question", ""))

            if memory_enabled:
                try:
                    agent.record_evaluation_feedback(
                        benchmark="GAIA",
                        sample=sample,
                        sample_result=sample_result,
                    )
                except Exception as feedback_error:
                    print(f"[WARN] sample {sample_index} feedback write failed: {feedback_error}")

                if tracer is not None:
                    tracer.register_feedback_marker(sample, normalized_question)
                print_memory_stats(network.memory_tool)

                try:
                    print_memory_debug(network.memory_tool, normalized_question)
                except Exception as memory_debug_error:
                    print(f"[WARN] sample {sample_index} memory debug failed: {memory_debug_error}")

            update_level_stats(level_stats, sample.get("level", args.level), sample_result)

            print(
                f"exact_match={sample_result.get('exact_match', False)} "
                f"partial_match={sample_result.get('partial_match', False)} "
                f"predicted={sample_result.get('predicted', '')!r}"
            )
            print_stage2_and_final(network)
            if compact_log_handle is not None:
                write_compact_sample_log(
                    compact_log_handle,
                    sample_index=sample_index,
                    sample=sample,
                    network=network,
                    sample_result=sample_result,
                )
            print_memory_hits(tracer)

        results = build_results(
            agent=agent,
            evaluator=evaluator,
            detailed_results=detailed_results,
            level_metrics=build_level_metrics(level_stats),
        )

        print("\n[OK] GAIA test finished")
        print(f"   exact_match_rate={results['exact_match_rate']:.2%}")
        print(f"   partial_match_rate={results['partial_match_rate']:.2%}")

        evaluator.export_to_gaia_format(
            results,
            get_eval_output_path("gaia_evaluation_results_enable_memory_with_attchment.json"),
            include_reasoning=True,
        )

        analysis_output_path = write_gaia_analysis_report(
            analysis_file_path,
            analysis_records,
            run_metadata={
                "run_started_at": memory_namespace.replace("gaia_stage1_reflection_", ""),
                "level": args.level,
                "memory_mode": network.memory_mode,
                "memory_enabled": memory_enabled,
            },
        )
        print(f"[OK] GAIA analysis report written: {analysis_output_path}")
        return results
    finally:
        restore_stdio(original_stdout=original_stdout, original_stderr=original_stderr)
        if compact_log_handle is not None:
            compact_log_handle.close()
        log_handle.close()
