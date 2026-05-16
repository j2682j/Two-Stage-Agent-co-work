from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.benchmarks.bfcl.bfcl_logging import (
    build_bfcl_analysis_record,
    open_utf8_text_log,
    restore_stdio,
    setup_utf8_log,
    write_bfcl_analysis_report,
    write_bfcl_compact_sample_log,
)
from evaluation.benchmarks.bfcl.dataset import BFCLDataset
from evaluation.benchmarks.bfcl.evaluator import BFCLEvaluator
from evaluation.benchmarks.bfcl_adapter import BFCLAdapter
from network.core.agent_network import AgentNetwork
from utils.project_paths import PROJECT_ROOT, ensure_runtime_dirs, get_eval_output_path, get_log_file_path


DEFAULT_BFCL_DATA_DIR = PROJECT_ROOT / "temp_gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"


def create_bfcl_network(args: Any, *, memory_namespace: str) -> AgentNetwork:
    """建立 create_bfcl_network 所需的物件。
    
    參數:
        args: 此流程需要使用的輸入資料。
        memory_namespace: 此流程需要使用的輸入資料。
    
    回傳:
        此函式的處理結果。
    """
    return AgentNetwork(
        agents=int(getattr(args, "agents", 3) or 3),
        rounds=int(getattr(args, "rounds", 3) or 3),
        seed=int(getattr(args, "seed", 2026) or 2026),
        enable_stage1_tools=False,
        enable_shared_memory=False,
        shared_memory_user_id=memory_namespace,
        memory_mode=str(getattr(args, "memory_mode", "stage1_first_round_only") or "stage1_first_round_only"),
    )


def clear_runtime_sample_state(network: AgentNetwork) -> None:
    """清除 clear_runtime_sample_state 相關狀態。
    
    參數:
        network: 此流程需要使用的輸入資料。
    
    回傳:
        此函式的處理結果。
    """
    runtime = getattr(network, "runtime", None)
    if runtime is None:
        return
    if hasattr(runtime, "clear_observability_records"):
        runtime.clear_observability_records()
        return
    runtime.shared_tool_traces.clear()
    runtime.shared_memory_reads.clear()
    runtime.shared_memory_writes.clear()
    runtime.shared_token_usage.clear()


def resolve_bfcl_paths(args: Any) -> dict[str, Path]:
    """處理 resolve_bfcl_paths 流程並回傳結果。
    
    參數:
        args: 此流程需要使用的輸入資料。
    
    回傳:
        此函式的處理結果。
    """
    base_name = str(getattr(args, "log_name", "test_bfcl") or "test_bfcl").strip()
    category = str(getattr(args, "category", "simple_python") or "simple_python").strip()

    log_file = getattr(args, "log_file", None) or get_log_file_path(f"{base_name}.log")
    compact_log_file = getattr(args, "compact_log_file", None) or get_log_file_path(f"{base_name}_compact.log")
    analysis_file = getattr(args, "analysis_file", None) or get_log_file_path(f"{base_name}_analysis.md")
    result_file = getattr(args, "result_file", None) or get_eval_output_path(f"{base_name}_{category}_results.json")

    return {
        "log_file": Path(log_file).resolve(),
        "compact_log_file": Path(compact_log_file).resolve(),
        "analysis_file": Path(analysis_file).resolve(),
        "result_file": Path(result_file).resolve(),
    }


def build_results(
    *,
    agent: BFCLAdapter,
    evaluator: BFCLEvaluator,
    detailed_results: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    """建立 build_results 所需的資料或輸出。
    
    參數:
        agent: 此流程需要使用的輸入資料。
        evaluator: 此流程需要使用的輸入資料。
        detailed_results: 此流程需要使用的輸入資料。
        category: 此流程需要使用的輸入資料。
    
    回傳:
        此函式的處理結果。
    """
    total_samples = len(detailed_results)
    correct_samples = sum(1 for result in detailed_results if result.get("success", False))
    return {
        "benchmark": "BFCL",
        "agent_name": getattr(agent, "name", "AgentNetwork"),
        "evaluation_mode": evaluator.evaluation_mode,
        "category": category,
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "overall_accuracy": correct_samples / total_samples if total_samples else 0.0,
        "detailed_results": detailed_results,
    }


def _ensure_ollama_timeout(min_timeout: int = 180) -> None:
    """處理 ensure_ollama_timeout 流程並回傳結果。
    
    參數:
        min_timeout: 此流程需要使用的輸入資料。
    
    回傳:
        此函式的處理結果。
    """
    current_timeout = int(os.getenv("OLLAMA_TIMEOUT", "60") or "60")
    if current_timeout < min_timeout:
        os.environ["OLLAMA_TIMEOUT"] = str(min_timeout)


def run_bfcl_evaluation(args: Any) -> dict[str, Any]:
    """執行 run_bfcl_evaluation 流程並回傳結果。
    
    參數:
        args: 此流程需要使用的輸入資料。
    
    回傳:
        此函式的處理結果。
    """
    ensure_runtime_dirs()
    paths = resolve_bfcl_paths(args)
    log_handle, original_stdout, original_stderr = setup_utf8_log(paths["log_file"])
    compact_log_handle = None

    try:
        compact_log_handle = open_utf8_text_log(paths["compact_log_file"])
        category = str(getattr(args, "category", "simple_python") or "simple_python")
        max_samples = int(getattr(args, "max_samples", 5) or 0)
        evaluation_mode = str(getattr(args, "evaluation_mode", "ast") or "ast")
        bfcl_data_dir = Path(getattr(args, "bfcl_data_dir", None) or DEFAULT_BFCL_DATA_DIR)

        _ensure_ollama_timeout(180)

        print(f"[INFO] BFCL full log: {paths['log_file']}")
        print(f"[INFO] BFCL compact log: {paths['compact_log_file']}")
        print(f"[INFO] BFCL analysis file: {paths['analysis_file']}")
        print(f"[INFO] BFCL result file: {paths['result_file']}")
        print(f"[INFO] BFCL data dir: {bfcl_data_dir}")
        print(f"[INFO] category={category} max_samples={max_samples} evaluation_mode={evaluation_mode}")
        print(f"[INFO] OLLAMA_TIMEOUT={os.getenv('OLLAMA_TIMEOUT')}")

        dataset = BFCLDataset(bfcl_data_dir=bfcl_data_dir, category=category)
        samples = dataset.load()
        if max_samples > 0:
            samples = samples[:max_samples]
        print(f"[INFO] Loaded {len(samples)} BFCL samples.")

        evaluator = BFCLEvaluator(dataset=dataset, category=category, evaluation_mode=evaluation_mode)
        memory_namespace = f"bfcl_graph_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        network = create_bfcl_network(args, memory_namespace=memory_namespace)
        agent = BFCLAdapter(network, use_two_stage=True, include_reasoning=False)

        detailed_results: list[dict[str, Any]] = []
        analysis_records: list[dict[str, Any]] = []

        for sample_index, sample in enumerate(samples, 1):
            print(f"\n========== BFCL sample {sample_index}/{len(samples)} ==========")
            print(f"[INFO] sample_id={sample.get('id', '')}")
            clear_runtime_sample_state(network)

            sample_result = evaluator.evaluate_sample(agent, sample)
            detailed_results.append(sample_result)

            record = build_bfcl_analysis_record(
                sample_index=sample_index,
                sample=sample,
                sample_result=sample_result,
                network=network,
            )
            analysis_records.append(record)
            write_bfcl_compact_sample_log(compact_log_handle, record)

            print(
                f"success={sample_result.get('success', False)} "
                f"score={sample_result.get('score', 0.0)} "
                f"predicted={sample_result.get('predicted', [])!r}"
            )

        results = build_results(
            agent=agent,
            evaluator=evaluator,
            detailed_results=detailed_results,
            category=category,
        )

        paths["result_file"].parent.mkdir(parents=True, exist_ok=True)
        with paths["result_file"].open("w", encoding="utf-8-sig", newline="\n") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

        analysis_path = write_bfcl_analysis_report(
            paths["analysis_file"],
            analysis_records,
            run_metadata={
                "run_started_at": memory_namespace.replace("bfcl_graph_memory_", ""),
                "category": category,
                "max_samples": max_samples,
                "evaluation_mode": evaluation_mode,
                "memory_namespace": memory_namespace,
                "graph_memory_enabled": getattr(network.runtime, "graph_memory", None) is not None,
            },
        )

        print("\n[OK] BFCL evaluation finished")
        print(f"   accuracy={results['overall_accuracy']:.2%}")
        print(f"   correct={results['correct_samples']}/{results['total_samples']}")
        print(f"   result_file={paths['result_file']}")
        print(f"   analysis_file={analysis_path}")
        return results
    finally:
        restore_stdio(original_stdout=original_stdout, original_stderr=original_stderr)
        if compact_log_handle is not None:
            compact_log_handle.close()
        log_handle.close()
