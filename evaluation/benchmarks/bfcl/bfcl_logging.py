"""
BFCL 專用 logging 工具。

此模組先承接 BFCL runner 需要的 compact log、analysis report、
GraphMemory trace 與 token usage 記錄功能。之後若要建立共用的
benchmark_logger，可以把本檔中與 benchmark 無關的通用邏輯上移，
讓 BFCL 與 GAIA logger 都繼承或組合該共用基底。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.benchmark_logger import (
    BenchmarkLogger,
    open_utf8_text_log,
    restore_stdio,
    setup_utf8_log,
)


class BFCLBenchmarkLogger(BenchmarkLogger):
    """
    負責提供 BFCL benchmark 專用 logger 的共用繼承點。

    Args:
        無。

    Returns:
        BFCLBenchmarkLogger 實例，可使用 BenchmarkLogger 的共用 logging 方法。

    限制或副作用:
        目前 BFCL 的實際寫檔流程仍以模組函式為主；此類別保留給後續 runner 轉成物件式 logger。
    """
    pass


def stage2_predictions(network: Any) -> list[dict[str, Any]]:
    """
    負責從 AgentNetwork 取出 stage2 top-k Agent 的回答摘要。

    Args:
        network: BFCL 評估使用的 AgentNetwork。

    Returns:
        每個 stage2 Agent 的 agent_idx、success、answer、routing 與 tool_usage 列表。

    限制或副作用:
        只讀取 network.last_stage2_outputs，不會修改 network 狀態。
    """
    predictions: list[dict[str, Any]] = []
    for output in getattr(network, "last_stage2_outputs", []) or []:
        predictions.append(
            {
                "agent_idx": output.get("agent_idx"),
                "model_name": output.get("model_name"),
                "success": output.get("success"),
                "answer": output.get("answer"),
                "routing": output.get("routing", {}),
                "tool_usage": output.get("tool_usage", []),
                "error": output.get("error"),
            }
        )
    return predictions


def graph_memory_trace_summary(network: Any) -> dict[str, Any]:
    """
    負責整理本題 GraphMemory retrieval 與寫入 trace。

    Args:
        network: BFCL 評估使用的 AgentNetwork。

    Returns:
        包含 reads、writes、related_task_ids、insight_ids、qdrant_hits、expanded_hits 與 hit count 的字典。

    限制或副作用:
        只讀取 runtime.shared_memory_reads 與 runtime.shared_memory_writes，不會觸發新的記憶檢索或寫入。
    """
    return BenchmarkLogger.graph_memory_trace_summary(network)


def token_usage_summary(network: Any) -> dict[str, Any]:
    """
    負責從 runtime 取出本題 token usage 統計。

    Args:
        network: BFCL 評估使用的 AgentNetwork。

    Returns:
        包含 prompt_tokens、completion_tokens、total_tokens、calls、by_stage、by_model 與 records 的字典。

    限制或副作用:
        若 runtime 不存在，會回傳所有數值為 0 的預設結構。
    """
    return BenchmarkLogger.token_usage_summary(network)


def build_bfcl_analysis_record(
    *,
    sample_index: int,
    sample: dict[str, Any],
    sample_result: dict[str, Any],
    network: Any,
) -> dict[str, Any]:
    """
    負責建立單題 BFCL logging 與分析報告需要的結構化資料。

    Args:
        sample_index: 本題在此次 run 中的 1-based 序號。
        sample: BFCL dataset 載入的原始樣本。
        sample_result: BFCLEvaluator.evaluate_sample 回傳的單題結果。
        network: BFCL 評估使用的 AgentNetwork。

    Returns:
        包含題目、標準答案、預測答案、stage1/stage2/final 狀態、GraphMemory trace 與 token usage 的字典。

    限制或副作用:
        只彙整既有狀態，不會重新呼叫模型、工具或記憶系統。
    """
    decision = getattr(network, "last_final_decision", None) or {}
    memory = graph_memory_trace_summary(network)
    token_usage = token_usage_summary(network)
    return {
        "sample_index": sample_index,
        "sample_id": sample.get("id", ""),
        "category": sample_result.get("category", sample.get("category", "")),
        "question": sample.get("question", ""),
        "functions": sample.get("function", sample.get("functions", [])),
        "expected": sample_result.get("expected", sample.get("ground_truth", [])),
        "predicted": sample_result.get("predicted", []),
        "success": bool(sample_result.get("success", False)),
        "score": sample_result.get("score", 0.0),
        "response": sample_result.get("response", ""),
        "stage1_result": getattr(network, "last_stage1_result", None),
        "top_k_indices": list(getattr(network, "last_top_k_indices", []) or []),
        "stage2_predictions": stage2_predictions(network),
        "final_decision": decision,
        "memory": memory,
        "graph_memory_retrieval_hit": memory["retrieval_hit"],
        "graph_memory_related_task_ids": memory["related_task_ids"],
        "graph_memory_insight_ids": memory["insight_ids"],
        "graph_memory_qdrant_hit_count": memory["qdrant_hit_count"],
        "graph_memory_expanded_hit_count": memory["expanded_hit_count"],
        "token_usage": token_usage,
        "llm_call_count": token_usage.get("calls", 0),
        "prompt_tokens": token_usage.get("prompt_tokens", 0),
        "completion_tokens": token_usage.get("completion_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
        "error": sample_result.get("error"),
    }


def _write_json_line(handle: Any, label: str, value: Any, *, indent: str = "") -> None:
    """
    負責把任意 Python 物件以單行 JSON 形式寫入 log。

    Args:
        handle: 已開啟的文字檔 handle。
        label: 欄位名稱。
        value: 要輸出的資料。
        indent: 寫入行前方要保留的縮排。

    Returns:
        無；會寫入 handle。

    限制或副作用:
        若 value 無法序列化，會 fallback 到 repr 字串。
    """
    BenchmarkLogger.write_json_line(handle, label, value, indent=indent)


def write_bfcl_compact_sample_log(handle: Any, record: dict[str, Any]) -> None:
    """
    負責將單題 BFCL 執行摘要寫入 compact log。

    Args:
        handle: 已開啟的 UTF-8 文字檔 handle。
        record: build_bfcl_analysis_record 產生的單題分析資料。

    Returns:
        無；資料會寫入 handle 並 flush。

    限制或副作用:
        compact log 會包含 GraphMemory retrieval hit、related_task_ids、insight_ids、Qdrant hits 與 token usage。
    """
    def writeln(text: str = "") -> None:
        handle.write(text + "\n")

    memory = record["memory"]
    token_usage = record["token_usage"]

    writeln(f"========== BFCL sample {record['sample_index']} ==========")
    writeln(f"- id: {record['sample_id']}")
    writeln(f"- category: {record['category']}")
    writeln(f"- success: {record['success']} score={record['score']}")
    _write_json_line(handle, "expected", record["expected"])
    _write_json_line(handle, "predicted", record["predicted"])
    _write_json_line(handle, "stage1_result", record["stage1_result"])
    _write_json_line(handle, "top_k_indices", record["top_k_indices"])
    _write_json_line(handle, "final_decision", record["final_decision"])

    writeln("- graph_memory:")
    writeln(f"  retrieval_hit: {memory['retrieval_hit']}")
    _write_json_line(handle, "related_task_ids", memory["related_task_ids"], indent="  ")
    _write_json_line(handle, "insight_ids", memory["insight_ids"], indent="  ")
    writeln(f"  qdrant_hit_count: {memory['qdrant_hit_count']}")
    writeln(f"  expanded_hit_count: {memory['expanded_hit_count']}")
    _write_json_line(handle, "qdrant_hits", memory["qdrant_hits"], indent="  ")
    _write_json_line(handle, "expanded_hits", memory["expanded_hits"], indent="  ")

    writeln("- token_usage:")
    writeln(f"  calls: {token_usage.get('calls', 0)}")
    writeln(f"  prompt_tokens: {token_usage.get('prompt_tokens', 0)}")
    writeln(f"  completion_tokens: {token_usage.get('completion_tokens', 0)}")
    writeln(f"  total_tokens: {token_usage.get('total_tokens', 0)}")
    _write_json_line(handle, "by_stage", token_usage.get("by_stage", {}), indent="  ")
    _write_json_line(handle, "by_model", token_usage.get("by_model", {}), indent="  ")

    writeln("- stage2:")
    if record["stage2_predictions"]:
        for item in record["stage2_predictions"]:
            tool_names = [
                tool.get("tool_name")
                for tool in item.get("tool_usage", []) or []
                if isinstance(tool, dict) and tool.get("tool_name")
            ]
            writeln(
                "  - "
                f"agent_idx={item.get('agent_idx')} "
                f"model={item.get('model_name')} "
                f"success={item.get('success')} "
                f"answer={item.get('answer')!r} "
                f"tools={tool_names} "
                f"error={item.get('error')!r}"
            )
            _write_json_line(handle, "routing", item.get("routing", {}), indent="    ")
    else:
        writeln("  - (none)")

    if record.get("error"):
        writeln(f"- error: {record['error']}")
    writeln()
    handle.flush()


def write_bfcl_analysis_report(path: Path, records: list[dict[str, Any]], *, run_metadata: dict[str, Any]) -> Path:
    """
    負責輸出 BFCL 評估 Markdown 分析報告。

    Args:
        path: analysis report 的輸出路徑。
        records: 每題 build_bfcl_analysis_record 產生的分析資料。
        run_metadata: 本次評估的 category、memory namespace 與執行設定。

    Returns:
        寫入完成的 analysis report 路徑。

    限制或副作用:
        會建立父資料夾並覆寫 path 指向的檔案。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    correct = sum(1 for record in records if record["success"])
    total = len(records)
    prompt_tokens = sum(int(record.get("prompt_tokens", 0) or 0) for record in records)
    completion_tokens = sum(int(record.get("completion_tokens", 0) or 0) for record in records)
    total_tokens = sum(int(record.get("total_tokens", 0) or 0) for record in records)
    memory_hits = sum(1 for record in records if record.get("graph_memory_retrieval_hit"))
    qdrant_hits = sum(1 for record in records if int(record.get("graph_memory_qdrant_hit_count", 0) or 0) > 0)

    with path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write("# BFCL Evaluation Analysis\n\n")
        handle.write("## Run Metadata\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(run_metadata, ensure_ascii=False, indent=2, default=str))
        handle.write("\n```\n\n")

        handle.write("## Summary\n\n")
        handle.write(f"- total: {total}\n")
        handle.write(f"- correct: {correct}\n")
        handle.write(f"- accuracy: {(correct / total if total else 0.0):.2%}\n")
        handle.write(f"- graph_memory_retrieval_hits: {memory_hits}\n")
        handle.write(f"- graph_memory_qdrant_hits: {qdrant_hits}\n")
        handle.write(f"- prompt_tokens: {prompt_tokens}\n")
        handle.write(f"- completion_tokens: {completion_tokens}\n")
        handle.write(f"- total_tokens: {total_tokens}\n\n")

        handle.write("## Samples\n\n")
        for record in records:
            handle.write(f"### {record['sample_index']}. {record['sample_id']}\n\n")
            handle.write(f"- category: {record['category']}\n")
            handle.write(f"- success: {record['success']} score={record['score']}\n")
            handle.write(f"- expected: `{record['expected']!r}`\n")
            handle.write(f"- predicted: `{record['predicted']!r}`\n")
            handle.write(f"- graph_memory_retrieval_hit: `{record['graph_memory_retrieval_hit']}`\n")
            handle.write(f"- related_task_ids: `{record['graph_memory_related_task_ids']}`\n")
            handle.write(f"- insight_ids: `{record['graph_memory_insight_ids']}`\n")
            handle.write(f"- qdrant_hit_count: `{record['graph_memory_qdrant_hit_count']}`\n")
            handle.write(f"- expanded_hit_count: `{record['graph_memory_expanded_hit_count']}`\n")
            handle.write(f"- token_total: `{record['total_tokens']}`\n\n")

        handle.write("## GraphMemory Read Details\n\n")
        for record in records:
            handle.write(f"### {record['sample_index']}. {record['sample_id']}\n\n")
            reads = record["memory"].get("reads", []) or []
            if not reads:
                handle.write("- reads: (none)\n\n")
                continue
            handle.write("```json\n")
            handle.write(json.dumps(reads, ensure_ascii=False, indent=2, default=str))
            handle.write("\n```\n\n")

        handle.write("## Raw Records\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(records, ensure_ascii=False, indent=2, default=str))
        handle.write("\n```\n")
    return path
