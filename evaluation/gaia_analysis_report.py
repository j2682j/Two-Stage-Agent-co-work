from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def clean_answer(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def attachment_type(sample: dict[str, Any]) -> str:
    file_path = str(sample.get("file_name") or sample.get("file_path") or "").strip()
    if not file_path:
        return "-"
    suffix = Path(file_path).suffix.lower()
    return suffix or "(unknown)"


def stage2_tool_names(stage2_outputs: list[dict[str, Any]]) -> list[str]:
    names = set()
    for output in stage2_outputs or []:
        for tool in output.get("tool_usage", []) or []:
            if isinstance(tool, dict) and tool.get("tool_name"):
                names.add(str(tool["tool_name"]))
    return sorted(names)


def build_sample_analysis_record(
    *,
    sample_index: int,
    sample: dict[str, Any],
    network: Any,
    sample_result: dict[str, Any],
    evaluator: Any,
    memory_used: bool,
    memory_hits: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = clean_answer(sample.get("final_answer", ""))
    stage1_result = clean_answer(getattr(network, "last_stage1_result", ""))
    final_decision = getattr(network, "last_final_decision", None) or {}
    final_result = clean_answer(final_decision.get("final_result", stage1_result))
    predicted = clean_answer(sample_result.get("predicted", ""))
    stage2_outputs = getattr(network, "last_stage2_outputs", []) or []
    runtime = getattr(network, "runtime", None)
    shared_attachment_bundle = getattr(runtime, "shared_attachment_bundle", None) if runtime is not None else None
    shared_search_bundle = getattr(runtime, "shared_stage2_search_bundle", None) if runtime is not None else None

    stage2_answers = []
    for item in stage2_outputs:
        stage2_answers.append(
            {
                "agent_idx": item.get("agent_idx"),
                "model_name": item.get("model_name"),
                "answer": clean_answer(item.get("answer", "")),
                "success": bool(item.get("success", False)),
                "tools": [
                    tool.get("tool_name")
                    for tool in item.get("tool_usage", []) or []
                    if isinstance(tool, dict) and tool.get("tool_name")
                ],
                "routing": item.get("routing", {}),
                "error": item.get("error"),
            }
        )

    return {
        "sample_index": sample_index,
        "task_id": str(sample.get("task_id", "") or ""),
        "attachment_type": attachment_type(sample),
        "expected": expected,
        "stage1_result": stage1_result,
        "stage1_exact": evaluator._check_exact_match(stage1_result, expected),
        "final_result": final_result,
        "predicted": predicted,
        "final_exact": bool(sample_result.get("exact_match", False)),
        "final_partial": bool(sample_result.get("partial_match", False)),
        "score": sample_result.get("score"),
        "selected_agent_idx": final_decision.get("selected_agent_idx"),
        "stage2_answers": stage2_answers,
        "stage2_tool_names": stage2_tool_names(stage2_outputs),
        "used_previous_memory": bool(memory_used),
        "memory_hits": [
            {
                "memory_type": hit.get("memory_type", ""),
                "marker": hit.get("marker", ""),
                "query": hit.get("query", ""),
                "preview": hit.get("preview", ""),
            }
            for hit in memory_hits or []
        ],
        "shared_attachment_used": bool(
            isinstance(shared_attachment_bundle, dict) and shared_attachment_bundle.get("used")
        ),
        "shared_attachment_reader": (
            (shared_attachment_bundle.get("metadata") or {}).get("reader", "")
            if isinstance(shared_attachment_bundle, dict)
            else ""
        ),
        "shared_search_enabled": bool(
            isinstance(shared_search_bundle, dict) and shared_search_bundle.get("enabled")
        ),
        "question_excerpt": " ".join(str(sample.get("question", "") or "").split())[:240],
    }


def status_text(record: dict[str, Any]) -> str:
    if record["final_exact"]:
        return "exact"
    if record["final_partial"]:
        return "partial"
    return "wrong"


def write_markdown_table(handle, headers: list[str], rows: list[list[Any]]) -> None:
    handle.write("| " + " | ".join(headers) + " |\n")
    handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for row in rows:
        clean_row = [str(cell).replace("\n", " ").replace("|", "\\|") for cell in row]
        handle.write("| " + " | ".join(clean_row) + " |\n")
    handle.write("\n")


def infer_wrong_common_features(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wrong = [record for record in records if not record["final_exact"]]
    total_wrong = len(wrong)
    if total_wrong == 0:
        return []

    def count_if(predicate):
        return sum(1 for record in wrong if predicate(record))

    features = [
        ("stage1_wrong", count_if(lambda r: not r["stage1_exact"]), "Stage1 already produced a wrong answer."),
        ("memory_used", count_if(lambda r: r["used_previous_memory"]), "A previous memory lesson was retrieved during the sample."),
        (
            "search_used",
            count_if(lambda r: r["shared_search_enabled"] or "search" in r["stage2_tool_names"]),
            "Search evidence was used or shared search was enabled.",
        ),
        ("attachment_task", count_if(lambda r: r["attachment_type"] != "-"), "The task had an attachment."),
        ("attachment_used", count_if(lambda r: r["shared_attachment_used"]), "Shared attachment evidence was available."),
        ("final_changed_stage1", count_if(lambda r: r["final_result"] != r["stage1_result"]), "Final decision changed the stage1 answer."),
        (
            "stage1_correct_but_final_wrong",
            count_if(lambda r: r["stage1_exact"] and not r["final_exact"]),
            "Stage1 was correct, but final answer became wrong.",
        ),
    ]
    return [
        {"feature": name, "count": count, "rate": count / total_wrong, "note": note}
        for name, count, note in features
    ]


def write_gaia_analysis_report(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    run_metadata: dict[str, Any],
) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage2_pulled_back = [record for record in records if not record["stage1_exact"] and record["final_exact"]]
    stage2_broke_correct = [record for record in records if record["stage1_exact"] and not record["final_exact"]]
    memory_wrong = [record for record in records if record["used_previous_memory"] and not record["final_exact"]]
    exact_count = sum(1 for record in records if record["final_exact"])
    partial_count = sum(1 for record in records if record["final_partial"])
    wrong_features = infer_wrong_common_features(records)

    with output_path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write("# GAIA Run Analysis\n\n")
        handle.write("## Summary\n\n")
        write_markdown_table(
            handle,
            ["metric", "value"],
            [
                ["run_started_at", run_metadata.get("run_started_at", "")],
                ["level", run_metadata.get("level", "")],
                ["total_samples", len(records)],
                ["exact_matches", exact_count],
                ["partial_matches", partial_count],
                ["stage2_pulled_back", len(stage2_pulled_back)],
                ["stage1_correct_but_stage2_wrong", len(stage2_broke_correct)],
                ["memory_used_and_final_wrong", len(memory_wrong)],
                ["memory_mode", run_metadata.get("memory_mode", "")],
                ["memory_enabled", run_metadata.get("memory_enabled", "")],
            ],
        )

        handle.write("## 1. Stage1 錯，但 Stage2/Final 拉回正確的題目\n\n")
        write_markdown_table(
            handle,
            ["#", "task_id", "attachment", "expected", "stage1_result", "final_result", "selected_agent"],
            [
                [
                    record["sample_index"],
                    record["task_id"],
                    record["attachment_type"],
                    record["expected"],
                    record["stage1_result"],
                    record["final_result"],
                    record["selected_agent_idx"],
                ]
                for record in stage2_pulled_back
            ] or [["(none)", "", "", "", "", "", ""]],
        )

        handle.write("## 2. Stage1 對，但 Stage2/Final 變錯的題目\n\n")
        write_markdown_table(
            handle,
            ["#", "task_id", "attachment", "expected", "stage1_result", "final_result", "selected_agent"],
            [
                [
                    record["sample_index"],
                    record["task_id"],
                    record["attachment_type"],
                    record["expected"],
                    record["stage1_result"],
                    record["final_result"],
                    record["selected_agent_idx"],
                ]
                for record in stage2_broke_correct
            ] or [["(none)", "", "", "", "", "", ""]],
        )

        handle.write("## 3. 可能因 Memory 學到錯誤經驗的題目\n\n")
        handle.write(
            "判定方式：本題 retrieval 命中 previous memory，且最後沒有 exact match。"
            "這不代表 memory 一定造成錯誤，但代表該題需要人工檢查 memory 是否干擾。\n\n"
        )
        write_markdown_table(
            handle,
            ["#", "task_id", "status", "expected", "predicted", "memory_hit_count", "hit_types"],
            [
                [
                    record["sample_index"],
                    record["task_id"],
                    status_text(record),
                    record["expected"],
                    record["predicted"],
                    len(record["memory_hits"]),
                    ", ".join(sorted({hit["memory_type"] for hit in record["memory_hits"] if hit["memory_type"]})),
                ]
                for record in memory_wrong
            ] or [["(none)", "", "", "", "", "", ""]],
        )

        if memory_wrong:
            handle.write("### Memory Hit Details\n\n")
            for record in memory_wrong:
                handle.write(f"#### 題目 {record['sample_index']} - {record['task_id']}\n\n")
                for hit in record["memory_hits"][:5]:
                    handle.write(
                        "- "
                        f"type={hit['memory_type']} | "
                        f"marker={hit['marker']} | "
                        f"query={hit['query']}\n"
                    )
                    preview = str(hit.get("preview", "") or "").replace("\n", " ")
                    if preview:
                        handle.write(f"  - preview: {preview[:300]}\n")
                handle.write("\n")

        handle.write("## 4. 錯誤題目的共同特徵\n\n")
        write_markdown_table(
            handle,
            ["feature", "wrong_count", "wrong_rate", "note"],
            [[item["feature"], item["count"], f"{item['rate']:.2%}", item["note"]] for item in wrong_features],
        )

        handle.write("## Per-Sample Overview\n\n")
        write_markdown_table(
            handle,
            ["#", "status", "attachment", "memory_used", "expected", "stage1_result", "final_result", "predicted"],
            [
                [
                    record["sample_index"],
                    status_text(record),
                    record["attachment_type"],
                    record["used_previous_memory"],
                    record["expected"],
                    record["stage1_result"],
                    record["final_result"],
                    record["predicted"],
                ]
                for record in records
            ],
        )

        handle.write("## Raw JSON\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(records, ensure_ascii=False, indent=2))
        handle.write("\n```\n")

    return output_path
