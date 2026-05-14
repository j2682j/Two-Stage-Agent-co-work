"""
Benchmark logging 共用基底。

此模組放置 GAIA、BFCL 等 benchmark 都會用到的 logging 能力：
UTF-8 full log tee、compact log 開檔、stdout/stderr 還原、JSON 安全寫入、
token usage 摘要與 GraphMemory retrieval 摘要。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO


class TeeStream:
    """
    負責把 stdout 或 stderr 同時寫到原本的 stream 與 log 檔案。

    Args:
        primary: 原本的 stdout 或 stderr stream。
        mirror: 要同步寫入的 log file handle。

    Returns:
        TeeStream 實例，可指派給 sys.stdout 或 sys.stderr。

    限制或副作用:
        write() 會同時寫入兩個 stream；呼叫端需要在使用完畢後還原原始 stdio。
    """

    def __init__(self, primary: TextIO, mirror: TextIO):
        """
        負責初始化 TeeStream 的主要輸出與鏡像輸出。

        Args:
            primary: 原本的輸出 stream。
            mirror: 鏡像寫入的 log stream。

        Returns:
            無。

        限制或副作用:
            不會接管 stdio；接管動作由 setup_utf8_log() 負責。
        """
        self.primary = primary
        self.mirror = mirror

    @property
    def encoding(self) -> str:
        """
        負責回傳 primary stream 的 encoding。

        Args:
            無。

        Returns:
            primary stream 的 encoding；若不存在則回傳 utf-8。

        限制或副作用:
            只讀取屬性，不會修改 stream。
        """
        return getattr(self.primary, "encoding", "utf-8")

    def write(self, data: Any) -> int:
        """
        負責把資料同步寫入 primary 與 mirror。

        Args:
            data: 要寫入的文字或可轉字串資料。

        Returns:
            寫入文字長度。

        限制或副作用:
            會立即寫入兩個 stream；若任一 stream 寫入失敗，例外會往外拋。
        """
        if not isinstance(data, str):
            data = str(data)
        self.primary.write(data)
        self.mirror.write(data)
        return len(data)

    def flush(self) -> None:
        """
        負責 flush primary 與 mirror。

        Args:
            無。

        Returns:
            無。

        限制或副作用:
            會觸發兩個 stream 的 flush。
        """
        self.primary.flush()
        self.mirror.flush()

    def isatty(self) -> bool:
        """
        負責回傳 primary 是否為互動式終端。

        Args:
            無。

        Returns:
            若 primary 是 TTY 回傳 True，否則回傳 False。

        限制或副作用:
            只讀取 primary 狀態。
        """
        return getattr(self.primary, "isatty", lambda: False)()

    def fileno(self) -> int:
        """
        負責回傳 primary stream 的檔案描述符。

        Args:
            無。

        Returns:
            primary.fileno() 的結果。

        限制或副作用:
            若 primary 不支援 fileno，會拋出原始例外。
        """
        return self.primary.fileno()

    def __getattr__(self, name: str) -> Any:
        """
        負責把未知屬性委派給 primary stream。

        Args:
            name: 屬性名稱。

        Returns:
            primary 上對應的屬性或方法。

        限制或副作用:
            若 primary 沒有該屬性，會拋出 AttributeError。
        """
        return getattr(self.primary, name)


class BenchmarkLogger:
    """
    負責提供 benchmark logging 的共用工具方法。

    Args:
        無。

    Returns:
        BenchmarkLogger 實例，可由 GAIA/BFCL logger 繼承或直接以 static method 使用。

    限制或副作用:
        setup_utf8_log() 會暫時替換 sys.stdout 與 sys.stderr。
    """

    @staticmethod
    def setup_utf8_log(log_file_path: Path):
        """
        負責建立 UTF-8 full log，並把 stdout/stderr tee 到 log 檔。

        Args:
            log_file_path: full log 輸出路徑。

        Returns:
            三元組：(log_handle, original_stdout, original_stderr)。

        限制或副作用:
            會建立父資料夾，開啟 log 檔，並修改 sys.stdout/sys.stderr。
        """
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

    @staticmethod
    def restore_stdio(*, original_stdout, original_stderr) -> None:
        """
        負責還原 setup_utf8_log() 替換過的 stdout/stderr。

        Args:
            original_stdout: setup 前的 stdout。
            original_stderr: setup 前的 stderr。

        Returns:
            無。

        限制或副作用:
            會 flush 目前 stdout/stderr，並改回原始 stream。
        """
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    @staticmethod
    def open_utf8_text_log(log_file_path: Path):
        """
        負責開啟 UTF-8-SIG 文字 log 檔。

        Args:
            log_file_path: 文字 log 輸出路徑。

        Returns:
            已開啟的文字檔 handle。

        限制或副作用:
            會建立父資料夾，並覆寫同名檔案。
        """
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        return log_file_path.open("w", encoding="utf-8-sig", buffering=1, newline="\n")

    @staticmethod
    def write_indented_block(handle: Any, text: Any, indent: str = "    ") -> None:
        """
        負責把多行文字以固定縮排寫入 log。

        Args:
            handle: 已開啟的文字檔 handle。
            text: 要寫入的文字。
            indent: 每一行前方要加上的縮排。

        Returns:
            無。

        限制或副作用:
            空內容會寫入 `(none)`。
        """
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

    @staticmethod
    def write_json_line(handle: Any, label: str, value: Any, *, indent: str = "") -> None:
        """
        負責把資料以單行 JSON 格式寫入 log。

        Args:
            handle: 已開啟的文字檔 handle。
            label: 欄位名稱。
            value: 要輸出的資料。
            indent: 行首縮排。

        Returns:
            無。

        限制或副作用:
            無法 JSON 序列化時會 fallback 到 repr 字串。
        """
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            encoded = repr(value)
        handle.write(f"{indent}- {label}: {encoded}\n")

    @staticmethod
    def token_usage_summary(network_or_runtime: Any) -> dict[str, Any]:
        """
        負責從 network 或 runtime 取得 token usage summary。

        Args:
            network_or_runtime: AgentNetwork、NetworkRuntime，或具備 token_usage_summary() 的物件。

        Returns:
            token usage 統計字典。

        限制或副作用:
            若找不到 runtime 或 token_usage_summary()，會回傳零值預設結構。
        """
        runtime = getattr(network_or_runtime, "runtime", network_or_runtime)
        if runtime is None or not hasattr(runtime, "token_usage_summary"):
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "by_stage": {},
                "by_model": {},
                "records": [],
            }
        return runtime.token_usage_summary()

    @staticmethod
    def graph_memory_trace_summary(network_or_runtime: Any, task_id: str | None = None) -> dict[str, Any]:
        """
        負責整理 GraphMemory retrieval 與寫入 trace 的共用摘要。

        Args:
            network_or_runtime: AgentNetwork 或 NetworkRuntime。
            task_id: 若提供，僅保留該 task_id 的讀寫紀錄。

        Returns:
            包含 reads、writes、related_task_ids、insight_ids、qdrant_hits、expanded_hits 與 hit count 的字典。

        限制或副作用:
            只讀取 runtime.shared_memory_reads/shared_memory_writes，不會觸發新的 retrieval 或寫入。
        """
        runtime = getattr(network_or_runtime, "runtime", network_or_runtime)
        reads_source = list(getattr(runtime, "shared_memory_reads", []) or []) if runtime is not None else []
        writes_source = list(getattr(runtime, "shared_memory_writes", []) or []) if runtime is not None else []

        def same_task(item: dict[str, Any]) -> bool:
            return not task_id or str(item.get("task_id", "")) == str(task_id)

        reads = [
            item
            for item in reads_source
            if isinstance(item, dict) and item.get("source") == "graph_memory" and same_task(item)
        ]
        writes = [
            item
            for item in writes_source
            if isinstance(item, dict)
            and item.get("memory_type") in {"graph_memory", "interaction_graph"}
            and same_task(item)
        ]

        related_task_ids: list[str] = []
        insight_ids: list[str] = []
        qdrant_hits: list[dict[str, Any]] = []
        expanded_hits: list[dict[str, Any]] = []

        def append_unique(values: list[str], value: Any) -> None:
            text = str(value or "").strip()
            if text and text not in values:
                values.append(text)

        for read in reads:
            for key in ("related_task_ids", "seed_task_ids", "expanded_task_ids"):
                for value in read.get(key, []) or []:
                    append_unique(related_task_ids, value)

            for value in read.get("insight_ids", []) or []:
                append_unique(insight_ids, value)

            for key in ("qdrant_hits", "seed_task_hits"):
                for hit in read.get(key, []) or []:
                    if isinstance(hit, dict):
                        qdrant_hits.append(hit)

            for key in ("expanded_hits", "expanded_task_hits"):
                for hit in read.get(key, []) or []:
                    if isinstance(hit, dict):
                        expanded_hits.append(hit)

        return {
            "reads": reads,
            "writes": writes,
            "related_task_ids": related_task_ids,
            "insight_ids": insight_ids,
            "qdrant_hits": qdrant_hits,
            "expanded_hits": expanded_hits,
            "qdrant_hit_count": len(qdrant_hits),
            "expanded_hit_count": len(expanded_hits),
            "retrieval_hit": bool(related_task_ids or insight_ids or qdrant_hits or expanded_hits),
        }


setup_utf8_log = BenchmarkLogger.setup_utf8_log
restore_stdio = BenchmarkLogger.restore_stdio
open_utf8_text_log = BenchmarkLogger.open_utf8_text_log
write_indented_block = BenchmarkLogger.write_indented_block
write_json_line = BenchmarkLogger.write_json_line
token_usage_summary = BenchmarkLogger.token_usage_summary
graph_memory_trace_summary = BenchmarkLogger.graph_memory_trace_summary
