from __future__ import annotations

import argparse

from utils.project_paths import get_log_file_path


def parse_gaia_args():
    parser = argparse.ArgumentParser(description="Run a small GAIA memory smoke test.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument(
        "--log-file",
        default=str(get_log_file_path("test_gaia_enable_memory_with_attchment.log")),
        help="UTF-8 log file path written directly by Python.",
    )
    parser.add_argument(
        "--compact-log-file",
        default=str(get_log_file_path("test_gaia_compact_enable_memory_with_attchment.log")),
        help="Compact UTF-8 summary log file path written directly by Python.",
    )
    parser.add_argument(
        "--analysis-file",
        default=str(get_log_file_path("test_gaia_analysis_enable_memory_with_attchment.md")),
        help="Markdown analysis report written after the run finishes.",
    )
    parser.add_argument(
        "--enable-stage1-attachment-after-first-round",
        action="store_true",
        help="Reuse attachment evidence in later stage-1 rounds after the first round.",
    )
    return parser.parse_args()
