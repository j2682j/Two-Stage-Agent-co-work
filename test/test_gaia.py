import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from evaluation.benchmarks.gaia.gaia_runner import run_gaia_evaluation
from utils.project_paths import get_log_file_path


def parse_gaia_args():
    """
    負責執行 test.test_gaia 中的 parse_gaia_args 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    parser = argparse.ArgumentParser(description="Run a GAIA evaluation.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument(
        "--log-name",
        default="test_gaia_enable_memory_with_attchment",
        help=(
            "Base output log name. Generates <name>.log, "
            "<name>_compact.log, and <name>_analysis.md under the log directory."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional explicit UTF-8 full log path. Overrides --log-name for the full log.",
    )
    parser.add_argument(
        "--compact-log-file",
        default=None,
        help="Optional explicit compact log path. Overrides --log-name for the compact log.",
    )
    parser.add_argument(
        "--analysis-file",
        default=None,
        help="Optional explicit Markdown analysis path. Overrides --log-name for the analysis report.",
    )
    parser.add_argument(
        "--enable-stage1-attachment-after-first-round",
        action="store_true",
        help="Reuse attachment evidence in later stage-1 rounds after the first round.",
    )
    args = parser.parse_args()
    base_name = str(args.log_name or "test_gaia").strip()
    if not args.log_file:
        args.log_file = str(get_log_file_path(f"{base_name}.log"))
    if not args.compact_log_file:
        args.compact_log_file = str(get_log_file_path(f"{base_name}_compact.log"))
    if not args.analysis_file:
        args.analysis_file = str(get_log_file_path(f"{base_name}_analysis.md"))
    return args


def main():
    """
    負責執行 test.test_gaia 中的 main 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    args = parse_gaia_args()
    run_gaia_evaluation(args)


if __name__ == "__main__":
    main()
