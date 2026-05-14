import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from evaluation.benchmarks.bfcl.dataset import BFCLDataset
from evaluation.benchmarks.bfcl.bfcl_runner import DEFAULT_BFCL_DATA_DIR, run_bfcl_evaluation
from utils.project_paths import get_eval_output_path, get_log_file_path


def parse_bfcl_args():
    """
    負責解析 test_bfcl.py 的命令列參數，並補齊 BFCL 評估輸出路徑。

    Args:
        無；參數會直接從 sys.argv 讀取。

    Returns:
        argparse.Namespace，包含 category、max_samples、log_file、compact_log_file、analysis_file 與 result_file 等設定。

    限制或副作用:
        若使用 --list-categories，main() 會只列出可用 category 而不執行評估。
    """
    categories = list(BFCLDataset.CATEGORY_MAPPING.keys())
    parser = argparse.ArgumentParser(description="Run a BFCL evaluation.")
    parser.add_argument(
        "--category",
        default="simple_python",
        choices=categories,
        help="BFCL category to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        "--max-sample",
        dest="max_samples",
        type=int,
        default=3,
        help="Maximum number of BFCL samples to evaluate. Use 0 to run all loaded samples.",
    )
    parser.add_argument(
        "--evaluation-mode",
        default="ast",
        choices=["ast", "execution"],
        help="Evaluation mode. execution currently falls back to structural matching.",
    )
    parser.add_argument(
        "--bfcl-data-dir",
        default=str(DEFAULT_BFCL_DATA_DIR),
        help="Path to the BFCL data directory.",
    )
    parser.add_argument(
        "--log-name",
        default="test_bfcl",
        help=(
            "Base output log name. Generates <name>.log, "
            "<name>_compact.log, <name>_analysis.md, and <name>_<category>_results.json."
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
        "--result-file",
        default=None,
        help="Optional explicit JSON result path. Overrides --log-name for the result file.",
    )
    parser.add_argument("--agents", type=int, default=3, help="Number of agents in AgentNetwork.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of stage-1 rounds in AgentNetwork.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for AgentNetwork.")
    parser.add_argument(
        "--memory-mode",
        default="stage1_first_round_only",
        help="GraphMemory injection mode used by AgentNetwork.",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print available BFCL categories and exit.",
    )
    args = parser.parse_args()

    base_name = str(args.log_name or "test_bfcl").strip()
    category = str(args.category or "simple_python").strip()
    if not args.log_file:
        args.log_file = str(get_log_file_path(f"{base_name}.log"))
    if not args.compact_log_file:
        args.compact_log_file = str(get_log_file_path(f"{base_name}_compact.log"))
    if not args.analysis_file:
        args.analysis_file = str(get_log_file_path(f"{base_name}_analysis.md"))
    if not args.result_file:
        args.result_file = str(get_eval_output_path(f"{base_name}_{category}_results.json"))
    return args


def main():
    """
    負責執行 BFCL 評估測試入口，或列出可用 BFCL category。

    Args:
        無；會呼叫 parse_bfcl_args() 取得命令列參數。

    Returns:
        無；評估結果會由 run_bfcl_evaluation() 寫入 log、analysis 與 result 檔案。

    限制或副作用:
        執行評估時會呼叫模型、寫入 result/log 檔案，並可能觸發 GraphMemory retrieval。
    """
    args = parse_bfcl_args()
    if args.list_categories:
        print("Available BFCL categories:")
        for category in BFCLDataset.CATEGORY_MAPPING:
            print(f"- {category}")
        return
    run_bfcl_evaluation(args)


if __name__ == "__main__":
    main()
