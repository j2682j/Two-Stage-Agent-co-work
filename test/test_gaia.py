import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from evaluation.gaia_run_config import parse_gaia_args
from evaluation.gaia_runner import run_gaia_evaluation


def main():
    args = parse_gaia_args()
    run_gaia_evaluation(args)


if __name__ == "__main__":
    main()
