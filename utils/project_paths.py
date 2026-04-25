from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_LOG_DIR = RESULT_DIR / "logs"
RESULT_EVAL_DIR = RESULT_DIR / "eval"
RESULT_MEMORY_DIR = RESULT_DIR / "memory"
MEMORY_DATA_DIR = PROJECT_ROOT / "memory_data"


def ensure_runtime_dirs() -> None:
    for path in (
        RESULT_DIR,
        RESULT_LOG_DIR,
        RESULT_EVAL_DIR,
        RESULT_MEMORY_DIR,
        MEMORY_DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_log_file_path(filename: str) -> Path:
    ensure_runtime_dirs()
    return RESULT_LOG_DIR / filename


def get_eval_output_path(filename: str) -> Path:
    ensure_runtime_dirs()
    return RESULT_EVAL_DIR / filename


def get_eval_output_dir(dirname: str | None = None) -> Path:
    ensure_runtime_dirs()
    return RESULT_EVAL_DIR if not dirname else RESULT_EVAL_DIR / dirname


def get_memory_output_dir(dirname: str | None = None) -> Path:
    ensure_runtime_dirs()
    return RESULT_MEMORY_DIR if not dirname else RESULT_MEMORY_DIR / dirname


def get_memory_data_dir() -> Path:
    ensure_runtime_dirs()
    return MEMORY_DATA_DIR
