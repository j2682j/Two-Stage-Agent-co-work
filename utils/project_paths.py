from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_LOG_DIR = RESULT_DIR / "logs"
RESULT_EVAL_DIR = RESULT_DIR / "eval"
RESULT_MEMORY_DIR = RESULT_DIR / "memory"
MEMORY_DATA_DIR = PROJECT_ROOT / "memory_data"


def ensure_runtime_dirs() -> None:
    """
    負責執行 utils.project_paths 中的 ensure_runtime_dirs 流程，依照 utils.project_paths 的流程需求處理 ensure_runtime_dirs 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    for path in (
        RESULT_DIR,
        RESULT_LOG_DIR,
        RESULT_EVAL_DIR,
        RESULT_MEMORY_DIR,
        MEMORY_DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_log_file_path(filename: str) -> Path:
    """
    負責執行 utils.project_paths 中的 get_log_file_path 流程，依照 utils.project_paths 的流程需求處理 get_log_file_path 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        filename: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Path。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    ensure_runtime_dirs()
    return RESULT_LOG_DIR / filename


def get_eval_output_path(filename: str) -> Path:
    """
    負責執行 utils.project_paths 中的 get_eval_output_path 流程，依照 utils.project_paths 的流程需求處理 get_eval_output_path 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        filename: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Path。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    ensure_runtime_dirs()
    return RESULT_EVAL_DIR / filename


def get_eval_output_dir(dirname: str | None = None) -> Path:
    """
    負責執行 utils.project_paths 中的 get_eval_output_dir 流程，依照 utils.project_paths 的流程需求處理 get_eval_output_dir 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        dirname: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Path。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    ensure_runtime_dirs()
    return RESULT_EVAL_DIR if not dirname else RESULT_EVAL_DIR / dirname


def get_memory_output_dir(dirname: str | None = None) -> Path:
    """
    負責執行 utils.project_paths 中的 get_memory_output_dir 流程，依照 utils.project_paths 的流程需求處理 get_memory_output_dir 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        dirname: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Path。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    ensure_runtime_dirs()
    return RESULT_MEMORY_DIR if not dirname else RESULT_MEMORY_DIR / dirname


def get_memory_data_dir() -> Path:
    """
    負責執行 utils.project_paths 中的 get_memory_data_dir 流程，依照 utils.project_paths 的流程需求處理 get_memory_data_dir 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Path。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    ensure_runtime_dirs()
    return MEMORY_DATA_DIR
