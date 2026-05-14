"""輔助工具函式"""

import importlib
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

def format_time(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    負責執行 utils.helpers 中的 format_time 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
    
    Args:
        timestamp: 此流程需要使用的輸入資料。
        format_str: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(format_str)

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    負責執行 utils.helpers 中的 validate_config 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
    
    Args:
        config: 控制此流程行為的設定資料。
        required_keys: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"設定缺少必需的鍵: {missing_keys}")
    return True

def safe_import(module_name: str, class_name: Optional[str] = None) -> Any:
    """
    負責執行 utils.helpers 中的 safe_import 流程，依照 utils.helpers 的流程需求處理 safe_import 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        module_name: 此流程需要使用的輸入資料。
        class_name: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Any。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        raise ImportError(f"無法匯入 {module_name}.{class_name or ''}: {e}")

def ensure_dir(path: Path) -> Path:
    """
    負責執行 utils.helpers 中的 ensure_dir 流程，依照 utils.helpers 的流程需求處理 ensure_dir 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        path: 要讀取或寫入的檔案或目錄路徑。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Path。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root() -> Path:
    """
    負責執行 utils.helpers 中的 get_project_root 流程，依照 utils.helpers 的流程需求處理 get_project_root 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Path。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return Path(__file__).parent.parent.parent

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    負責執行 utils.helpers 中的 merge_dicts 流程，依照 utils.helpers 的流程需求處理 merge_dicts 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        dict1: 此流程需要使用的輸入資料。
        dict2: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dict。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result