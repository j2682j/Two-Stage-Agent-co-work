"""輔助工具函式"""

import importlib
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

def format_time(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化時間
    
    Args:
        timestamp: 時間戳，預設為目前時間
        format_str: 格式字串
        
    Returns:
        格式化後的時間字串
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(format_str)

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    驗證設定是否包含必需的鍵
    
    Args:
        config: 設定字典
        required_keys: 必需的鍵列表
        
    Returns:
        是否驗證通過
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"設定缺少必需的鍵: {missing_keys}")
    return True

def safe_import(module_name: str, class_name: Optional[str] = None) -> Any:
    """
    安全匯入模組或類
    
    Args:
        module_name: 模組名
        class_name: 類名（可選）
        
    Returns:
        匯入的模組或類
    """
    try:
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        raise ImportError(f"無法匯入 {module_name}.{class_name or ''}: {e}")

def ensure_dir(path: Path) -> Path:
    """確保目錄存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root() -> Path:
    """取得項目根目錄"""
    return Path(__file__).parent.parent.parent

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """深度合併兩個字典"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result