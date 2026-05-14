"""序列化工具"""

import json
import pickle
from typing import Any, Union
from pathlib import Path

def serialize_object(obj: Any, format: str = "json") -> Union[str, bytes]:
    """
    負責執行 utils.serialization 中的 serialize_object 流程，依照 utils.serialization 的流程需求處理 serialize_object 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        obj: 此流程需要使用的輸入資料。
        format: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Union[str, bytes]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if format == "json":
        return json.dumps(obj, ensure_ascii=False, indent=2)
    elif format == "pickle":
        return pickle.dumps(obj)
    else:
        raise ValueError(f"不支援的序列化格式: {format}")

def deserialize_object(data: Union[str, bytes], format: str = "json") -> Any:
    """
    負責執行 utils.serialization 中的 deserialize_object 流程，依照 utils.serialization 的流程需求處理 deserialize_object 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        data: 此流程需要使用的輸入資料。
        format: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Any。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if format == "json":
        return json.loads(data)
    elif format == "pickle":
        return pickle.loads(data)
    else:
        raise ValueError(f"不支援的反序列化格式: {format}")

def save_to_file(obj: Any, filepath: Union[str, Path], format: str = "json") -> None:
    """
    負責執行 utils.serialization 中的 save_to_file 流程，將目前處理結果、設定或狀態寫入指定儲存位置。
    
    Args:
        obj: 此流程需要使用的輸入資料。
        filepath: 此流程需要使用的輸入資料。
        format: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    filepath = Path(filepath)
    data = serialize_object(obj, format)
    
    mode = "w" if format == "json" else "wb"
    with open(filepath, mode) as f:
        f.write(data)

def load_from_file(filepath: Union[str, Path], format: str = "json") -> Any:
    """
    負責執行 utils.serialization 中的 load_from_file 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
    
    Args:
        filepath: 此流程需要使用的輸入資料。
        format: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Any。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    filepath = Path(filepath)
    mode = "r" if format == "json" else "rb"
    
    with open(filepath, mode) as f:
        data = f.read()
    
    return deserialize_object(data, format) 