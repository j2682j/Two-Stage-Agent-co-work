"""日誌工具"""

import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "hello_agents",
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    負責執行 utils.logging 中的 setup_logger 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
    
    Args:
        name: 此流程需要使用的輸入資料。
        level: 此流程需要使用的輸入資料。
        format_string: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 logging.Logger。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            format_string or 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def get_logger(name: str = "hello_agents") -> logging.Logger:
    """
    負責執行 utils.logging 中的 get_logger 流程，依照 utils.logging 的流程需求處理 get_logger 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        name: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 logging.Logger。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return logging.getLogger(name) 