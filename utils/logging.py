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
    設定日誌紀錄器
    
    Args:
        name: 日誌紀錄器名稱
        level: 日誌等級
        format_string: 日誌格式
        
    Returns:
        設定好的日誌紀錄器
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
    """取得日誌紀錄器"""
    return logging.getLogger(name) 