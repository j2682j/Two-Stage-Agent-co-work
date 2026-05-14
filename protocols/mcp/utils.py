"""
MCP 協議工具函式

提供上下文管理、消息解析等輔助功能。
這些函式主要用於處理 MCP 協議的資料結構。
"""

from typing import Dict, Any, List, Optional, Union
import json


def create_context(
    messages: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    resources: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    負責執行 protocols.mcp.utils 中的 create_context 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        messages: 此流程需要使用的輸入資料。
        tools: 可呼叫的工具、工具名稱或工具註冊表。
        resources: 此流程需要使用的輸入資料。
        metadata: 目前流程所需的上下文、狀態或附加資訊。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return {
        "messages": messages or [],
        "tools": tools or [],
        "resources": resources or [],
        "metadata": metadata or {}
    }


def parse_context(context: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    負責執行 protocols.mcp.utils 中的 parse_context 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
    
    Args:
        context: 目前流程所需的上下文、狀態或附加資訊。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON context: {e}")
    
    if not isinstance(context, dict):
        raise ValueError("Context must be a dictionary or JSON string")
    
    # 確保必需字段存在
    for field in ["messages", "tools", "resources"]:
        context.setdefault(field, [])
    context.setdefault("metadata", {})
    
    return context


def create_error_response(
    error_message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    負責執行 protocols.mcp.utils 中的 create_error_response 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        error_message: 此流程需要使用的輸入資料。
        error_code: 此流程需要使用的輸入資料。
        details: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    response = {
        "error": {
            "message": error_message,
            "code": error_code or "UNKNOWN_ERROR"
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    return response


def create_success_response(
    data: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    負責執行 protocols.mcp.utils 中的 create_success_response 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        data: 此流程需要使用的輸入資料。
        metadata: 目前流程所需的上下文、狀態或附加資訊。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    response = {
        "success": True,
        "data": data
    }
    
    if metadata:
        response["metadata"] = metadata
    
    return response


__all__ = [
    "create_context",
    "parse_context",
    "create_error_response",
    "create_success_response",
]

