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
    建立 MCP 上下文對象
    
    Args:
        messages: 消息列表
        tools: 工具列表
        resources: 資源列表
        metadata: 元資料
        
    Returns:
        上下文字典
        
    Example:
        >>> context = create_context(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     tools=[{"name": "calculator", "description": "計算器"}]
        ... )
    """
    return {
        "messages": messages or [],
        "tools": tools or [],
        "resources": resources or [],
        "metadata": metadata or {}
    }


def parse_context(context: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    解析 MCP 上下文
    
    Args:
        context: 上下文字串或字典
        
    Returns:
        解析後的上下文字典
        
    Raises:
        ValueError: 如果上下文格式無效
        
    Example:
        >>> context_str = '{"messages": [], "tools": []}'
        >>> parsed = parse_context(context_str)
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
    建立錯誤回應
    
    Args:
        error_message: 錯誤消息
        error_code: 錯誤代碼
        details: 錯誤詳情
        
    Returns:
        錯誤回應字典
        
    Example:
        >>> error = create_error_response("Tool not found", "TOOL_NOT_FOUND")
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
    建立成功回應
    
    Args:
        data: 回應資料
        metadata: 元資料
        
    Returns:
        成功回應字典
        
    Example:
        >>> response = create_success_response({"result": 42})
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

