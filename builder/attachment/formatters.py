from __future__ import annotations

from typing import Any


def truncate_text(text: str, max_chars: int) -> str:
    """
    負責執行 builder.attachment.formatters 中的 truncate_text 流程，依照 builder.attachment.formatters 的流程需求處理 truncate_text 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        text: 此流程需要使用的輸入資料。
        max_chars: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def compact_single_line(value: Any, default: str = "(blank)") -> str:
    """
    負責執行 builder.attachment.formatters 中的 compact_single_line 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
    
    Args:
        value: 此流程需要使用的輸入資料。
        default: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    return " ".join(text.split())


def format_rows(title: str, rows: list[list[str]], *, truncated: bool, max_rows: int) -> str:
    """
    負責執行 builder.attachment.formatters 中的 format_rows 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
    
    Args:
        title: 此流程需要使用的輸入資料。
        rows: 此流程需要使用的輸入資料。
        truncated: 此流程需要使用的輸入資料。
        max_rows: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    lines = [title]
    for row in rows:
        lines.append(" | ".join(row))
    if truncated:
        lines.append(f"[showing first {max_rows} rows]")
    return "\n".join(lines)


def format_attachment_context(
    *,
    file_name: str,
    file_path: Any,
    extension: str,
    content: str,
    warnings: list[str],
) -> str:
    """
    負責執行 builder.attachment.formatters 中的 format_attachment_context 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
    
    Args:
        file_name: 此流程需要使用的輸入資料。
        file_path: 要讀取或寫入的檔案或目錄路徑。
        extension: 此流程需要使用的輸入資料。
        content: 此流程需要使用的輸入資料。
        warnings: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    lines = [
        "Attachment evidence:",
        "File:",
        f"- name: {file_name}",
        f"- path: {file_path}",
        f"- type: {extension or '(none)'}",
        "- exists: true",
    ]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    if str(content or "").strip():
        lines.append("Extracted content:")
        lines.append(str(content).strip())
    return "\n".join(lines).strip()
