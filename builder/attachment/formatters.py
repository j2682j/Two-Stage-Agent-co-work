from __future__ import annotations

from typing import Any


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def compact_single_line(value: Any, default: str = "(blank)") -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    return " ".join(text.split())


def format_rows(title: str, rows: list[list[str]], *, truncated: bool, max_rows: int) -> str:
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
