from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttachmentReaderConfig:
    max_text_chars: int = 12000
    max_table_rows: int = 80
    max_pdf_pages: int = 20
    python_timeout: int = 20
    vision_model: str = "qwen3-vl:8b"
    vision_timeout: int = 180
    audio_model_size: str = "base"
    audio_device: str = "cuda"
    audio_compute_type: str = "float16"
    max_zip_members: int = 30
    max_zip_file_bytes: int = 8 * 1024 * 1024
    max_zip_total_bytes: int = 40 * 1024 * 1024
    max_zip_depth: int = 1


@dataclass
class AttachmentReadResult:
    ok: bool
    reader: str
    content: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
