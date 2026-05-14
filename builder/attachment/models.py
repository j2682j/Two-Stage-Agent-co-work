from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttachmentReaderConfig:
    """
    負責在 builder.attachment.models 中封裝 AttachmentReaderConfig，封裝附件讀取與內容萃取流程，將檔案轉成可推理的證據。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
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
    """
    負責在 builder.attachment.models 中封裝 AttachmentReadResult，封裝附件讀取與內容萃取流程，將檔案轉成可推理的證據。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    ok: bool
    reader: str
    content: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
