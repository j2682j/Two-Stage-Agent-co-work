from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .formatters import format_attachment_context, truncate_text
from .models import AttachmentReadResult, AttachmentReaderConfig
from .readers.archive_reader import ArchiveAttachmentReader
from .readers.code_reader import CodeAttachmentReader
from .readers.excel_reader import ExcelAttachmentReader
from .readers.media_reader import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, MediaAttachmentReader
from .readers.office_reader import OfficeAttachmentReader
from .readers.text_reader import TEXT_EXTENSIONS, TextAttachmentReader


class AttachmentEvidenceBuilder:
    """
    負責在 builder.attachment.evidence_builder 中封裝 AttachmentEvidenceBuilder，封裝附件讀取與內容萃取流程，將檔案轉成可推理的證據。
    
    Args:
        max_text_chars: 控制檢索、篩選或輸出數量的數值參數。
        max_table_rows: 控制檢索、篩選或輸出數量的數值參數。
        max_pdf_pages: 控制檢索、篩選或輸出數量的數值參數。
        python_timeout: 此流程需要使用的輸入資料。
        vision_model: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        vision_timeout: 此流程需要使用的輸入資料。
        audio_model_size: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        audio_device: 此流程需要使用的輸入資料。
        audio_compute_type: 此流程需要使用的輸入資料。
        max_zip_members: 控制檢索、篩選或輸出數量的數值參數。
        max_zip_file_bytes: 控制檢索、篩選或輸出數量的數值參數。
        max_zip_total_bytes: 控制檢索、篩選或輸出數量的數值參數。
        max_zip_depth: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    TEXT_EXTENSIONS = TEXT_EXTENSIONS
    IMAGE_EXTENSIONS = IMAGE_EXTENSIONS
    AUDIO_EXTENSIONS = AUDIO_EXTENSIONS

    def __init__(
        self,
        *,
        max_text_chars: int = 12000,
        max_table_rows: int = 80,
        max_pdf_pages: int = 20,
        python_timeout: int = 20,
        vision_model: str | None = None,
        vision_timeout: int | None = None,
        audio_model_size: str | None = None,
        audio_device: str | None = None,
        audio_compute_type: str | None = None,
        max_zip_members: int = 30,
        max_zip_file_bytes: int = 8 * 1024 * 1024,
        max_zip_total_bytes: int = 40 * 1024 * 1024,
        max_zip_depth: int = 1,
    ) -> None:
        """
        負責執行 AttachmentEvidenceBuilder 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            max_text_chars: 控制檢索、篩選或輸出數量的數值參數。
            max_table_rows: 控制檢索、篩選或輸出數量的數值參數。
            max_pdf_pages: 控制檢索、篩選或輸出數量的數值參數。
            python_timeout: 此流程需要使用的輸入資料。
            vision_model: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            vision_timeout: 此流程需要使用的輸入資料。
            audio_model_size: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            audio_device: 此流程需要使用的輸入資料。
            audio_compute_type: 此流程需要使用的輸入資料。
            max_zip_members: 控制檢索、篩選或輸出數量的數值參數。
            max_zip_file_bytes: 控制檢索、篩選或輸出數量的數值參數。
            max_zip_total_bytes: 控制檢索、篩選或輸出數量的數值參數。
            max_zip_depth: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.config = AttachmentReaderConfig(
            max_text_chars=max_text_chars,
            max_table_rows=max_table_rows,
            max_pdf_pages=max_pdf_pages,
            python_timeout=python_timeout,
            vision_model=vision_model or os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b"),
            vision_timeout=vision_timeout
            or int(os.getenv("OLLAMA_VISION_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "180"))),
            audio_model_size=audio_model_size or os.getenv("FASTER_WHISPER_MODEL_SIZE", "base"),
            audio_device=audio_device or os.getenv("FASTER_WHISPER_DEVICE", "cuda"),
            audio_compute_type=audio_compute_type
            or os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "float16"),
            max_zip_members=max_zip_members,
            max_zip_file_bytes=max_zip_file_bytes,
            max_zip_total_bytes=max_zip_total_bytes,
            max_zip_depth=max_zip_depth,
        )
        self.text_reader = TextAttachmentReader(self.config)
        self.office_reader = OfficeAttachmentReader(self.config)
        self.excel_reader = ExcelAttachmentReader(self.config)
        self.code_reader = CodeAttachmentReader(self.config)
        self.media_reader = MediaAttachmentReader(self.config)
        self.archive_reader = ArchiveAttachmentReader(self.config, self)

    def build(self, question: str, attachment: dict[str, Any] | None) -> dict[str, Any]:
        """
        負責執行 AttachmentEvidenceBuilder 中的 build 流程，建立任務需要的證據區塊，整理搜尋、附件或工具輸出的可引用內容。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            attachment: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not attachment:
            return self._empty()

        file_path = Path(str(attachment.get("file_path", "") or ""))
        file_name = str(attachment.get("file_name", "") or file_path.name)
        extension = str(attachment.get("extension", "") or file_path.suffix).lower()
        warnings: list[str] = []

        if not file_path.exists() or not file_path.is_file():
            warnings.append("attachment file does not exist or is not a file")
            return self._result(
                context=format_attachment_context(
                    file_name=file_name,
                    file_path=file_path,
                    extension=extension,
                    content="",
                    warnings=warnings,
                ),
                used=False,
                file_path=file_path,
                extension=extension,
                warnings=warnings,
            )

        result = self.read_file(question=question, file_path=file_path, extension=extension)
        content = truncate_text(result.content, self.config.max_text_chars)
        context = format_attachment_context(
            file_name=file_name,
            file_path=file_path,
            extension=extension,
            content=content,
            warnings=result.warnings,
        )
        return self._result(
            context=context,
            used=True,
            file_path=file_path,
            extension=extension,
            warnings=result.warnings,
            reader=result.reader,
        )

    def read_file(
        self,
        *,
        question: str,
        file_path: Path,
        extension: str | None = None,
        depth: int = 0,
    ) -> AttachmentReadResult:
        """
        負責執行 AttachmentEvidenceBuilder 中的 read_file 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            file_path: 要讀取或寫入的檔案或目錄路徑。
            extension: 此流程需要使用的輸入資料。
            depth: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 AttachmentReadResult。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        extension = (extension or file_path.suffix).lower()
        try:
            if extension in TEXT_EXTENSIONS:
                return AttachmentReadResult(True, "text_reader", self.text_reader.read(file_path, extension))
            if extension == ".docx":
                return AttachmentReadResult(True, "docx_reader", self.office_reader.read_docx(file_path))
            if extension == ".pptx":
                return AttachmentReadResult(True, "pptx_reader", self.office_reader.read_pptx(file_path))
            if extension == ".xlsx":
                return AttachmentReadResult(True, "excel_reader", self.excel_reader.read_xlsx(question, file_path))
            if extension == ".xls":
                return AttachmentReadResult(
                    True, "pandas_xls_reader", self.excel_reader.read_xls(question, file_path)
                )
            if extension == ".pdf":
                return AttachmentReadResult(True, "pdf_reader", self.office_reader.read_pdf(file_path))
            if extension == ".py":
                return AttachmentReadResult(True, "python_reader", self.code_reader.read_python(file_path))
            if extension == ".zip":
                return AttachmentReadResult(
                    True,
                    "zip_reader",
                    self.archive_reader.read_zip(question, file_path, depth=depth),
                )
            if extension in IMAGE_EXTENSIONS:
                return AttachmentReadResult(
                    True, "ollama_vision_reader", self.media_reader.read_image(question, file_path)
                )
            if extension in AUDIO_EXTENSIONS:
                return AttachmentReadResult(
                    True,
                    "faster_whisper_audio_reader",
                    self.media_reader.analyze_audio(question, file_path),
                )

            return AttachmentReadResult(
                False,
                "unsupported_reader",
                f"Unsupported attachment type: {extension or '(none)'}",
                ["unsupported attachment extension"],
            )
        except Exception as exc:
            return AttachmentReadResult(
                False,
                "error_reader",
                "",
                [f"attachment read failed: {type(exc).__name__}: {exc}"],
            )

    def format_nested_file_result(
        self,
        *,
        question: str,
        file_path: Path,
        display_name: str,
        depth: int,
    ) -> str:
        """
        負責執行 AttachmentEvidenceBuilder 中的 format_nested_file_result 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            file_path: 要讀取或寫入的檔案或目錄路徑。
            display_name: 評估、推理或工具執行後產生的結果與分數資料。
            depth: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        extension = file_path.suffix.lower()
        result = self.read_file(
            question=question,
            file_path=file_path,
            extension=extension,
            depth=depth,
        )
        header = f"Nested reader for {display_name} ({extension or '(none)'})"
        content = result.content
        if result.warnings:
            content = "Warnings:\n" + "\n".join(f"- {warning}" for warning in result.warnings)
        return f"{header}\n{truncate_text(content, self.config.max_text_chars)}"

    def _empty(self) -> dict[str, Any]:
        """
        負責執行 AttachmentEvidenceBuilder 中的 _empty 流程，依照 AttachmentEvidenceBuilder 的流程需求處理 _empty 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {"context": "", "used": False, "tool_usage": [], "metadata": {}}

    def _result(
        self,
        *,
        context: str,
        used: bool,
        file_path: Path,
        extension: str,
        warnings: list[str],
        reader: str = "attachment_reader",
    ) -> dict[str, Any]:
        """
        負責執行 AttachmentEvidenceBuilder 中的 _result 流程，依照 AttachmentEvidenceBuilder 的流程需求處理 _result 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            context: 目前流程所需的上下文、狀態或附加資訊。
            used: 評估、推理或工具執行後產生的結果與分數資料。
            file_path: 要讀取或寫入的檔案或目錄路徑。
            extension: 評估、推理或工具執行後產生的結果與分數資料。
            warnings: 評估、推理或工具執行後產生的結果與分數資料。
            reader: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        failed = any("failed" in warning for warning in warnings)
        output_text = context if used else "\n".join(warnings)
        return {
            "context": context if used else "",
            "used": used,
            "tool_usage": [
                {
                    "ok": used and not failed,
                    "tool_name": "attachment_reader",
                    "output_text": output_text,
                    "raw_result": {
                        "file_path": str(file_path),
                        "file_type": extension,
                        "reader": reader,
                        "warnings": warnings,
                    },
                    "error": "; ".join(warnings) if warnings and (failed or not used) else None,
                }
            ],
            "metadata": {
                "file_path": str(file_path),
                "file_type": extension,
                "reader": reader,
                "warnings": warnings,
            },
        }
