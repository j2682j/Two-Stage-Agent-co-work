from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import AttachmentReaderConfig

if TYPE_CHECKING:
    from ..evidence_builder import AttachmentEvidenceBuilder


class ArchiveAttachmentReader:
    """
    負責在 builder.attachment.readers.archive_reader 中封裝 ArchiveAttachmentReader，封裝附件讀取與內容萃取流程，將檔案轉成可推理的證據。
    
    Args:
        config: 控制此流程行為的設定資料。
        dispatcher: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(
        self,
        config: AttachmentReaderConfig,
        dispatcher: AttachmentEvidenceBuilder,
    ) -> None:
        """
        負責執行 ArchiveAttachmentReader 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
            dispatcher: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.config = config
        self.dispatcher = dispatcher

    def read_zip(self, question: str, file_path: Path, depth: int = 0) -> str:
        """
        負責執行 ArchiveAttachmentReader 中的 read_zip 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            file_path: 要讀取或寫入的檔案或目錄路徑。
            depth: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if depth > self.config.max_zip_depth:
            return f"ZIP archive skipped because nested zip depth exceeded {self.config.max_zip_depth}."

        with zipfile.ZipFile(file_path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            listed_names = [info.filename for info in infos[:100]]
            selected_infos, warnings = self._select_safe_zip_members(infos)

            sections = [
                "ZIP archive:",
                f"- file_count: {len(infos)}",
                f"- processed_file_count: {len(selected_infos)}",
                "ZIP contents:",
                *[f"- {name}" for name in listed_names],
            ]
            if len(infos) > len(listed_names):
                sections.append(f"- [showing first {len(listed_names)} entries]")
            if warnings:
                sections.append("ZIP warnings:")
                sections.extend(f"- {warning}" for warning in warnings)

            if not selected_infos:
                return "\n".join(sections)

            with tempfile.TemporaryDirectory(prefix="gaia_attachment_zip_") as tmp_dir:
                tmp_root = Path(tmp_dir).resolve()
                extracted_sections: list[str] = []
                for info in selected_infos:
                    try:
                        extracted_path = self._extract_zip_member_safely(
                            archive=archive,
                            info=info,
                            tmp_root=tmp_root,
                        )
                        extracted_content = self.dispatcher.format_nested_file_result(
                            question=question,
                            file_path=extracted_path,
                            display_name=info.filename,
                            depth=depth + 1,
                        )
                    except Exception as exc:
                        extracted_content = f"[ERROR] {type(exc).__name__}: {exc}"

                    extracted_sections.append(
                        f"Extracted file: {info.filename}\n"
                        f"- size_bytes: {info.file_size}\n"
                        f"{extracted_content}".strip()
                    )

            if extracted_sections:
                sections.append("Extracted file evidence:")
                sections.append("\n\n".join(extracted_sections))
            return "\n".join(sections)

    def _select_safe_zip_members(
        self, infos: list[zipfile.ZipInfo]
    ) -> tuple[list[zipfile.ZipInfo], list[str]]:
        """
        負責執行 ArchiveAttachmentReader 中的 _select_safe_zip_members 流程，依照 ArchiveAttachmentReader 的流程需求處理 _select_safe_zip_members 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            infos: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 tuple[list[zipfile.ZipInfo], list[str]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        selected: list[zipfile.ZipInfo] = []
        warnings: list[str] = []
        total_size = 0

        for info in infos:
            if len(selected) >= self.config.max_zip_members:
                warnings.append(
                    f"skipped remaining files after max_zip_members={self.config.max_zip_members}"
                )
                break

            normalized_name = str(info.filename or "").replace("\\", "/")
            if self._is_unsafe_zip_name(normalized_name):
                warnings.append(f"skipped unsafe zip path: {info.filename}")
                continue
            if info.file_size > self.config.max_zip_file_bytes:
                warnings.append(
                    f"skipped {info.filename}: file_size={info.file_size} exceeds "
                    f"max_zip_file_bytes={self.config.max_zip_file_bytes}"
                )
                continue
            if total_size + info.file_size > self.config.max_zip_total_bytes:
                warnings.append(
                    f"skipped {info.filename}: total extracted bytes would exceed "
                    f"max_zip_total_bytes={self.config.max_zip_total_bytes}"
                )
                continue

            selected.append(info)
            total_size += info.file_size

        return selected, warnings

    def _is_unsafe_zip_name(self, name: str) -> bool:
        """
        負責執行 ArchiveAttachmentReader 中的 _is_unsafe_zip_name 流程，依照 ArchiveAttachmentReader 的流程需求處理 _is_unsafe_zip_name 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        normalized = name.replace("\\", "/")
        return (
            not normalized
            or normalized.startswith("/")
            or ":" in normalized.split("/", 1)[0]
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
        )

    def _extract_zip_member_safely(
        self,
        *,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        tmp_root: Path,
    ) -> Path:
        """
        負責執行 ArchiveAttachmentReader 中的 _extract_zip_member_safely 流程，依照 ArchiveAttachmentReader 的流程需求處理 _extract_zip_member_safely 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            archive: 此流程需要使用的輸入資料。
            info: 此流程需要使用的輸入資料。
            tmp_root: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Path。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        normalized_name = str(info.filename or "").replace("\\", "/")
        if self._is_unsafe_zip_name(normalized_name):
            raise ValueError(f"unsafe zip path: {info.filename}")

        target_path = (tmp_root / normalized_name).resolve()
        try:
            target_path.relative_to(tmp_root)
        except ValueError as exc:
            raise ValueError(f"zip path escapes extraction root: {info.filename}") from exc

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target_path.open("wb") as target:
            copied = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > self.config.max_zip_file_bytes:
                    raise ValueError(
                        f"zip member exceeded max_zip_file_bytes={self.config.max_zip_file_bytes}: "
                        f"{info.filename}"
                    )
                target.write(chunk)
        return target_path
