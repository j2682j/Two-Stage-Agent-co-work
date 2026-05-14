from __future__ import annotations

import csv
import json
from pathlib import Path

from ..formatters import format_rows
from ..models import AttachmentReaderConfig


TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".jsonld", ".xml", ".pdb"}


class TextAttachmentReader:
    """
    負責在 builder.attachment.readers.text_reader 中封裝 TextAttachmentReader，封裝附件讀取與內容萃取流程，將檔案轉成可推理的證據。
    
    Args:
        config: 控制此流程行為的設定資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, config: AttachmentReaderConfig) -> None:
        """
        負責執行 TextAttachmentReader 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.config = config

    def read(self, file_path: Path, extension: str) -> str:
        """
        負責執行 TextAttachmentReader 中的 read 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
        
        Args:
            file_path: 要讀取或寫入的檔案或目錄路徑。
            extension: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if extension == ".csv":
            return self._read_csv(file_path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if extension in {".json", ".jsonld"}:
            try:
                parsed = json.loads(text)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                return text
        return text

    def _read_csv(self, file_path: Path) -> str:
        """
        負責執行 TextAttachmentReader 中的 _read_csv 流程，依照 TextAttachmentReader 的流程需求處理 _read_csv 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            file_path: 要讀取或寫入的檔案或目錄路徑。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        rows: list[list[str]] = []
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for idx, row in enumerate(reader):
                if idx >= self.config.max_table_rows:
                    break
                rows.append([str(cell) for cell in row])
        if not rows:
            return "(empty csv)"
        return format_rows(
            "CSV",
            rows,
            truncated=len(rows) >= self.config.max_table_rows,
            max_rows=self.config.max_table_rows,
        )
