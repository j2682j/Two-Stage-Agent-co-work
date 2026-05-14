from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..models import AttachmentReaderConfig


class CodeAttachmentReader:
    """
    負責在 builder.attachment.readers.code_reader 中封裝 CodeAttachmentReader，封裝附件讀取與內容萃取流程，將檔案轉成可推理的證據。
    
    Args:
        config: 控制此流程行為的設定資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, config: AttachmentReaderConfig) -> None:
        """
        負責執行 CodeAttachmentReader 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.config = config

    def read_python(self, file_path: Path) -> str:
        """
        負責執行 CodeAttachmentReader 中的 read_python 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
        
        Args:
            file_path: 要讀取或寫入的檔案或目錄路徑。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        source = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            completed = subprocess.run(
                [sys.executable, str(file_path)],
                cwd=str(file_path.parent),
                capture_output=True,
                text=True,
                timeout=self.config.python_timeout,
            )
            execution = (
                f"exit_code={completed.returncode}\n"
                f"stdout:\n{completed.stdout.strip() or '(empty)'}\n"
                f"stderr:\n{completed.stderr.strip() or '(empty)'}"
            )
        except subprocess.TimeoutExpired:
            execution = f"execution timed out after {self.config.python_timeout} seconds"
        return f"Python source:\n{source}\n\nExecution result:\n{execution}"
