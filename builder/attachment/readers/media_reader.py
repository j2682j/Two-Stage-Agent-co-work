from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from ..models import AttachmentReaderConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}


class MediaAttachmentReader:
    """
    負責在 builder.attachment.readers.media_reader 中封裝 MediaAttachmentReader，封裝附件讀取與內容萃取流程，將檔案轉成可推理的證據。
    
    Args:
        config: 控制此流程行為的設定資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, config: AttachmentReaderConfig) -> None:
        """
        負責執行 MediaAttachmentReader 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.config = config

    def read_image(self, question: str, file_path: Path) -> str:
        """
        負責執行 MediaAttachmentReader 中的 read_image 流程，讀取圖片附件並轉成可放入證據或模型上下文的文字描述。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            file_path: 要讀取或寫入的檔案或目錄路徑。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        image_bytes = file_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        endpoint = self._ollama_chat_endpoint()
        prompt = (
            "You are extracting evidence from an image attachment for a GAIA benchmark question.\n"
            "Read the image carefully. Return concise structured text only.\n"
            "Include:\n"
            "- visible text/OCR, if any\n"
            "- important objects, layout, chart/table values, board positions, symbols, numbers, colors, and labels\n"
            "- facts directly relevant to the question\n"
            "- uncertainties if the image is ambiguous\n\n"
            f"Question:\n{question}"
        )
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.vision_timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama vision HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama vision request failed: {exc.reason}") from exc

        data = json.loads(raw)
        message = data.get("message") or {}
        content = str(message.get("content", "") or "").strip()
        if not content:
            content = str(data.get("response", "") or "").strip()
        if not content:
            raise RuntimeError("Ollama vision response did not include text content")

        return f"Ollama vision model: {self.config.vision_model}\n{content}"

    def _ollama_chat_endpoint(self) -> str:
        """
        負責執行 MediaAttachmentReader 中的 _ollama_chat_endpoint 流程，依照 MediaAttachmentReader 的流程需求處理 _ollama_chat_endpoint 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        base_url = (
            os.getenv("OLLAMA_NATIVE_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        ).strip()
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url.rstrip("/") + "/api/chat"

    def analyze_audio(self, question: str, file_path: Path) -> str:
        """
        負責執行 MediaAttachmentReader 中的 analyze_audio 流程，依照 MediaAttachmentReader 的流程需求處理 analyze_audio 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            file_path: 要讀取或寫入的檔案或目錄路徑。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed in the Python environment running this evaluation"
            ) from exc

        model = WhisperModel(
            self.config.audio_model_size,
            device=self.config.audio_device,
            compute_type=self.config.audio_compute_type,
        )
        segments, info = model.transcribe(
            str(file_path),
            beam_size=5,
            vad_filter=True,
        )

        lines: list[str] = []
        for segment in segments:
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                continue
            start = float(getattr(segment, "start", 0.0) or 0.0)
            end = float(getattr(segment, "end", 0.0) or 0.0)
            lines.append(f"[{start:.2f}-{end:.2f}] {text}")

        language = str(getattr(info, "language", "") or "unknown")
        probability = getattr(info, "language_probability", None)
        probability_text = ""
        if probability is not None:
            try:
                probability_text = f" confidence={float(probability):.2f}"
            except Exception:
                probability_text = f" confidence={probability}"

        transcript = "\n".join(lines).strip() or "(empty transcription)"
        return (
            "Audio transcription:\n"
            f"- faster_whisper_model: {self.config.audio_model_size}\n"
            f"- device: {self.config.audio_device}\n"
            f"- compute_type: {self.config.audio_compute_type}\n"
            f"- detected_language: {language}{probability_text}\n"
            f"- question_focus: {question}\n"
            "Transcript:\n"
            f"{transcript}"
        )
