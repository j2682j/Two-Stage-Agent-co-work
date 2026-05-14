from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from utils.network_utils import normalize_text

DEFAULT_SYSTEM_PROMPT = (
    "You are a reliable and careful AI assistant. "
    "Follow the required format exactly. "
    "Do not add extra text outside the required format. "
    "Keep reasoning concise."
)

DEFAULT_STAGE2_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + " Use tool evidence when it is relevant. "
    + "If tool evidence is not useful, solve the question with your own reasoning. "
    + "Return only the required structured output."
)


@dataclass
class PromptPacket:
    """
    負責在 prompt.builder 中封裝 PromptPacket，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    content: str
    packet_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0
    token_count: int = 0

    def __post_init__(self):
        """
        負責執行 PromptPacket 中的 __post_init__ 流程，依照 PromptPacket 的流程需求處理 __post_init__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.token_count <= 0:
            # 使用簡單估算，避免 packet 缺少 token_count。
            self.token_count = max(1, len(self.content) // 4)


@dataclass
class PromptBuildConfig:
    """
    負責在 prompt.builder 中封裝 PromptBuildConfig，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    max_formers: int = 3
    max_tool_lines: int = 80
    max_tool_chars: int = 6000
    max_reasoning_chars: int = 220
    max_candidate_reasoning_chars: int = 160
    short_answer_max_chars: int = 80
    include_tool_evidence: bool = True
    include_importance: bool = True


class PromptBuilder:
    """
    負責在 prompt.builder 中封裝 PromptBuilder，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        config: 控制此流程行為的設定資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, config: PromptBuildConfig | None = None):
        """
        負責執行 PromptBuilder 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.config = config or PromptBuildConfig()

    def build(self, **kwargs):
        """
        負責執行 PromptBuilder 中的 build 流程，組裝提示詞內容，將任務、記憶、證據或格式要求整理成模型可讀的輸入。
        
        Args:
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        packets = self.gather(**kwargs)
        selected = self.select(packets, **kwargs)
        structured = self.structure(selected, **kwargs)
        compressed = self.compress(structured, **kwargs)
        return self.render(compressed, **kwargs)

    def gather(self, **kwargs) -> list[PromptPacket]:
        """
        負責執行 PromptBuilder 中的 gather 流程，依照 PromptBuilder 的流程需求處理 gather 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        """
        負責執行 PromptBuilder 中的 select 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        """
        負責執行 PromptBuilder 中的 structure 流程，依照 PromptBuilder 的流程需求處理 structure 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        負責執行 PromptBuilder 中的 compress 流程，依照 PromptBuilder 的流程需求處理 compress 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            structured: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    def render(self, compressed: dict[str, Any], **kwargs):
        """
        負責執行 PromptBuilder 中的 render 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            compressed: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    def _normalize_text(self, text: Any) -> str:
        """
        負責執行 PromptBuilder 中的 _normalize_text 流程，依照 PromptBuilder 的流程需求處理 _normalize_text 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return normalize_text(text)

    def _extract_keywords(self, text: str) -> set[str]:
        """
        負責執行 PromptBuilder 中的 _extract_keywords 流程，依照 PromptBuilder 的流程需求處理 _extract_keywords 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 set[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
            "at", "for", "and", "or", "that", "this", "it", "as", "with", "by",
            "from", "your", "their", "into", "then", "than", "will", "would",
            "should", "could", "how", "what", "when", "where", "why", "use",
            "using", "answer", "final", "question",
        }
        tokens = re.findall(r"[A-Za-z0-9_./:-]+", text.lower())
        return {token for token in tokens if len(token) > 2 and token not in stopwords}

    def _truncate_sentences(self, text: str, max_chars: int) -> str:
        """
        負責執行 PromptBuilder 中的 _truncate_sentences 流程，依照 PromptBuilder 的流程需求處理 _truncate_sentences 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
            max_chars: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        kept: list[str] = []
        current_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_len + len(sentence) > max_chars and kept:
                break
            kept.append(sentence)
            current_len += len(sentence) + 1

        if not kept:
            return normalized[:max_chars].rstrip()
        return " ".join(kept).strip()

    def _compress_multiline_text(self, text: str, max_lines: int, max_chars: int) -> str:
        """
        負責執行 PromptBuilder 中的 _compress_multiline_text 流程，依照 PromptBuilder 的流程需求處理 _compress_multiline_text 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
            max_lines: 控制檢索、篩選或輸出數量的數值參數。
            max_chars: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raw = "" if text is None else str(text).strip()
        if not raw or raw == "No tool result available.":
            return ""

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        kept = lines[:max_lines]
        compressed = "\n".join(kept).strip()
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars].rstrip() + " ..."
        return compressed
