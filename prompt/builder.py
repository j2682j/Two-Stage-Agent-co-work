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
    """Prompt 資訊包，供 Gather/Select/Structure/Compress 階段傳遞。"""

    content: str
    packet_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0
    token_count: int = 0

    def __post_init__(self):
        if self.token_count <= 0:
            # 使用簡單估算，避免 packet 缺少 token_count。
            self.token_count = max(1, len(self.content) // 4)


@dataclass
class PromptBuildConfig:
    """Prompt 建構設定。"""

    max_formers: int = 3
    max_tool_lines: int = 6
    max_tool_chars: int = 600
    max_reasoning_chars: int = 220
    max_candidate_reasoning_chars: int = 160
    short_answer_max_chars: int = 80
    include_tool_evidence: bool = True
    include_importance: bool = True


class PromptBuilder:
    """Prompt 建構器基底類別，統一實作 GSSC 流程。"""

    def __init__(self, config: PromptBuildConfig | None = None):
        self.config = config or PromptBuildConfig()

    def build(self, **kwargs):
        packets = self.gather(**kwargs)
        selected = self.select(packets, **kwargs)
        structured = self.structure(selected, **kwargs)
        compressed = self.compress(structured, **kwargs)
        return self.render(compressed, **kwargs)

    def gather(self, **kwargs) -> list[PromptPacket]:
        raise NotImplementedError

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        raise NotImplementedError

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, compressed: dict[str, Any], **kwargs):
        raise NotImplementedError

    def _normalize_text(self, text: Any) -> str:
        return normalize_text(text)

    def _extract_keywords(self, text: str) -> set[str]:
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
        normalized = self._normalize_text(text)
        if not normalized or normalized == "No tool result available.":
            return ""

        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        kept = lines[:max_lines]
        compressed = "\n".join(kept).strip()
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars].rstrip() + " ..."
        return compressed
