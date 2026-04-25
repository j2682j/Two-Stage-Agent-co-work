from __future__ import annotations

from typing import Any

from .builder import PromptBuilder, PromptPacket


class RepairPromptBuilder(PromptBuilder):
    """建立 stage1 repair prompt。"""

    def gather(self, **kwargs) -> list[PromptPacket]:
        expected_weight_count = kwargs.get("expected_weight_count", 0)
        return [
            PromptPacket(
                content=str(expected_weight_count),
                packet_type="expected_weight_count",
                priority=10.0,
                metadata={"expected_weight_count": int(expected_weight_count)},
            )
        ]

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        return packets

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        packet = packets[0] if packets else None
        expected_weight_count = 0
        if packet is not None:
            expected_weight_count = int(packet.metadata.get("expected_weight_count", 0))
        return {"expected_weight_count": expected_weight_count}

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        return structured

    def render(self, compressed: dict[str, Any], **kwargs):
        expected_weight_count = int(compressed.get("expected_weight_count", 0))
        return f"""
Your previous reply did not follow the required output format.

Return plain text only.
Do not include markdown fences.
Do not include any extra text outside the required format.

Required format:
REASONING=<brief key steps and correction checks only>
FINAL_ANSWER=<your final answer as a string>
WEIGHTS=[w1, w2, ..., w{expected_weight_count}]

Requirements:
- REASONING must be a short string with only the essential correction/checking steps.
- FINAL_ANSWER must be a string, even if the answer is numeric.
- WEIGHTS must contain exactly {expected_weight_count} integers.
- If there are no previous agents, WEIGHTS must be [].
- If previous agents made mistakes, correct them briefly in REASONING before giving FINAL_ANSWER.
- The WEIGHTS line must be the final line of your reply.
        """.strip()
