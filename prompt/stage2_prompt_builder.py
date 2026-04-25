from typing import Any

from .builder import PromptBuilder, PromptPacket


class Stage2PromptBuilder(PromptBuilder):
    def gather(self, **kwargs) -> list[PromptPacket]:
        packets = [
            PromptPacket(content=self._normalize_text(kwargs.get("question", "")), packet_type="question", priority=10.0),
            PromptPacket(content=self._normalize_text(kwargs.get("stage1_result", "")), packet_type="stage1_result", priority=8.0),
        ]

        importance = kwargs.get("importance", None)
        if self.config.include_importance and importance is not None:
            packets.append(
                PromptPacket(content=self._normalize_text(importance), packet_type="importance", priority=6.0)
            )

        tool_context = self._normalize_text(kwargs.get("tool_context", ""))
        if self.config.include_tool_evidence and tool_context and tool_context != "No tool result available.":
            packets.append(
                PromptPacket(content=tool_context, packet_type="tool_context", priority=9.0)
            )

        return packets

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        return packets

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        structured = {
            "question": "",
            "stage1_result": "",
            "importance": "",
            "tool_context": "",
        }
        for packet in packets:
            structured[packet.packet_type] = packet.content
        return structured

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        structured["tool_context"] = self._compress_multiline_text(
            structured["tool_context"],
            self.config.max_tool_lines,
            self.config.max_tool_chars,
        )
        return structured

    def render(self, compressed: dict[str, Any], **kwargs):
        user_prompt = f"""
            Question:
            {compressed["question"]}

            Stage-1 answer:
            {compressed["stage1_result"]}

            Agent importance:
            {compressed["importance"] if compressed["importance"] else ""}

            Additional tool evidence:
            {compressed["tool_context"] if compressed["tool_context"] else "No tool result available."}

            Task:
            1. Solve the question again using the tool evidence if it helps.
            2. Use the most relevant tool evidence instead of repeating everything.
            3. Before giving the final answer, verify that the answer unit matches the unit requested in the question.
            4. If needed, convert the result before giving the final answer.

            Return JSON only with:
            {{
              "reasoning": "your reasoning",
              "final_answer": "your final answer"
            }}
        """.strip()

        return [
            {"role": "system", "content": kwargs.get("system_prompt", "")},
            {"role": "user", "content": user_prompt},
        ]
