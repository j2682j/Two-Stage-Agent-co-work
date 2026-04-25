from typing import Any

from .builder import PromptBuilder, PromptPacket

class RankingPromptBuilder(PromptBuilder):
    def gather(self, **kwargs) -> list[PromptPacket]:
        question = self._normalize_text(kwargs.get("question", ""))
        responses = kwargs.get("responses", []) or []

        packets = [PromptPacket(content=question, packet_type="question", priority=10.0)]
        for idx, response in enumerate(responses):
            packets.append(
                PromptPacket(
                    content=self._normalize_text(response),
                    packet_type="candidate",
                    priority=5.0,
                    metadata={"candidate_index": idx},
                )
            )
        return packets

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        return packets

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        question = ""
        candidates: list[dict[str, Any]] = []
        for packet in packets:
            if packet.packet_type == "question":
                question = packet.content
            elif packet.packet_type == "candidate":
                candidates.append(
                    {
                        "candidate_index": packet.metadata["candidate_index"],
                        "content": packet.content,
                    }
                )
        return {
            "question": question,
            "candidates": candidates,
        }

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        compressed_candidates = []
        for candidate in structured["candidates"]:
            compressed_candidates.append(
                {
                    "candidate_index": candidate["candidate_index"],
                    "content": self._truncate_sentences(
                        candidate["content"],
                        self.config.max_candidate_reasoning_chars,
                    ),
                }
            )
        return {
            "question": structured["question"],
            "candidates": compressed_candidates,
        }

    def render(self, compressed: dict[str, Any], **kwargs):
        prefix_string = (
            "You are ranking candidate solutions to the same question.\n\n"
            "Question:\n"
            + compressed["question"]
            + "\n\nThese are the candidate solutions from other agents:"
        )

        for item in compressed["candidates"]:
            response = "\n\nAgent solution " + str(item["candidate_index"] + 1) + ": ```{}```".format(item["content"])
            prefix_string += response

        prefix_string += """

            Your goal is to select the 2 most reliable solutions.

            Priority:
            1. Final-answer correctness.
            2. Whether the reasoning supports the final answer.
            3. Whether the response directly solves the question.
            4. Prefer concise and reliable solutions over verbose but weak ones.

            Return only:
            [i, j]
        """.strip()

        return {"role": "user", "content": prefix_string}
