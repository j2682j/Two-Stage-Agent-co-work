from __future__ import annotations

from typing import Any

from .builder import PromptBuilder, PromptPacket

class Stage1PromptBuilder(PromptBuilder):
    def gather(self, **kwargs) -> list[PromptPacket]:
        question = self._normalize_text(kwargs.get("question", ""))
        formers = kwargs.get("formers", []) or []
        tool_context = self._normalize_text(kwargs.get("tool_context", ""))
        reflection_context = self._normalize_text(kwargs.get("reflection_context", ""))

        packets = [
            PromptPacket(content=question, packet_type="question", priority=10.0),
        ]

        if reflection_context:
            packets.append(
                PromptPacket(content=reflection_context, packet_type="reflection_context", priority=9.0)
            )

        if self.config.include_tool_evidence and tool_context and tool_context != "No tool result available.":
            packets.append(
                PromptPacket(content=tool_context, packet_type="tool_context", priority=8.0)
            )

        for idx, former in enumerate(formers):
            reasoning = self._normalize_text(former.get("reasoning", ""))
            final_answer = self._normalize_text(former.get("final_answer", ""))
            content = f"reasoning={reasoning}\nfinal_answer={final_answer}".strip()
            packets.append(
                PromptPacket(
                    content=content,
                    packet_type="former",
                    priority=5.0,
                    metadata={"former_index": idx, "reasoning": reasoning, "final_answer": final_answer},
                )
            )

        return packets

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        question_packet = next((p for p in packets if p.packet_type == "question"), None)
        question_keywords = self._extract_keywords(question_packet.content if question_packet else "")

        selected: list[PromptPacket] = []
        former_packets: list[tuple[float, PromptPacket]] = []

        for packet in packets:
            if packet.packet_type == "former":
                reasoning = packet.metadata.get("reasoning", "")
                final_answer = packet.metadata.get("final_answer", "")
                overlap = len(
                    question_keywords
                    & (self._extract_keywords(reasoning) | self._extract_keywords(final_answer))
                )
                score = overlap
                if final_answer:
                    score += 1.0
                if reasoning:
                    score += min(len(reasoning.split()), 40) / 40.0
                former_packets.append((score, packet))
            else:
                selected.append(packet)

        former_packets.sort(key=lambda item: item[0], reverse=True)
        selected.extend(packet for _, packet in former_packets[: self.config.max_formers])
        return selected

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        question = ""
        tool_context = ""
        reflection_context = ""
        formers: list[dict[str, str]] = []

        for packet in packets:
            if packet.packet_type == "question":
                question = packet.content
            elif packet.packet_type == "reflection_context":
                reflection_context = packet.content
            elif packet.packet_type == "tool_context":
                tool_context = packet.content
            elif packet.packet_type == "former":
                formers.append(
                    {
                        "reasoning": packet.metadata.get("reasoning", ""),
                        "final_answer": packet.metadata.get("final_answer", ""),
                    }
                )

        return {
            "question": question,
            "reflection_context": reflection_context,
            "tool_context": tool_context,
            "formers": formers,
        }

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        compressed_formers = []
        for former in structured["formers"]:
            compressed_formers.append(
                {
                    "reasoning": self._truncate_sentences(
                        former.get("reasoning", ""),
                        self.config.max_reasoning_chars,
                    ),
                    "final_answer": self._normalize_text(former.get("final_answer", "")),
                }
            )

        return {
            "question": structured["question"],
            "reflection_context": self._compress_multiline_text(
                structured["reflection_context"],
                max_lines=4,
                max_chars=360,
            ),
            "tool_context": self._compress_multiline_text(
                structured["tool_context"],
                self.config.max_tool_lines,
                self.config.max_tool_chars,
            ),
            "formers": compressed_formers,
        }

    def render(self, compressed: dict[str, Any], **kwargs):
        question = compressed["question"]
        reflection_context = compressed["reflection_context"]
        tool_context = compressed["tool_context"]
        formers = compressed["formers"]

        if len(formers) == 0:
            content = f"""
                You are solving the following question.

                Question:
                {question}
            """.strip()

            if reflection_context:
                content += f"\n\nRelevant reflection rules:\n{reflection_context}"

            if tool_context:
                content += f"\n\nAvailable tool evidence:\n{tool_context}"

            content += """

                There are no previous agent answers for this question.

                Task:
                1. Solve the question carefully.
            2. Check whether your first answer could be wrong.
            3. Keep the reasoning short and focused on the key checks.
            4. Make sure the final answer uses exactly the unit requested in the question.
            5. If relevant reflection rules apply, use them as compact error checks rather than as answer lookup.

                Return plain text only in exactly this format:
                REASONING=<brief key steps and self-checks only>
                FINAL_ANSWER=<your final answer>
                WEIGHTS=[]

                Rules:
                - REASONING must be short and only include the essential checks.
                - FINAL_ANSWER must contain only your final answer.
                - WEIGHTS must be [] because there are no previous agents.
                - The WEIGHTS line must be the final line of your reply.
                - Do not include markdown fences or any extra text outside the required format.
            """
            content = content.rstrip()
            return {"role": "user", "content": content}

        prefix_string = f"""
            You are solving the following question.

            Question:
            {question}
        """.strip()

        if reflection_context:
            prefix_string += f"\n\nRelevant reflection rules:\n{reflection_context}"

        if tool_context:
            prefix_string += f"\n\nAvailable tool evidence:\n{tool_context}"

        prefix_string += f"""

            These are the stage-1 reasoning traces from {len(formers)} previous agents:
        """.strip()

        for aid, former in enumerate(formers, 1):
            previous_reasoning = former.get("reasoning", "")
            previous_final_answer = former.get("final_answer", "")
            prefix_string += f"""

                Previous Agent {aid} reasoning:{previous_reasoning}""".rstrip()
            if previous_final_answer:
                prefix_string += f"""

                Previous Agent {aid} final answer:{previous_final_answer}""".rstrip()

        prefix_string += f"""

            Task:
            1. First inspect the previous agents' reasoning steps.
            2. Identify the most important mistake, weak assumption, or wrong calculation.
            3. Be skeptical of both the previous answers and your own first instinct.
            4. Correct the most important issue before producing your answer.
            5. Give your own short reasoning and final answer.
            6. Make sure the final answer uses exactly the unit requested in the question.
            7. If relevant reflection rules apply, use them as compact error checks rather than as answer lookup.

            Important rules:
            - Do not blindly follow previous agents.
            - If previous agents are wrong, explicitly correct them in your own reasoning.
            - Keep your own reasoning concise.
            - In stage 1, you must still give a final answer.
            - If needed, convert the result before giving the final answer.

            Return plain text only in exactly this format:
            REASONING=<brief key steps and correction checks only>
            FINAL_ANSWER=<your final answer>
            WEIGHTS=[w1, w2, ..., w{len(formers)}]

            Rules:
            - REASONING must be short and include only the most important checking/correction steps.
            - FINAL_ANSWER must contain only your final answer.
            - WEIGHTS must contain exactly {len(formers)} integers.
            - Each weight corresponds to the previous agents in the same order shown above.
            - Each weight must be an integer between 1 and 5.
            - If a previous answer is poor, give it a lower score.
            - The WEIGHTS line must be the final line of your reply.
            - Do not include markdown fences or any extra text outside the required format.
        """.strip()

        return {"role": "user", "content": prefix_string}
