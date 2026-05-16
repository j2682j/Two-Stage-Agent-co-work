from __future__ import annotations

from typing import Any

from .builder import PromptBuilder, PromptPacket
from .contracts import resolve_prompt_contract

class Stage1PromptBuilder(PromptBuilder):
    """
    負責在 prompt.stage1_prompt_builder 中封裝 Stage1PromptBuilder，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def gather(self, **kwargs) -> list[PromptPacket]:
        """
        負責執行 Stage1PromptBuilder 中的 gather 流程，依照 Stage1PromptBuilder 的流程需求處理 gather 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        question = self._normalize_text(kwargs.get("question", ""))
        formers = kwargs.get("formers", []) or []
        tool_context = str(kwargs.get("tool_context", "") or "").strip()
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
        """
        負責執行 Stage1PromptBuilder 中的 select 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 Stage1PromptBuilder 中的 structure 流程，依照 Stage1PromptBuilder 的流程需求處理 structure 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 Stage1PromptBuilder 中的 compress 流程，依照 Stage1PromptBuilder 的流程需求處理 compress 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            structured: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 Stage1PromptBuilder 中的 render 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            compressed: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        question = compressed["question"]
        reflection_context = compressed["reflection_context"]
        tool_context = compressed["tool_context"]
        formers = compressed["formers"]
        contract = kwargs.get("contract") or resolve_prompt_contract(
            kwargs.get("task_context"),
            question=question,
        )

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

            content += "\n\nThere are no previous agent answers for this question.\n\n"
            content += contract.stage1_output_contract(
                expected_weight_count=0,
                has_formers=False,
            )
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

        prefix_string += "\n\n" + contract.stage1_output_contract(
            expected_weight_count=len(formers),
            has_formers=True,
        )

        return {"role": "user", "content": prefix_string}
