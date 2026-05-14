from typing import Any

from .builder import PromptBuilder, PromptPacket


class Stage2PromptBuilder(PromptBuilder):
    """
    負責在 prompt.stage2_prompt_builder 中封裝 Stage2PromptBuilder，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def gather(self, **kwargs) -> list[PromptPacket]:
        """
        負責執行 Stage2PromptBuilder 中的 gather 流程，依照 Stage2PromptBuilder 的流程需求處理 gather 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        packets = [
            PromptPacket(content=self._normalize_text(kwargs.get("question", "")), packet_type="question", priority=10.0),
            PromptPacket(content=self._normalize_text(kwargs.get("stage1_result", "")), packet_type="stage1_result", priority=8.0),
        ]

        importance = kwargs.get("importance", None)
        if self.config.include_importance and importance is not None:
            packets.append(
                PromptPacket(content=self._normalize_text(importance), packet_type="importance", priority=6.0)
            )

        tool_context = str(kwargs.get("tool_context", "") or "").strip()
        if self.config.include_tool_evidence and tool_context and tool_context != "No tool result available.":
            packets.append(
                PromptPacket(content=tool_context, packet_type="tool_context", priority=9.0)
            )

        return packets

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        """
        負責執行 Stage2PromptBuilder 中的 select 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return packets

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        """
        負責執行 Stage2PromptBuilder 中的 structure 流程，依照 Stage2PromptBuilder 的流程需求處理 structure 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 Stage2PromptBuilder 中的 compress 流程，依照 Stage2PromptBuilder 的流程需求處理 compress 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            structured: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        structured["tool_context"] = self._compress_multiline_text(
            structured["tool_context"],
            self.config.max_tool_lines,
            self.config.max_tool_chars,
        )
        return structured

    def render(self, compressed: dict[str, Any], **kwargs):
        """
        負責執行 Stage2PromptBuilder 中的 render 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            compressed: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
