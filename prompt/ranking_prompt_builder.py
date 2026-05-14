from typing import Any

from .builder import PromptBuilder, PromptPacket

class RankingPromptBuilder(PromptBuilder):
    """
    負責在 prompt.ranking_prompt_builder 中封裝 RankingPromptBuilder，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def gather(self, **kwargs) -> list[PromptPacket]:
        """
        負責執行 RankingPromptBuilder 中的 gather 流程，依照 RankingPromptBuilder 的流程需求處理 gather 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 RankingPromptBuilder 中的 select 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
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
        負責執行 RankingPromptBuilder 中的 structure 流程，依照 RankingPromptBuilder 的流程需求處理 structure 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 RankingPromptBuilder 中的 compress 流程，依照 RankingPromptBuilder 的流程需求處理 compress 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            structured: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 RankingPromptBuilder 中的 render 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            compressed: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
