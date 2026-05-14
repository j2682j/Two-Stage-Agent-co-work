from __future__ import annotations

from typing import Any

from .builder import PromptBuilder, PromptPacket


class RepairPromptBuilder(PromptBuilder):
    """
    負責在 prompt.repair_prompt_builder 中封裝 RepairPromptBuilder，封裝提示詞組裝規則，將任務上下文整理成模型輸入。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def gather(self, **kwargs) -> list[PromptPacket]:
        """
        負責執行 RepairPromptBuilder 中的 gather 流程，依照 RepairPromptBuilder 的流程需求處理 gather 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 RepairPromptBuilder 中的 select 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
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
        負責執行 RepairPromptBuilder 中的 structure 流程，依照 RepairPromptBuilder 的流程需求處理 structure 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            packets: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        packet = packets[0] if packets else None
        expected_weight_count = 0
        if packet is not None:
            expected_weight_count = int(packet.metadata.get("expected_weight_count", 0))
        return {"expected_weight_count": expected_weight_count}

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        負責執行 RepairPromptBuilder 中的 compress 流程，依照 RepairPromptBuilder 的流程需求處理 compress 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            structured: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return structured

    def render(self, compressed: dict[str, Any], **kwargs):
        """
        負責執行 RepairPromptBuilder 中的 render 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            compressed: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
