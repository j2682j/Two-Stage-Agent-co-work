from __future__ import annotations

import re

from .base_parser import BaseParser


class RankingParser(BaseParser):
    """
    負責在 parser.ranking_parser 中封裝 RankingParser，封裝模型輸出解析流程，將文字結果轉成結構化資料。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def parse(self, completion: str, max_num: int = 4) -> list[int]:
        """
        負責執行 RankingParser 中的 parse 流程，解析模型輸出並取出答案、決策、排序或 JSON 結構。
        
        Args:
            completion: 此流程需要使用的輸入資料。
            max_num: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[int]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        content = completion or ""
        pattern = r"\[([1234567]),\s*([1234567])\]"
        matches = re.findall(pattern, content)

        try:
            match = matches[-1]
            tops = [int(match[0]) - 1, int(match[1]) - 1]

            def clip(x: int) -> int:
                """
                負責執行 RankingParser 中的 clip 流程，依照 RankingParser 的流程需求處理 clip 對應的資料轉換、狀態操作或結果產生。
                
                Args:
                    x: 此流程需要使用的輸入資料。
                
                Returns:
                    執行結果；若函式標註回傳型別，預期型別為 int。
                
                限制或副作用:
                    可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
                """
                if x < 0:
                    return 0
                if x > max_num - 1:
                    return max_num - 1
                return x

            return [clip(x) for x in tops]
        except Exception:
            return list(range(min(2, max_num)))
