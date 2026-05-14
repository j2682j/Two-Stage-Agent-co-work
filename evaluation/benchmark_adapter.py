from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBenchmarkAdapter(ABC):
    """
    負責在 evaluation.benchmark_adapter 中封裝 BaseBenchmarkAdapter，封裝 benchmark 評估、答案判定、分數計算或報告資料整理流程。
    
    Args:
        agent: 此流程需要使用的輸入資料。
        name: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, agent: Any, name: str | None = None):
        """
        負責執行 BaseBenchmarkAdapter 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            agent: 此流程需要使用的輸入資料。
            name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.agent = agent
        self.name = name or getattr(agent, "name", agent.__class__.__name__)

    @abstractmethod
    def run(self, prompt: str) -> str:
        """
        負責執行 BaseBenchmarkAdapter 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            prompt: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    @abstractmethod
    def normalize_question(self, question: str) -> str:
        """
        負責執行 BaseBenchmarkAdapter 中的 normalize_question 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    def record_evaluation_feedback(
        self,
        *,
        benchmark: str,
        sample: dict[str, Any],
        sample_result: dict[str, Any],
    ) -> None:
        """
        負責執行 BaseBenchmarkAdapter 中的 record_evaluation_feedback 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            benchmark: 此流程需要使用的輸入資料。
            sample: 此流程需要使用的輸入資料。
            sample_result: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return None





