"""
BFCL 資料集載入模組

負責載入 Berkeley Function Calling Leaderboard 資料集
支援從BFCL官方資料目錄載入，包括測試資料與 ground truth
"""

from typing import List, Dict, Any, Optional, Union
import json
import os
from pathlib import Path


class BFCLDataset:
    """
    負責在 evaluation.benchmarks.bfcl.dataset 中封裝 BFCLDataset，封裝 benchmark 評估、答案判定、分數計算或報告資料整理流程。
    
    Args:
        bfcl_data_dir: 此流程需要使用的輸入資料。
        category: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    # BFCL v4 資料集的標準類別對應
    CATEGORY_MAPPING = {
        "simple_python": "BFCL_v4_simple_python",
        "simple_java": "BFCL_v4_simple_java",
        "simple_javascript": "BFCL_v4_simple_javascript",
        "multiple": "BFCL_v4_multiple",
        "parallel": "BFCL_v4_parallel",
        "parallel_multiple": "BFCL_v4_parallel_multiple",
        "irrelevance": "BFCL_v4_irrelevance",
        "live_simple": "BFCL_v4_live_simple",
        "live_multiple": "BFCL_v4_live_multiple",
        "live_parallel": "BFCL_v4_live_parallel",
        "live_parallel_multiple": "BFCL_v4_live_parallel_multiple",
        "live_irrelevance": "BFCL_v4_live_irrelevance",
        "live_relevance": "BFCL_v4_live_relevance",
        "multi_turn_base": "BFCL_v4_multi_turn_base",
        "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func",
        "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param",
        "multi_turn_long_context": "BFCL_v4_multi_turn_long_context",
        "memory": "BFCL_v4_memory",
        "web_search": "BFCL_v4_web_search",
    }

    def __init__(
        self,
        bfcl_data_dir: Union[str, Path] = "./temp_gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
        category: Optional[str] = None
    ):
        """
        負責執行 BFCLDataset 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            bfcl_data_dir: 此流程需要使用的輸入資料。
            category: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.bfcl_data_dir = Path(bfcl_data_dir)
        self.category = category
        self.data = []
        self.ground_truth = {}

        # 驗證資料目錄
        if not self.bfcl_data_dir.exists():
            print(f"   [WARN] BFCL資料目錄不存在: {self.bfcl_data_dir}")
            print(f"   請確保已克隆BFCL倉庫到正確位置")

        # 驗證possible_answer目錄
        self.answer_dir = self.bfcl_data_dir / "possible_answer"
        if not self.answer_dir.exists():
            print(f"   [WARN] Ground truth目錄不存在: {self.answer_dir}")

    def load(self) -> List[Dict[str, Any]]:
        """
        負責執行 BFCLDataset 中的 load 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.bfcl_data_dir.exists():
            print(f"   [WARN] 資料目錄不存在，無法載入資料")
            return []

        # 確定要載入的檔案
        if self.category:
            # 載入指定類別
            filename = self.CATEGORY_MAPPING.get(self.category)
            if not filename:
                print(f"   [WARN] 未知類別: {self.category}")
                print(f"   支援的類別: {list(self.CATEGORY_MAPPING.keys())}")
                return []

            self.data = self._load_category(filename)
        else:
            # 載入所有類別（不推薦，資料量大）
            print(f"   [WARN] 未指定類別，將載入simple_python作為範例")
            self.data = self._load_category(self.CATEGORY_MAPPING["simple_python"])

        print(f"[OK] BFCL資料集載入完成")
        print(f"   資料目錄: {self.bfcl_data_dir}")
        print(f"   類別: {self.category or 'simple_python'}")
        print(f"   樣本數: {len(self.data)}")
        print(f"   Ground truth數: {len(self.ground_truth)}")

        return self.data
    
    def _load_category(self, filename: str) -> List[Dict[str, Any]]:
        """
        負責執行 BFCLDataset 中的 _load_category 流程，依照 BFCLDataset 的流程需求處理 _load_category 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            filename: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 載入測試資料
        test_file = self.bfcl_data_dir / f"{filename}.json"
        if not test_file.exists():
            print(f"   [WARN] 測試資料檔案不存在: {test_file}")
            return []

        test_data = self._load_jsonl_file(test_file)
        print(f"   ✓ 載入測試資料: {test_file.name} ({len(test_data)} 樣本)")

        # 載入ground truth
        gt_file = self.answer_dir / f"{filename}.json"
        if gt_file.exists():
            gt_data = self._load_jsonl_file(gt_file)
            # 建構ground truth 字典
            for item in gt_data:
                item_id = item.get("id")
                if item_id:
                    self.ground_truth[item_id] = item.get("ground_truth", [])
            print(f"   ✓ 載入ground truth: {gt_file.name} ({len(gt_data)} 樣本)")
        else:
            print(f"   [WARN] Ground truth檔案不存在: {gt_file}")

        # 合併測試資料與 ground truth
        for item in test_data:
            item_id = item.get("id")
            if item_id and item_id in self.ground_truth:
                item["ground_truth"] = self.ground_truth[item_id]

        return test_data

    def _load_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        負責執行 BFCLDataset 中的 _load_jsonl_file 流程，依照 BFCLDataset 的流程需求處理 _load_jsonl_file 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            file_path: 要讀取或寫入的檔案或目錄路徑。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"   [WARN] JSON解析失敗: {e}")
                        continue
        return data

    def get_ground_truth(self, sample_id: str) -> List[Dict[str, Any]]:
        """
        負責執行 BFCLDataset 中的 get_ground_truth 流程，依照 BFCLDataset 的流程需求處理 get_ground_truth 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sample_id: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.ground_truth.get(sample_id, [])

    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        負責執行 BFCLDataset 中的 get_sample 流程，依照 BFCLDataset 的流程需求處理 get_sample 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            index: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_available_categories(self) -> List[str]:
        """
        負責執行 BFCLDataset 中的 get_available_categories 流程，依照 BFCLDataset 的流程需求處理 get_available_categories 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return list(self.CATEGORY_MAPPING.keys())

    def __len__(self) -> int:
        """
        負責執行 BFCLDataset 中的 __len__ 流程，依照 BFCLDataset 的流程需求處理 __len__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.data:
            self.load()
        return len(self.data)

    def __iter__(self):
        """
        負責執行 BFCLDataset 中的 __iter__ 流程，依照 BFCLDataset 的流程需求處理 __iter__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.data:
            self.load()
        return iter(self.data)

 
