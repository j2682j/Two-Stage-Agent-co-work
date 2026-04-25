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
    """BFCL 資料集載入器

    支援從BFCL官方資料目錄載入資料集，包括測試資料與 ground truth。

    資料集類別（BFCL v4）:
    - simple_python: 簡單Python函式呼叫
    - simple_java: 簡單Java函式呼叫
    - simple_javascript: 簡單JavaScript函式呼叫
    - multiple: 多函式呼叫
    - parallel: 平行函式呼叫
    - parallel_multiple: 平行多函式呼叫
    - irrelevance: 無關檢測
    - live_simple: 使用者貢獻的簡單函式呼叫
    - live_multiple: 使用者貢獻的多函式呼叫
    - live_parallel: 使用者貢獻的平行函式呼叫
    - multi_turn_base: 多輪對話基礎
    - multi_turn_miss_func: 多輪對話缺失函式
    - multi_turn_miss_param: 多輪對話缺失參數
    - multi_turn_long_context: 多輪對話長上下文

    Attributes:
        bfcl_data_dir: BFCL官方資料目錄路徑
        category: 評估類別
        data: 載入的測試資料清單
        ground_truth: ground truth 字典，key為樣本id
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
        """初始化 BFCL 資料集載入器

        Args:
            bfcl_data_dir: BFCL官方資料目錄路徑（包含BFCL_v4_*.json檔案）
            category: 評估類別，如'simple_python', 'multiple'等
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
        """載入資料集（包括測試資料與 ground truth）

        Returns:
            資料集列表，每個元素包含問題、函式定義、ground truth等
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
        """載入指定類別的資料（包括測試資料與 ground truth）

        Args:
            filename: 檔名（不含.json后綴），如'BFCL_v4_simple_python'

        Returns:
            測試資料列表
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
        """載入JSONL檔案（每行一個JSON對象）

        Args:
            file_path: JSON/JSONL檔案路徑

        Returns:
            資料列表
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
        """取得指定樣本的ground truth

        Args:
            sample_id: 樣本ID

        Returns:
            Ground truth列表
        """
        return self.ground_truth.get(sample_id, [])

    def get_sample(self, index: int) -> Dict[str, Any]:
        """取得單個樣本

        Args:
            index: 樣本索引

        Returns:
            樣本資料
        """
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_available_categories(self) -> List[str]:
        """取得所有可用的類別

        Returns:
            類別列表
        """
        return list(self.CATEGORY_MAPPING.keys())

    def __len__(self) -> int:
        """回傳資料集大小"""
        if not self.data:
            self.load()
        return len(self.data)

    def __iter__(self):
        """迭代器"""
        if not self.data:
            self.load()
        return iter(self.data)

 
