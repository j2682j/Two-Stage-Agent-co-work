"""
GAIA 資料集載入模組

負責從 HuggingFace 載入 GAIA (General AI Assistants) 資料集
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json


class GAIADataset:
    """
    負責在 evaluation.benchmarks.gaia.dataset 中封裝 GAIADataset，封裝 benchmark 評估、答案判定、分數計算或報告資料整理流程。
    
    Args:
        dataset_name: 此流程需要使用的輸入資料。
        split: 此流程需要使用的輸入資料。
        level: 此流程需要使用的輸入資料。
        local_data_dir: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        dataset_name: str = "gaia-benchmark/GAIA",
        split: str = "validation",
        level: Optional[int] = None,
        local_data_dir: Optional[Union[str, Path]] = None
    ):
        """
        負責執行 GAIADataset 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            dataset_name: 此流程需要使用的輸入資料。
            split: 此流程需要使用的輸入資料。
            level: 此流程需要使用的輸入資料。
            local_data_dir: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.dataset_name = dataset_name
        self.split = split
        self.level = level
        self.local_data_dir = Path(local_data_dir) if local_data_dir else None
        self.data = []
        self._is_local = self._check_if_local_source()

    def _check_if_local_source(self) -> bool:
        """
        負責執行 GAIADataset 中的 _check_if_local_source 流程，依照 GAIADataset 的流程需求處理 _check_if_local_source 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.local_data_dir and self.local_data_dir.exists():
            return True
        return False

    def load(self) -> List[Dict[str, Any]]:
        """
        負責執行 GAIADataset 中的 load 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.data:
            return self.data

        if self._is_local:
            self.data = self._load_from_local()
        else:
            self.data = self._load_from_huggingface()

        # 按等級過濾
        # if self.level is not None:
        #     self.data = [item for item in self.data if item.get("level") == self.level]

        print(f"[OK] GAIA資料集載入完成")
        print(f"   資料源: {self.dataset_name}")
        print(f"   分割: {self.split}")
        print(f"   等級: {self.level or '全部'}")
        print(f"   樣本數: {len(self.data)}")

        return self.data

    def _load_from_local(self) -> List[Dict[str, Any]]:
        """
        負責執行 GAIADataset 中的 _load_from_local 流程，依照 GAIADataset 的流程需求處理 _load_from_local 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        data = []

        if not self.local_data_dir or not self.local_data_dir.exists():
            print("   [WARN] 本地資料目錄不存在")
            return data

        try:
            import pandas as pd
        except ImportError:
            print("   [WARN] 缺少 pandas / pyarrow，無法讀取 parquet")
            return data

        metadata_dir = self.local_data_dir / "2023" / self.split
        if not metadata_dir.exists():
            print(f"   [WARN] 找不到本地 GAIA 目錄: {metadata_dir}")
            return data

        candidate_files = []
        if self.level is not None:
            candidate_files.append(metadata_dir / f"metadata.level{self.level}.parquet")
        candidate_files.append(metadata_dir / "metadata.parquet")

        metadata_file = next((path for path in candidate_files if path.exists()), None)
        if metadata_file is None:
            expected = ", ".join(str(path) for path in candidate_files)
            print(f"   [WARN] 找不到本地 metadata 檔案: {expected}")
            return data

        try:
            df = pd.read_parquet(metadata_file)
            records = df.to_dict(orient="records")

            for item in records:
                if item.get("task_id") == "0-0-0-0-0":
                    continue

                if item.get("file_name"):
                    item["file_name"] = str(metadata_dir / item["file_name"])

                data.append(self._standardize_item(item))

            print(f"   載入本地 metadata: {metadata_file.name} ({len(data)} 筆樣本)")
        except Exception as e:
            print(f"   [WARN] 讀取本地 metadata 失敗: {metadata_file.name} - {e}")

        return data

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """
        負責執行 GAIADataset 中的 _load_from_huggingface 流程，依照 GAIADataset 的流程需求處理 _load_from_huggingface 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            from huggingface_hub import snapshot_download
            import os
            import pandas as pd
            from pathlib import Path

            print(f"   正在從HuggingFace下載: {self.dataset_name}")

            # 取得HF token
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("   [WARN] 找不到HF_TOKEN環境變數")
                print("   GAIA是gated dataset，需要在HuggingFace上申請訪問權限")
                print("   然後設定環境變數: HF_TOKEN=your_token")
                return []

            # 下載資料集到本地
            print(f"...下載GAIA資料集...")
            # 使用目前工作目錄下的data/gaia檔案夾
            local_dir = Path.cwd() / "data" / "gaia"
            local_dir.mkdir(parents=True, exist_ok=True)

            try:
                snapshot_download(
                    repo_id=self.dataset_name,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    token=hf_token,
                    local_dir_use_symlinks=False  # Windows相容性
                )
                print(f"   ✓ 資料集下載完成: {local_dir}")
            except Exception as e:
                print(f"   [WARN] 下載失敗: {e}")
                print("   請確保:")
                print("   1. 已在HuggingFace上申請GAIA訪問權限")
                print("   2. HF_TOKEN正確且有效")
                return []

            metadata_dir = local_dir / "2023" / self.split
            candidate_files = []
            if self.level is not None:
                candidate_files.append(metadata_dir / f"metadata.level{self.level}.parquet")
            candidate_files.append(metadata_dir / "metadata.parquet")

            metadata_file = next((path for path in candidate_files if path.exists()), None)
            if metadata_file is None:
                expected = ", ".join(str(path) for path in candidate_files)
                print(f"   [WARN] 找不到 metadata 檔案: {expected}")
                return []

            df = pd.read_parquet(metadata_file)
            records = df.to_dict(orient="records")

            data = []
            for item in records:
                if item.get("task_id") == "0-0-0-0-0":
                    continue

                if item.get("file_name"):
                    item["file_name"] = str(metadata_dir / item["file_name"])

                standardized_item = self._standardize_item(item)
                data.append(standardized_item)

            print(f"   ✓ 載入了 {len(data)} 個樣本")
            return data

        except ImportError:
            print("   [WARN] 缺少 huggingface_hub 或 pandas / pyarrow 相關依賴")
            print("   提示: pip install huggingface_hub pandas pyarrow")
            return []
        except Exception as e:
            print(f"   [WARN] 載入失敗: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _standardize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        負責執行 GAIADataset 中的 _standardize_item 流程，依照 GAIADataset 的流程需求處理 _standardize_item 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            item: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # GAIA資料集的標準字段
        standardized = {
            "task_id": item.get("task_id", ""),
            "question": item.get("Question", item.get("question", "")),
            "level": item.get("Level", item.get("level", "1")),
            "final_answer": item.get("Final answer", item.get("final_answer", "")),
            "file_name": item.get("file_name", ""),
            "file_path": item.get("file_path", ""),
            "annotator_metadata": item.get("Annotator Metadata", item.get("annotator_metadata", {})),
            "steps": item.get("Steps", item.get("steps", 0)),
            "tools": item.get("Tools", item.get("tools", [])),
            "raw_item": item  # 保留原始資料
        }

        return standardized
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        負責執行 GAIADataset 中的 get_sample 流程，依照 GAIADataset 的流程需求處理 get_sample 對應的資料轉換、狀態操作或結果產生。
        
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

    def get_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        負責執行 GAIADataset 中的 get_by_level 流程，依照 GAIADataset 的流程需求處理 get_by_level 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            level: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.data:
            self.load()
        return [item for item in self.data if item.get("level") == level]

    def get_level_distribution(self) -> Dict[int, int]:
        """
        負責執行 GAIADataset 中的 get_level_distribution 流程，依照 GAIADataset 的流程需求處理 get_level_distribution 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[int, int]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.data:
            self.load()

        distribution = {1: 0, 2: 0, 3: 0}
        for item in self.data:
            level = item.get("level", 1)
            if level in distribution:
                distribution[level] += 1

        return distribution

    def get_statistics(self) -> Dict[str, Any]:
        """
        負責執行 GAIADataset 中的 get_statistics 流程，依照 GAIADataset 的流程需求處理 get_statistics 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.data:
            self.load()

        level_dist = self.get_level_distribution()

        # 統計需要檔案的樣本數
        with_files = sum(1 for item in self.data if item.get("file_name"))

        # 統計平均步數
        steps_list = [item.get("steps", 0) for item in self.data if item.get("steps")]
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0

        return {
            "total_samples": len(self.data),
            "level_distribution": level_dist,
            "samples_with_files": with_files,
            "average_steps": avg_steps,
            "split": self.split
        }

    def __len__(self) -> int:
        """
        負責執行 GAIADataset 中的 __len__ 流程，依照 GAIADataset 的流程需求處理 __len__ 對應的資料轉換、狀態操作或結果產生。
        
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
        負責執行 GAIADataset 中的 __iter__ 流程，依照 GAIADataset 的流程需求處理 __iter__ 對應的資料轉換、狀態操作或結果產生。
        
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

 
