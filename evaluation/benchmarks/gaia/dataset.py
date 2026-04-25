"""
GAIA 資料集載入模組

負責從 HuggingFace 載入 GAIA (General AI Assistants) 資料集
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json


class GAIADataset:
    """GAIA 資料集載入器

    從 HuggingFace 載入 GAIA 資料集,支援不同難度等級。

    GAIA是一個通用AI 助理評估基準,包含466個真實世界問題,
    需要推理、多模態處理、網頁瀏覽和工具使用等能力。

    難度等級:
    - Level 1: 簡單問題 (0步推理, 直接回答)
    - Level 2: 中等問題 (1-5步推理, 需要簡單工具使用)
    - Level 3: 複雜問題 (5+步推理, 需要複雜工具鏈和多步推理)

    Attributes:
        dataset_name: HuggingFace 資料集名稱
        split: 資料集分割(validation/test)
        level: 難度等級
        data: 載入的資料列表
    """

    def __init__(
        self,
        dataset_name: str = "gaia-benchmark/GAIA",
        split: str = "validation",
        level: Optional[int] = None,
        local_data_dir: Optional[Union[str, Path]] = None
    ):
        """初始化 GAIA 資料集載入器

        Args:
            dataset_name: HuggingFace 資料集名稱
            split: 資料集分割 (validation/test)
            level: 難度等級 (1-3),None表示載入所有等級
            local_data_dir: 本地資料目錄路徑
        """
        self.dataset_name = dataset_name
        self.split = split
        self.level = level
        self.local_data_dir = Path(local_data_dir) if local_data_dir else None
        self.data = []
        self._is_local = self._check_if_local_source()

    def _check_if_local_source(self) -> bool:
        """檢查是否使用本地資料源"""
        if self.local_data_dir and self.local_data_dir.exists():
            return True
        return False

    def load(self) -> List[Dict[str, Any]]:
        """載入資料集

        Returns:
            資料集列表,每個元素包含問題、答案、難度等
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
        """從本地載入資料集"""
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
        """從HuggingFace下載GAIA資料集

        注意：GAIA是gated dataset，需要HF_TOKEN環境變數
        使用snapshot_download下載整個資料集到本地
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
        """標準化資料項格式"""
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
        """取得單個樣本

        Args:
            index: 樣本索引

        Returns:
            樣本資料
        """
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_by_level(self, level: int) -> List[Dict[str, Any]]:
        """取得指定難度等級的樣本

        Args:
            level: 難度等級 (1-3)

        Returns:
            該等級的所有樣本
        """
        if not self.data:
            self.load()
        return [item for item in self.data if item.get("level") == level]

    def get_level_distribution(self) -> Dict[int, int]:
        """取得難度等級分布

        Returns:
            字典，鍵為等級，值為該等級的樣本數
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
        """取得資料集統計資訊

        Returns:
            統計資訊字典
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
        """回傳資料集大小"""
        if not self.data:
            self.load()
        return len(self.data)

    def __iter__(self):
        """迭代器"""
        if not self.data:
            self.load()
        return iter(self.data)

 
