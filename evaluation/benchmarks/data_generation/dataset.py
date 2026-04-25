"""
AIME ???????

?? AIME ???????????
- ? HuggingFace ??????
- ?????????
- ??????
"""


import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class AIDataset:
    """AIME資料集載入器"""
    
    def __init__(
        self,
        dataset_type: str = "generated",  # "generated" or "real"
        data_path: Optional[str] = None,
        year: Optional[int] = None,  # 用於真題資料，如2024, 2025
        cache_dir: Optional[str] = None
    ):
        """
        初始化AIME資料集
        
        Args:
            dataset_type: 資料集類型，"generated"（生成的）或"real"（真題）
            data_path: 本地資料路徑（用於generated類型）
            year: AIME年份（用於real類型），如2024, 2025
            cache_dir: 快取目錄
        """
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.year = year
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/hello_agents/aime")
        
        self.problems: List[Dict[str, Any]] = []
        
    def load(self) -> List[Dict[str, Any]]:
        """
        載入資料集
        
        Returns:
            問題列表，每個問題包含：
            - problem_id: 問題ID
            - problem: 問題描述
            - answer: 答案
            - solution: 解答過程（可選）
            - difficulty: 難度（可選）
            - topic: 主題（可選）
        """
        if self.dataset_type == "generated":
            return self._load_generated_data()
        elif self.dataset_type == "real":
            return self._load_real_data()
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
    
    def _load_generated_data(self) -> List[Dict[str, Any]]:
        """載入生成的資料"""
        if not self.data_path:
            raise ValueError("data_path is required for generated dataset")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"📥 載入生成資料: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 統一資料格式
        problems = []
        for idx, item in enumerate(data):
            problem = {
                "problem_id": item.get("id", f"gen_{idx}"),
                "problem": item.get("problem", item.get("question", "")),
                "answer": item.get("answer", ""),
                "solution": item.get("solution", item.get("reasoning", "")),
                "difficulty": item.get("difficulty", None),
                "topic": item.get("topic", item.get("category", None))
            }
            problems.append(problem)
        
        self.problems = problems
        print(f"[OK] 載入了 {len(problems)} 個生成題目")
        return problems
    
    def _load_real_data(self) -> List[Dict[str, Any]]:
        from huggingface_hub import snapshot_download
        """從HuggingFace載入AIME真題資料"""
        if not self.year:
            raise ValueError("year is required for real dataset")

        print(f"📥 從HuggingFace載入AIME {self.year}真題...")

        try:
            # 使用AIME 2025資料集
            repo_id = "math-ai/aime25"
            use_datasets_lib = False  # 使用snapshot_download（JSONL格式）

            print(f"   使用資料集: {repo_id}")

            # 使用snapshot_download下載檔案
            local_dir = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=self.cache_dir
            )

            # 查找JSONL資料檔案
            data_files = list(Path(local_dir).glob("*.jsonl"))

            if not data_files:
                raise FileNotFoundError(f"No JSONL data file found in {repo_id}")

            data_file = data_files[0]
            print(f"   ✓ 找到資料檔案: {data_file.name}")

            # 載入JSONL資料
            data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            # 統一資料格式（AIME 2025使用小寫字段名）
            problems = []
            for idx, item in enumerate(data):
                problem = {
                    "problem_id": item.get("id", f"aime_2025_{idx}"),
                    "problem": item.get("problem", ""),
                    "answer": item.get("answer", ""),
                    "solution": item.get("solution", ""),  # AIME 2025沒有solution字段
                    "difficulty": item.get("difficulty", None),
                    "topic": item.get("topic", None)
                }
                problems.append(problem)
            
            self.problems = problems
            print(f"[OK] 載入了 {len(problems)} 個AIME {self.year}真題")
            return problems
            
        except Exception as e:
            print(f"[ERROR] 載入失敗: {e}")
            print(f"提示: 請確保已安裝huggingface_hub並設定HF_TOKEN")
            raise
    
    def get_problem(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """根據ID取得問題"""
        for problem in self.problems:
            if problem["problem_id"] == problem_id:
                return problem
        return None
    
    def get_problems_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """根據主題取得問題"""
        return [p for p in self.problems if p.get("topic") == topic]
    
    def get_problems_by_difficulty(self, min_diff: int, max_diff: int) -> List[Dict[str, Any]]:
        """根據難度范圍取得問題"""
        return [
            p for p in self.problems 
            if p.get("difficulty") and min_diff <= p["difficulty"] <= max_diff
        ]
    
    def __len__(self) -> int:
        """回傳資料集大小"""
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """支援索引訪問"""
        return self.problems[idx]
