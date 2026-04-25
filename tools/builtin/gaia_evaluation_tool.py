"""GAIA評估工具

GAIA (General AI Assistants) 評估工具
用於評估智慧代理的通用能力
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime
from ..base import Tool, ToolParameter
from hello_agents.evaluation.benchmarks.gaia.dataset import GAIADataset
from hello_agents.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator
from hello_agents.evaluation.benchmarks.gaia.metrics import GAIAMetrics
from utils.project_paths import get_eval_output_dir


class GAIAEvaluationTool(Tool):
    """GAIA評估工具
    
    用於評估智慧代理的通用AI 助理能力。
    支援三個難度等級：
    - Level 1: 簡單任務（0步推理）
    - Level 2: 中等任務（1-5步推理）
    - Level 3: 困難任務（5+步推理）
    """
    
    def __init__(self, local_data_path: Optional[str] = None):
        """初始化GAIA評估工具
        
        Args:
            local_data_path: 本地資料路徑（可選）
        """
        super().__init__(
            name="gaia_evaluation",
            description=(
                "評估智慧代理的通用AI 助理能力。使用GAIA (General AI Assistants)基準測試。"
                "支援三個難度等級：Level 1(簡單)、Level 2(中等)、Level 3(困難)。"
            )
        )
        self.local_data_path = local_data_path
        self.dataset = None
        self.evaluator = None
        self.metrics_calculator = GAIAMetrics()
    
    def get_parameters(self) -> List[ToolParameter]:
        """取得工具參數定義"""
        return [
            ToolParameter(
                name="agent",
                type="object",
                description="要評估的智慧代理實例",
                required=True
            ),
            ToolParameter(
                name="level",
                type="integer",
                description="難度等級：1(簡單), 2(中等), 3(困難), None(全部)",
                required=False,
                default=None
            ),
            ToolParameter(
                name="max_samples",
                type="integer",
                description="最大評估樣本數，None表示全部",
                required=False,
                default=None
            ),
            ToolParameter(
                name="local_data_dir",
                type="string",
                description="本地資料集目錄路徑",
                required=False,
                default=None
            )
        ]
    
    def run(
        self,
        agent: Any,
        level: Optional[int] = None,
        max_samples: Optional[int] = None,
        local_data_dir: Optional[str] = None,
        export_results: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """執行GAIA一鍵評估

        Args:
            agent: 要評估的智慧代理
            level: 難度等級 (1-3)，None表示全部
            max_samples: 最大樣本數，None表示全部
            local_data_dir: 本地資料目錄路徑
            export_results: 是否匯出GAIA格式結果
            generate_report: 是否生成評估報告

        Returns:
            評估結果字典
        """
        print("\n" + "=" * 60)
        print("GAIA一鍵評估")
        print("=" * 60)

        # 顯示設定
        print(f"\n設定:")
        print(f"   智慧代理: {getattr(agent, 'name', 'Unknown')}")
        print(f"   難度等級: {level or '全部'}")
        print(f"   樣本數量: {max_samples or '全部'}")

        try:
            # 步驟1: 執行HelloAgents評估
            print("\n" + "=" * 60)
            print("步驟1: 執行HelloAgents評估")
            print("=" * 60)

            results = self._run_evaluation(agent, level, max_samples, local_data_dir)

            # 步驟2: 匯出GAIA格式結果
            if export_results:
                print("\n" + "=" * 60)
                print("步驟2: 匯出GAIA格式結果")
                print("=" * 60)

                self._export_results(results)

            # 步驟3: 生成評估報告
            if generate_report:
                print("\n" + "=" * 60)
                print("步驟3: 生成評估報告")
                print("=" * 60)

                self.generate_report(results)

            # 顯示最終結果
            print("\n" + "=" * 60)
            print("[INFO] 最終結果")
            print("=" * 60)
            print(f"   精確匹配率: {results['exact_match_rate']:.2%}")
            print(f"   部分匹配率: {results['partial_match_rate']:.2%}")
            print(f"   正確數: {results['exact_matches']}/{results['total_samples']}")

            return results

        except Exception as e:
            print(f"\n[ERROR] 評估失敗: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "benchmark": "GAIA",
                "agent_name": getattr(agent, 'name', 'Unknown')
            }

    def _run_evaluation(
        self,
        agent: Any,
        level: Optional[int],
        max_samples: Optional[int],
        local_data_dir: Optional[str]
    ) -> Dict[str, Any]:
        """執行評估"""
        # 載入資料集
        self.dataset = GAIADataset(
            level=level,
            local_data_dir=local_data_dir or self.local_data_path
        )
        dataset_items = self.dataset.load()

        if not dataset_items:
            raise ValueError("資料集載入失敗或為空")

        # 建立評估器
        self.evaluator = GAIAEvaluator(
            dataset=self.dataset,
            level=level,
            local_data_dir=local_data_dir or self.local_data_path
        )

        # 執行評估
        results = self.evaluator.evaluate(agent, max_samples)

        return results

    def _export_results(self, results: Dict[str, Any]) -> None:
        """匯出GAIA格式結果和提交說明"""
        # 建立輸出目錄
        output_dir = get_eval_output_dir("gaia_official")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成檔名
        agent_name = results.get("agent_name", "Unknown").replace("/", "_")
        level = results.get("level_filter")
        level_str = f"_level{level}" if level else "_all"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"gaia{level_str}_result_{timestamp}.jsonl"

        # 匯出JSONL結果
        self.evaluator.export_to_gaia_format(
            results,
            output_file,
            include_reasoning=True
        )

        # 生成提交說明檔案
        self._generate_submission_guide(results, output_dir, output_file)

    def _generate_submission_guide(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        result_file: Path
    ) -> None:
        """生成提交說明檔案

        Args:
            results: 評估結果
            output_dir: 輸出目錄
            result_file: 結果檔案路徑
        """
        agent_name = results.get("agent_name", "Unknown")
        level = results.get("level_filter")
        total_samples = results.get("total_samples", 0)
        exact_matches = results.get("exact_matches", 0)
        exact_match_rate = results.get("exact_match_rate", 0)

        # 生成提交說明
        guide_content = f"""# GAIA評估結果提交指南

## [INFO] 評估結果摘要

- **模型名稱**: {agent_name}
- **評估等級**: {level or '全部'}
- **總樣本數**: {total_samples}
- **精確匹配數**: {exact_matches}
- **精確匹配率**: {exact_match_rate:.2%}

## [INFO] 提交檔案

**結果檔案**: `{result_file.name}`

此檔案包含：
- 每個任務的task_id
- 模型的答案（model_answer）
- 推理軌跡（reasoning_trace）

## [INFO] 如何提交到GAIA排行榜

### 步驟1: 訪問GAIA排行榜

打開瀏覽器，訪問：
```
https://huggingface.co/spaces/gaia-benchmark/leaderboard
```

### 步驟2: 準備提交資訊

在提交表單中填寫以下資訊：

1. **Model Name（模型名稱）**: `{agent_name}`
2. **Model Family（模型家族）**: 例如 `GPT`, `Claude`, `Qwen` 等
3. **Model Type（模型類型）**:
   - `Open-source` (開源)
   - `Proprietary` (專有)
4. **Results File（結果檔案）**: 上傳 `{result_file.name}`

### 步驟3: 上傳結果檔案

1. 點擊 "Choose File" 按鈕
2. 選擇檔案: `{result_file.absolute()}`
3. 確認檔案格式為 `.jsonl`

### 步驟4: 提交

1. 檢查所有資訊是否正確
2. 點擊 "Submit" 按鈕
3. 等待評估結果（通常需要幾分鐘）

## 📋 結果檔案格式說明

GAIA要求的JSONL格式（每行一個JSON對象）：

```json
{{"task_id": "xxx", "model_answer": "答案", "reasoning_trace": "推理過程"}}
```

**字段說明**：
- `task_id`: 任務ID（與GAIA資料集對應）
- `model_answer`: 模型的最終答案
- `reasoning_trace`: 模型的推理過程（可選）

## [WARN] 注意事項

1. **答案格式**：
   - 數字：不使用逗號分隔符，不使用單位符號
   - 字串：不使用冠詞，使用小寫
   - 列表：逗號分隔，按字母順序排列

2. **檔案大小**：
   - 確保檔案不超過10MB
   - 如果檔案過大，考慮移除reasoning_trace

3. **提交頻率**：
   - 建議先在小樣本上測試
   - 確認結果正確后再提交完整評估

## 📞 取得幫助

如果遇到問題：
1. 查看GAIA官方文檔：https://huggingface.co/gaia-benchmark
2. 在HuggingFace論壇提問
3. 檢查結果檔案格式是否正確

---

**生成時間**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**工具版本**: HelloAgents GAIA Evaluation Tool v1.0
"""

        # 保存提交說明
        guide_file = output_dir / f"SUBMISSION_GUIDE_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)

        print(f"📄 提交說明已生成: {guide_file}")

    def generate_report(
        self,
        results: Dict[str, Any],
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        """生成評估報告

        Args:
            results: 評估結果
            output_file: 輸出檔案路徑（可選）

        Returns:
            Markdown格式的報告
        """
        # 生成報告內容
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        agent_name = results.get("agent_name", "Unknown")
        level = results.get("level_filter")
        total_samples = results.get("total_samples", 0)
        exact_matches = results.get("exact_matches", 0)
        partial_matches = results.get("partial_matches", 0)
        exact_match_rate = results.get("exact_match_rate", 0)
        partial_match_rate = results.get("partial_match_rate", 0)
        level_metrics = results.get("level_metrics", {})
        detailed_results = results.get("detailed_results", [])

        # 建構報告
        report = f"""# GAIA評估報告

**生成時間**: {timestamp}

## [INFO] 評估概覽

- **智慧代理**: {agent_name}
- **難度等級**: {level or '全部'}
- **總樣本數**: {total_samples}
- **精確匹配數**: {exact_matches}
- **部分匹配數**: {partial_matches}
- **精確匹配率**: {exact_match_rate:.2%}
- **部分匹配率**: {partial_match_rate:.2%}

## 📈 詳細指標

### 分級正確率

"""

        # 添加分級統計
        for level_name, metrics in level_metrics.items():
            level_num = level_name.replace("Level_", "")
            total = metrics.get("total", 0)
            exact = metrics.get("exact_matches", 0)
            partial = metrics.get("partial_matches", 0)
            exact_rate = metrics.get("exact_match_rate", 0)
            partial_rate = metrics.get("partial_match_rate", 0)

            report += f"- **Level {level_num}**: {exact_rate:.2%} 精確 / {partial_rate:.2%} 部分 ({exact}/{total})\n"

        # 添加樣本詳情（前10個）
        report += "\n## [INFO] 樣本詳情（前10個）\n\n"
        report += "| 任務ID | 等級 | 預測答案 | 正確答案 | 精確匹配 | 部分匹配 |\n"
        report += "|--------|------|----------|----------|----------|----------|\n"

        for i, detail in enumerate(detailed_results[:10]):
            task_id = detail.get("task_id", "")
            level_num = detail.get("level", "")
            predicted = str(detail.get("predicted", ""))[:50]  # 限制長度
            expected = str(detail.get("expected", ""))[:50]
            exact = "[OK]" if detail.get("exact_match") else "[ERROR]"
            partial = "[OK]" if detail.get("partial_match") else "[ERROR]"

            report += f"| {task_id} | {level_num} | {predicted} | {expected} | {exact} | {partial} |\n"

        # 添加正確率可視化
        report += "\n## [INFO] 正確率可視化\n\n"
        report += "```\n"
        bar_length = 50
        filled = int(exact_match_rate * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        report += f"精確匹配: {bar} {exact_match_rate:.2%}\n"

        filled_partial = int(partial_match_rate * bar_length)
        bar_partial = "█" * filled_partial + "░" * (bar_length - filled_partial)
        report += f"部分匹配: {bar_partial} {partial_match_rate:.2%}\n"
        report += "```\n"

        # 添加建議
        report += "\n## [INFO] 建議\n\n"
        if exact_match_rate >= 0.9:
            report += "- [OK] 表現優秀！智慧代理在GAIA基準上表現出色。\n"
        elif exact_match_rate >= 0.7:
            report += "- 👍 表現良好，但仍有提升空間。\n"
            report += "- [INFO] 建議優化提示詞和推理策略。\n"
        elif exact_match_rate >= 0.5:
            report += "- [WARN] 表現一般，需要改進。\n"
            report += "- [INFO] 建議檢查工具使用和多步推理能力。\n"
        else:
            report += "- [ERROR] 表現較差，需要大幅改進。\n"
            report += "- [INFO] 建議從簡單等級開始，逐步提升。\n"

        # 保存報告
        if output_file is None:
            output_dir = Path("./evaluation_reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"gaia_report_{timestamp_str}.md"
        else:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 報告已生成: {output_file}")

        return report

    def get_dataset_info(self, level: Optional[int] = None) -> Dict[str, Any]:
        """取得資料集資訊
        
        Args:
            level: 難度等級
            
        Returns:
            資料集資訊字典
        """
        try:
            dataset = GAIADataset(level=level, local_data_path=self.local_data_path)
            items = dataset.load()
            
            # 取得統計資訊
            stats = dataset.get_statistics()
            level_dist = dataset.get_level_distribution()
            
            return {
                "level": level,
                "total_samples": len(items),
                "level_distribution": level_dist,
                "statistics": stats,
                "sample_keys": list(items[0].keys()) if items else [],
                "levels_available": [1, 2, 3]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def validate_agent(self, agent: Any) -> bool:
        """驗證智慧代理是否具備必要的介面
        
        Args:
            agent: 要驗證的智慧代理
            
        Returns:
            是否有效
        """
        # 檢查agent是否有run方法
        if not hasattr(agent, 'run'):
            return False
        
        # 檢查run方法是否可呼叫
        if not callable(getattr(agent, 'run')):
            return False
        
        return True
    

 
