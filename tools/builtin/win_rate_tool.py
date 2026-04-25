"""
Win Rate Evaluation Tool

利用成對比對計算生成資料相對於真題的勝率
"""


import json
import os
from typing import Dict, Any
from datetime import datetime

from hello_agents.tools.base import Tool
from hello_agents.evaluation.benchmarks.data_generation.dataset import AIDataset
from hello_agents.evaluation.benchmarks.data_generation.win_rate import WinRateEvaluator
from hello_agents.core.llm import HelloAgentsLLM


class WinRateTool(Tool):
    """Win Rate評估工具"""
    
    def __init__(self, llm: HelloAgentsLLM = None):
        """
        初始化Win Rate工具
        
        Args:
            llm: LLM實例，用於評估
        """
        super().__init__(
            name="win_rate_evaluation",
            description="通過成對對比計算生成資料相對於真題的勝率"
        )
        self.llm = llm
        
    def get_parameters(self) -> Dict[str, Any]:
        """取得工具參數定義"""
        return {
            "type": "object",
            "properties": {
                "generated_data_path": {
                    "type": "string",
                    "description": "生成資料的JSON檔案路徑"
                },
                "reference_data_path": {
                    "type": "string",
                    "description": "參考資料的JSON檔案路徑（可選）"
                },
                "reference_year": {
                    "type": "integer",
                    "description": "AIME真題年份（可選，如2024, 2025）"
                },
                "num_comparisons": {
                    "type": "integer",
                    "description": "對比次數（可選，預設為min(生成資料數量, 參考資料數量)）"
                },
                "output_dir": {
                    "type": "string",
                    "description": "輸出目錄（可選，預設為evaluation_results/win_rate）"
                },
                "judge_model": {
                    "type": "string",
                    "description": "評委模型名稱（可選，預設為gpt-4o）"
                }
            },
            "required": ["generated_data_path"]
        }
    
    def run(self, params: Dict[str, Any]) -> str:
        """
        執行Win Rate評估
        
        Args:
            params: 工具參數
        
        Returns:
            評估結果的JSON字串
        """
        # 解析參數
        generated_data_path = params["generated_data_path"]
        reference_data_path = params.get("reference_data_path")
        reference_year = params.get("reference_year")
        num_comparisons = params.get("num_comparisons")
        output_dir = params.get("output_dir", "evaluation_results/win_rate")
        judge_model = params.get("judge_model", "gpt-4o")
        
        # 建立輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("[INFO] Win Rate評估")
        print("="*60)
        
        # 1. 載入生成資料
        print(f"\n📥 步驟1: 載入生成資料")
        gen_dataset = AIDataset(dataset_type="generated", data_path=generated_data_path)
        gen_problems = gen_dataset.load()
        
        # 2. 載入參考資料
        if reference_data_path:
            print(f"\n📥 步驟2: 載入參考資料（本地檔案）")
            ref_dataset = AIDataset(dataset_type="generated", data_path=reference_data_path)
            ref_problems = ref_dataset.load()
        elif reference_year:
            print(f"\n📥 步驟2: 載入參考資料（AIME {reference_year}真題）")
            ref_dataset = AIDataset(dataset_type="real", year=reference_year)
            ref_problems = ref_dataset.load()
        else:
            raise ValueError("必須提供reference_data_path或reference_year之一")
        
        # 3. 建立評估器
        print(f"\n[INFO] 步驟3: 建立Win Rate評估器")
        evaluator = WinRateEvaluator(llm=self.llm, judge_model=judge_model)
        
        # 4. 執行評估
        print(f"\n[INFO] 步驟4: 開始成對對比")
        results = evaluator.evaluate_win_rate(
            gen_problems,
            ref_problems,
            num_comparisons=num_comparisons
        )
        
        # 5. 保存結果
        print(f"\n💾 步驟5: 保存評估結果")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"win_rate_results_{timestamp}.json")
        evaluator.export_results(results, result_file)
        
        # 6. 生成報告
        print(f"\n[INFO] 步驟6: 生成評估報告")
        report_file = os.path.join(output_dir, f"win_rate_report_{timestamp}.md")
        self._generate_report(results, report_file)
        
        print("\n" + "="*60)
        print("[OK] Win Rate評估完成")
        print("="*60)
        print(f"\n[INFO] 輸出檔案:")
        print(f"   - 評估結果: {result_file}")
        print(f"   - 評估報告: {report_file}")
        
        # 回傳簡化的結果
        return json.dumps({
            "status": "success",
            "metrics": results["metrics"],
            "result_file": result_file,
            "report_file": report_file
        }, ensure_ascii=False, indent=2)
    
    def _generate_report(self, results: Dict[str, Any], output_path: str):
        """生成Markdown評估報告"""
        metrics = results["metrics"]
        
        report = f"""# Win Rate評估報告

## 基本資訊

- **評估日期**: {results['evaluation_date']}
- **評委模型**: {results['judge_model']}
- **對比次數**: {metrics['total_comparisons']} 次

## 評估結果

### 勝率統計

| 指標 | 數值 | 百分比 |
|------|------|--------|
| 生成資料勝出 | {metrics['wins']} 次 | {metrics['win_rate']:.2%} |
| 參考資料勝出 | {metrics['losses']} 次 | {metrics['loss_rate']:.2%} |
| 平局 | {metrics['ties']} 次 | {metrics['tie_rate']:.2%} |

### 結果分析

**Win Rate**: {metrics['win_rate']:.2%}

{self._get_win_rate_analysis(metrics['win_rate'])}

## 詳細對比結果

"""
        
        # 添加前10次對比的詳細結果
        for idx, comparison in enumerate(results['comparisons'][:10]):
            winner_emoji = "[WIN]" if comparison['winner'] == "Generated" else "[LOSE]" if comparison['winner'] == "Reference" else "[TIE]"
            report += f"""
### 對比 {idx + 1}

- **生成題目**: {comparison['problem_a_id']}
- **參考題目**: {comparison['problem_b_id']}
- **勝者**: {winner_emoji} {comparison['winner']}
- **理由**: {comparison['reason']}
"""
        
        if len(results['comparisons']) > 10:
            report += f"\n*（僅顯示前10次對比的詳細結果，完整結果請查看JSON檔案）*\n"
        
        report += f"""
## 結論

{self._get_conclusion(metrics['win_rate'])}

---

*報告生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] 評估報告已保存: {output_path}")
    
    def _get_win_rate_analysis(self, win_rate: float) -> str:
        """根據勝率生成分析"""
        if win_rate >= 0.55:
            return """
[OK] **優秀**: 生成資料品質超過參考資料！這表明資料生成系統表現出色。
"""
        elif win_rate >= 0.45:
            return """
[OK] **良好**: 生成資料品質接近參考資料（差距<10%）。這是理想的結果，說明生成品質達到了真題水平。
"""
        elif win_rate >= 0.35:
            return """
[WARN] **合格**: 生成資料品質略低於參考資料，但仍在可接受范圍內。建議進一步優化生成策略。
"""
        else:
            return """
[ERROR] **需改進**: 生成資料品質明顯低於參考資料。建議檢查生成Pipeline並進行優化。
"""
    
    def _get_conclusion(self, win_rate: float) -> str:
        """根據勝率生成結論"""
        if win_rate >= 0.45:
            return f"""基於Win Rate評估，生成資料集的品質**接近或達到AIME真題水平**（Win Rate = {win_rate:.2%}）。

這證明了資料生成系統的有效性，生成的題目在品質上可以與真題相媲美。
"""
        else:
            return f"""基於Win Rate評估，生成資料集的品質**仍有提升空間**（Win Rate = {win_rate:.2%}）。

建議：
1. 優化題目生成的提示詞
2. 增加品質過濾步驟
3. 使用更強的生成模型
4. 增加人工審核環節
"""
