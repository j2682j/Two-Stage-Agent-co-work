"""
LLM Judge Evaluation Tool

使用 LLM 作為評估資料生成品質的工具
"""


import json
import os
from typing import Dict, Any
from datetime import datetime

from hello_agents.tools.base import Tool
from hello_agents.evaluation.benchmarks.data_generation.dataset import AIDataset
from hello_agents.evaluation.benchmarks.data_generation.llm_judge import LLMJudgeEvaluator
from hello_agents.core.llm import HelloAgentsLLM


class LLMJudgeTool(Tool):
    """LLM Judge評估工具"""
    
    def __init__(self, llm: HelloAgentsLLM = None):
        """
        初始化LLM Judge工具
        
        Args:
            llm: LLM實例，用於評估
        """
        super().__init__(
            name="llm_judge_evaluation",
            description="使用LLM作為評委評估資料生成品質"
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
                    "description": "參考資料的JSON檔案路徑（可選，用於對比）"
                },
                "reference_year": {
                    "type": "integer",
                    "description": "AIME真題年份（可選，如2024, 2025）"
                },
                "max_samples": {
                    "type": "integer",
                    "description": "最大評估樣本數（可選，預設評估所有）"
                },
                "output_dir": {
                    "type": "string",
                    "description": "輸出目錄（可選，預設為evaluation_results/llm_judge）"
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
        執行LLM Judge評估
        
        Args:
            params: 工具參數
        
        Returns:
            評估結果的JSON字串
        """
        # 解析參數
        generated_data_path = params["generated_data_path"]
        reference_data_path = params.get("reference_data_path")
        reference_year = params.get("reference_year")
        max_samples = params.get("max_samples")
        output_dir = params.get("output_dir", "evaluation_results/llm_judge")
        judge_model = params.get("judge_model", "gpt-4o")
        
        # 建立輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("[INFO] LLM Judge評估")
        print("="*60)
        
        # 1. 載入生成資料
        print(f"\n 步驟1: 載入生成資料")
        gen_dataset = AIDataset(dataset_type="generated", data_path=generated_data_path)
        gen_problems = gen_dataset.load()
        
        if max_samples:
            gen_problems = gen_problems[:max_samples]
            print(f"   限制評估樣本數: {max_samples}")
        
        # 2. 載入參考資料（可選）
        ref_problems = None
        if reference_data_path:
            print(f"\n 步驟2: 載入參考資料（本地檔案）")
            ref_dataset = AIDataset(dataset_type="generated", data_path=reference_data_path)
            ref_problems = ref_dataset.load()
        elif reference_year:
            print(f"\n 步驟2: 載入參考資料（AIME {reference_year}真題）")
            ref_dataset = AIDataset(dataset_type="real", year=reference_year)
            ref_problems = ref_dataset.load()
        else:
            print(f"\n[INFO] 步驟2: 跳過參考資料載入（無對比）")
        
        # 3. 建立評估器
        print(f"\n[INFO] 步驟3: 建立LLM Judge評估器")
        evaluator = LLMJudgeEvaluator(llm=self.llm, judge_model=judge_model)
        
        # 4. 執行評估
        print(f"\n[INFO] 步驟4: 開始評估")
        results = evaluator.evaluate_batch(gen_problems, ref_problems)
        
        # 5. 保存結果
        print(f"\n 步驟5: 保存評估結果")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"llm_judge_results_{timestamp}.json")
        evaluator.export_results(results, result_file)
        
        # 6. 生成報告
        print(f"\n[INFO] 步驟6: 生成評估報告")
        report_file = os.path.join(output_dir, f"llm_judge_report_{timestamp}.md")
        self._generate_report(results, report_file)
        
        print("\n" + "="*60)
        print("[OK] LLM Judge評估完成")
        print("="*60)
        print(f"\n[INFO] 輸出檔案:")
        print(f"   - 評估結果: {result_file}")
        print(f"   - 評估報告: {report_file}")
        
        # 回傳簡化的結果
        return json.dumps({
            "status": "success",
            "metrics": results["metrics"],
            "num_problems": results["num_problems"],
            "result_file": result_file,
            "report_file": report_file
        }, ensure_ascii=False, indent=2)
    
    def _generate_report(self, results: Dict[str, Any], output_path: str):
        """生成Markdown評估報告"""
        metrics = results["metrics"]
        
        report = f"""# LLM Judge評估報告

## 基本資訊

- **評估日期**: {results['evaluation_date']}
- **評委模型**: {results['judge_model']}
- **評估數量**: {results['num_problems']} 個題目

## 評估結果

### 總體評分

- **平均總分**: {metrics['average_total_score']:.2f}/5.0
- **通過率**: {metrics['pass_rate']:.2%} (≥3.5分)
- **優秀率**: {metrics['excellent_rate']:.2%} (≥4.5分)

### 各維度評分

| 維度 | 平均分 | 評級 |
|------|--------|------|
| 正確性 (Correctness) | {metrics['dimension_averages']['correctness']:.2f}/5.0 | {self._get_rating(metrics['dimension_averages']['correctness'])} |
| 清晰度 (Clarity) | {metrics['dimension_averages']['clarity']:.2f}/5.0 | {self._get_rating(metrics['dimension_averages']['clarity'])} |
| 難度匹配 (Difficulty Match) | {metrics['dimension_averages']['difficulty_match']:.2f}/5.0 | {self._get_rating(metrics['dimension_averages']['difficulty_match'])} |
| 完整性 (Completeness) | {metrics['dimension_averages']['completeness']:.2f}/5.0 | {self._get_rating(metrics['dimension_averages']['completeness'])} |

## 詳細結果

"""
        
        # 添加每個題目的詳細評分
        for idx, result in enumerate(results['results'][:10]):  # 只顯示前10個
            report += f"""
### 題目 {idx + 1}: {result['problem_id']}

- **總分**: {result['total_score']:.2f}/5.0
- **各維度評分**:
  - 正確性: {result['scores']['correctness']:.1f}
  - 清晰度: {result['scores']['clarity']:.1f}
  - 難度匹配: {result['scores']['difficulty_match']:.1f}
  - 完整性: {result['scores']['completeness']:.1f}
"""
        
        if len(results['results']) > 10:
            report += f"\n*（僅顯示前10個題目的詳細評分，完整結果請查看JSON檔案）*\n"
        
        report += f"""
## 結論

基於LLM Judge的評估，生成的資料集品質{'優秀' if metrics['average_total_score'] >= 4.5 else '良好' if metrics['average_total_score'] >= 3.5 else '需要改進'}。

---

*報告生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] 評估報告已保存: {output_path}")
    
    def _get_rating(self, score: float) -> str:
        """根據分數取得評級"""
        if score >= 4.5:
            return "優秀 ⭐⭐⭐⭐⭐"
        elif score >= 4.0:
            return "良好 ⭐⭐⭐⭐"
        elif score >= 3.5:
            return "合格 ⭐⭐⭐"
        elif score >= 3.0:
            return "一般 ⭐⭐"
        else:
            return "需改進 ⭐"
