"""BFCL評估工具

Berkeley Function Calling Leaderboard (BFCL) 一鍵評估工具

本工具封裝了完整的BFCL評估流程：
1. 自動檢查和準備BFCL資料
2. 執行HelloAgents評估
3. 匯出BFCL格式結果
4. 呼叫BFCL官方評估工具（可選）
5. 生成評估報告

使用範例：
    from hello_agents import SimpleAgent, HelloAgentsLLM
    from hello_agents.tools.builtin import BFCLEvaluationTool

    # 建立智慧代理
    llm = HelloAgentsLLM()
    agent = SimpleAgent(name="TestAgent", llm=llm)

    # 建立評估工具
    bfcl_tool = BFCLEvaluationTool()

    # 執行評估（預設會執行BFCL官方評估）
    results = bfcl_tool.run(
        agent=agent,
        category="simple_python",
        max_samples=5
    )

    print(f"正確率: {results['overall_accuracy']:.2%}")
    # 報告自動生成到: evaluation_reports/bfcl_report_{timestamp}.md
"""

import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..base import Tool, ToolParameter


class BFCLEvaluationTool(Tool):
    """BFCL一鍵評估工具

    封裝了完整的BFCL評估流程，提供簡單易用的介面。

    支援的評估類別：
    - simple_python: 簡單Python函式呼叫（400樣本）
    - simple_java: 簡單Java函式呼叫（400樣本）
    - simple_javascript: 簡單JavaScript函式呼叫（400樣本）
    - multiple: 多函式呼叫（240樣本）
    - parallel: 平行函式呼叫（280樣本）
    - parallel_multiple: 平行多函式呼叫（200樣本）
    - irrelevance: 無關檢測（200樣本）
    """

    def __init__(self, bfcl_data_dir: Optional[str] = None, project_root: Optional[str] = None):
        """初始化BFCL評估工具

        Args:
            bfcl_data_dir: BFCL資料目錄路徑（預設：./temp_gorilla/berkeley-function-call-leaderboard/bfcl_eval/data）
            project_root: 項目根目錄（預設：目前目錄）
        """
        super().__init__(
            name="bfcl_evaluation",
            description=(
                "BFCL一鍵評估工具。評估智慧代理的工具呼叫能力，支援多個評估類別。"
                "自動完成資料載入、評估執行、結果匯出和報告生成。"
            )
        )

        # 設定路徑
        self.project_root = Path(project_root) if project_root else Path.cwd()
        if bfcl_data_dir:
            self.bfcl_data_dir = Path(bfcl_data_dir)
        else:
            self.bfcl_data_dir = self.project_root / "temp_gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"

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
                name="category",
                type="string",
                description="評估類別：simple_python, simple_java, simple_javascript, multiple, parallel, parallel_multiple, irrelevance",
                required=False,
                default="simple_python"
            ),
            ToolParameter(
                name="max_samples",
                type="integer",
                description="評估樣本數（預設：5，設為0表示全部）",
                required=False,
                default=5
            ),
            ToolParameter(
                name="run_official_eval",
                type="boolean",
                description="是否執行BFCL官方評估",
                required=False,
                default=True
            ),
            ToolParameter(
                name="model_name",
                type="string",
                description="模型名稱（用於BFCL官方評估）",
                required=False,
                default="Qwen/Qwen3-8B"
            )
        ]

    def run(self, agent: Any, category: str = "simple_python", max_samples: int = 5,
            run_official_eval: bool = True, model_name: Optional[str] = None) -> Dict[str, Any]:
        """執行BFCL評估

        Args:
            agent: 要評估的智慧代理
            category: 評估類別（預設：simple_python）
            max_samples: 評估樣本數（預設：5，設為0表示全部）
            run_official_eval: 是否執行BFCL官方評估（預設：True）
            model_name: 模型名稱（用於BFCL官方評估，預設：Qwen/Qwen3-8B）

        Returns:
            評估結果字典，包含：
            - overall_accuracy: 總體正確率
            - correct_samples: 正確樣本數
            - total_samples: 總樣本數
            - category_metrics: 分類指標
            - detailed_results: 詳細結果
        """
        from hello_agents.evaluation import BFCLDataset, BFCLEvaluator

        print("\n" + "="*60)
        print("BFCL一鍵評估")
        print("="*60)
        print(f"\n設定:")
        print(f"   評估類別: {category}")
        print(f"   樣本數量: {max_samples if max_samples > 0 else '全部'}")
        print(f"   智慧代理: {getattr(agent, 'name', 'Unknown')}")

        # 步驟1: 檢查BFCL資料
        if not self._check_bfcl_data():
            return self._create_error_result("BFCL資料目錄不存在")

        # 步驟2: 執行HelloAgents評估
        print("\n" + "="*60)
        print("步驟1: 執行HelloAgents評估")
        print("="*60)

        dataset = BFCLDataset(bfcl_data_dir=str(self.bfcl_data_dir), category=category)
        evaluator = BFCLEvaluator(dataset=dataset, category=category)

        if max_samples > 0:
            results = evaluator.evaluate(agent, max_samples=max_samples)
        else:
            results = evaluator.evaluate(agent, max_samples=None)

        print(f"\n[INFO] 評估結果:")
        print(f"   正確率: {results['overall_accuracy']:.2%}")
        print(f"   正確數: {results['correct_samples']}/{results['total_samples']}")

        # 步驟3: 匯出BFCL格式結果
        print("\n" + "="*60)
        print("步驟2: 匯出BFCL格式結果")
        print("="*60)

        output_dir = self.project_root / "evaluation_results" / "bfcl_official"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"BFCL_v4_{category}_result.json"

        evaluator.export_to_bfcl_format(results, output_file)

        # 步驟4: 執行BFCL官方評估（可選）
        if run_official_eval:
            if not model_name:
                model_name = "Qwen/Qwen3-8B"

            self._run_official_evaluation(output_file, model_name, category)

        # 步驟5: 生成評估報告
        print("\n" + "="*60)
        print("步驟3: 生成評估報告")
        print("="*60)

        # 添加智慧代理和類別資訊到結果中
        results['agent_name'] = getattr(agent, 'name', 'Unknown')
        results['category'] = category

        self.generate_report(results)

        return results

    def _check_bfcl_data(self) -> bool:
        """檢查BFCL資料是否存在"""
        if not self.bfcl_data_dir.exists():
            print(f"\n[ERROR] BFCL資料目錄不存在: {self.bfcl_data_dir}")
            print(f"\n請先克隆BFCL倉庫：")
            print(f"   git clone --depth 1 https://github.com/ShishirPatil/gorilla.git temp_gorilla")
            return False
        return True

    def _run_official_evaluation(self, source_file: Path, model_name: str, category: str):
        """執行BFCL官方評估"""
        print("\n" + "="*60)
        print("步驟3: 執行BFCL官方評估")
        print("="*60)

        # 復制結果檔案到BFCL結果目錄
        safe_model_name = model_name.replace("/", "_")
        result_dir = self.project_root / "result" / safe_model_name
        result_dir.mkdir(parents=True, exist_ok=True)

        target_file = result_dir / f"BFCL_v4_{category}_result.json"
        shutil.copy(source_file, target_file)

        print(f"\n[OK] 結果檔案已復制到:")
        print(f"   {target_file}")

        # 執行BFCL評估
        try:
            import os
            os.environ['PYTHONUTF8'] = '1'

            cmd = [
                "bfcl", "evaluate",
                "--model", model_name,
                "--test-category", category,
                "--partial-eval"
            ]

            print(f"\n🔄 執行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            if result.stdout:
                print(result.stdout)

            if result.returncode != 0:
                print(f"\n[ERROR] BFCL評估失敗:")
                if result.stderr:
                    print(result.stderr)
            else:
                self._show_official_results(model_name, category)

        except FileNotFoundError:
            print("\n[ERROR] 找不到bfcl命令")
            print("   請先安裝: pip install bfcl-eval")
        except Exception as e:
            print(f"\n[ERROR] 執行BFCL評估時出錯: {e}")

    def _show_official_results(self, model_name: str, category: str):
        """展示BFCL官方評估結果"""
        print("\n" + "="*60)
        print("BFCL官方評估結果")
        print("="*60)

        # CSV檔案
        csv_file = self.project_root / "score" / "data_non_live.csv"

        if csv_file.exists():
            print(f"\n[INFO] 評估結果匯總:")
            with open(csv_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)

        # 詳細評分檔案
        safe_model_name = model_name.replace("/", "_")
        score_file = self.project_root / "score" / safe_model_name / "non_live" / f"BFCL_v4_{category}_score.json"

        if score_file.exists():
            print(f"\n[INFO] 詳細評分檔案:")
            print(f"   {score_file}")

            with open(score_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                summary = json.loads(first_line)
                print(f"\n[INFO] 最終結果:")
                print(f"   正確率: {summary['accuracy']:.2%}")
                print(f"   正確數: {summary['correct_count']}/{summary['total_count']}")

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """建立錯誤結果"""
        return {
            "error": error_message,
            "overall_accuracy": 0.0,
            "correct_samples": 0,
            "total_samples": 0,
            "category_metrics": {},
            "detailed_results": []
        }

    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """生成評估報告

        Args:
            results: 評估結果字典
            output_file: 輸出檔案路徑（可選，預設：evaluation_reports/bfcl_report_{timestamp}.md）

        Returns:
            報告內容（Markdown格式）
        """
        from datetime import datetime

        # 生成報告內容
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# BFCL評估報告

**生成時間**: {timestamp}

## [INFO] 評估概覽

- **智慧代理**: {results.get('agent_name', 'Unknown')}
- **評估類別**: {results.get('category', 'Unknown')}
- **總體正確率**: {results['overall_accuracy']:.2%}
- **正確樣本數**: {results['correct_samples']}/{results['total_samples']}

## 📈 詳細指標

"""

        # 添加分類指標
        if 'category_metrics' in results and results['category_metrics']:
            report += "### 分類正確率\n\n"
            for category, metrics in results['category_metrics'].items():
                accuracy = metrics.get('accuracy', 0.0)
                correct = metrics.get('correct', 0)
                total = metrics.get('total', 0)
                report += f"- **{category}**: {accuracy:.2%} ({correct}/{total})\n"
            report += "\n"

        # 添加樣本詳情
        if 'detailed_results' in results and results['detailed_results']:
            report += "## [INFO] 樣本詳情\n\n"
            report += "| 樣本ID | 問題 | 預測結果 | 正確答案 | 是否正確 |\n"
            report += "|--------|------|----------|----------|----------|\n"

            for detail in results['detailed_results'][:10]:  # 只顯示前10個
                sample_id = detail.get('sample_id', 'N/A')

                # 提取問題文字
                question = detail.get('question', 'N/A')
                if isinstance(question, list) and len(question) > 0:
                    if isinstance(question[0], list) and len(question[0]) > 0:
                        if isinstance(question[0][0], dict) and 'content' in question[0][0]:
                            question = question[0][0]['content']
                question_str = str(question)
                if len(question_str) > 60:
                    question_str = question_str[:60] + "..."

                # 提取預測結果（字段名是predicted）
                prediction = detail.get('predicted', 'N/A')
                if prediction and prediction != 'N/A':
                    pred_str = str(prediction)
                    if len(pred_str) > 40:
                        pred_str = pred_str[:40] + "..."
                else:
                    pred_str = "N/A"

                # 提取正確答案（字段名是expected）
                ground_truth = detail.get('expected', 'N/A')
                if ground_truth and ground_truth != 'N/A':
                    gt_str = str(ground_truth)
                    if len(gt_str) > 40:
                        gt_str = gt_str[:40] + "..."
                else:
                    gt_str = "N/A"

                # 判斷是否正確（字段名是success）
                is_correct = "[OK]" if detail.get('success', False) else "[ERROR]"

                report += f"| {sample_id} | {question_str} | {pred_str} | {gt_str} | {is_correct} |\n"

            if len(results['detailed_results']) > 10:
                report += f"\n*顯示前10個樣本，共{len(results['detailed_results'])}個樣本*\n"
            report += "\n"

        # 添加可視化（ASCII圖表）
        report += "## [INFO] 正確率可視化\n\n"
        report += "```\n"
        accuracy = results['overall_accuracy']
        bar_length = int(accuracy * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        report += f"正確率: {bar} {accuracy:.2%}\n"
        report += "```\n\n"

        # 添加建議
        report += "## [INFO] 建議\n\n"
        if accuracy >= 0.9:
            report += "- [OK] 表現優秀！智慧代理在工具呼叫方面表現出色。\n"
        elif accuracy >= 0.7:
            report += "- [WARN] 表現良好，但仍有提升空間。建議檢查錯誤樣本，優化提示詞。\n"
        else:
            report += "- [ERROR] 表現需要改進。建議：\n"
            report += "  1. 檢查智慧代理的工具呼叫邏輯\n"
            report += "  2. 優化系統提示詞\n"
            report += "  3. 增加更多訓練樣本\n"

        # 保存報告
        if output_file is None:
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.project_root / "evaluation_reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"bfcl_report_{timestamp_file}.md"
        else:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 報告已生成: {output_file}")

        return report
