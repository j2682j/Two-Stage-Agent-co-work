"""
?????

??????????????????
"""


import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from core.llm import HelloAgentsLLM


class WinRateEvaluator:
    """Win Rate評估器"""
    
    def __init__(
        self,
        llm: Optional[HelloAgentsLLM] = None,
        judge_model: str = "gpt-4o"
    ):
        """
        初始化Win Rate評估器
        
        Args:
            llm: LLM實例，如果為None則建立新實例
            judge_model: 評委模型名稱
        """
        self.llm = llm or HelloAgentsLLM(model=judge_model)
        self.judge_model = judge_model
        
    def compare_pair(
        self,
        problem_a: Dict[str, Any],
        problem_b: Dict[str, Any],
        label_a: str = "A",
        label_b: str = "B"
    ) -> Dict[str, Any]:
        """
        對比兩個問題，判斷哪個更好
        
        Args:
            problem_a: 問題A
            problem_b: 問題B
            label_a: 問題A的標簽
            label_b: 問題B的標簽
        
        Returns:
            對比結果，包含勝者和理由
        """
        start_time = time.time()
        
        # 建構對比提示詞
        prompt = self._build_comparison_prompt(problem_a, problem_b, label_a, label_b)

        # 呼叫LLM進行對比
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        
        # 解析對比結果
        winner, reason = self._parse_comparison_response(response, label_a, label_b)
        
        execution_time = time.time() - start_time
        
        return {
            "problem_a_id": problem_a.get("problem_id", "unknown"),
            "problem_b_id": problem_b.get("problem_id", "unknown"),
            "winner": winner,
            "reason": reason,
            "comparison_text": response,
            "execution_time": execution_time
        }
    
    def evaluate_win_rate(
        self,
        generated_problems: List[Dict[str, Any]],
        reference_problems: List[Dict[str, Any]],
        num_comparisons: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        評估生成資料相對於參考資料的勝率
        
        Args:
            generated_problems: 生成的問題列表
            reference_problems: 參考問題列表（如AIME真題）
            num_comparisons: 對比次數，如果為None則對比所有可能的配對
        
        Returns:
            勝率評估結果
        """
        print(f"\n🏆 開始Win Rate評估")
        print(f"   評委模型: {self.judge_model}")
        print(f"   生成資料: {len(generated_problems)} 個")
        print(f"   參考資料: {len(reference_problems)} 個")
        
        # 確定對比次數
        if num_comparisons is None:
            num_comparisons = min(len(generated_problems), len(reference_problems))

        # 限制對比次數不超過生成題目數量
        num_comparisons = min(num_comparisons, len(generated_problems))

        print(f"   對比次數: {num_comparisons}")

        # 隨機采樣生成題目索引
        import random
        gen_indices = random.sample(range(len(generated_problems)), num_comparisons)

        print(f"   采樣方式: 隨機采樣")

        # 進行成對對比
        comparisons = []
        wins = 0
        losses = 0
        ties = 0

        for i, gen_idx in enumerate(gen_indices):
            gen_problem = generated_problems[gen_idx]
            # 隨機選擇一個參考題目
            ref_idx = random.randint(0, len(reference_problems) - 1)
            ref_problem = reference_problems[ref_idx]

            print(f"\n   對比進度: {i + 1}/{num_comparisons}")
            print(f"   生成題目: #{gen_idx + 1}, 參考題目: #{ref_idx + 1}")

            # 隨機化題目順序以避免位置偏向
            if random.random() < 0.5:
                # Generated在前
                result = self.compare_pair(
                    gen_problem,
                    ref_problem,
                    label_a="Problem A",
                    label_b="Problem B"
                )
                # 紀錄實際順序
                result["actual_order"] = {"A": "Generated", "B": "Reference"}

                # 轉換winner
                if result["winner"] == "Problem A":
                    actual_winner = "Generated"
                elif result["winner"] == "Problem B":
                    actual_winner = "Reference"
                else:
                    actual_winner = "Tie"
            else:
                # Reference在前
                result = self.compare_pair(
                    ref_problem,
                    gen_problem,
                    label_a="Problem A",
                    label_b="Problem B"
                )
                # 紀錄實際順序
                result["actual_order"] = {"A": "Reference", "B": "Generated"}

                # 轉換winner
                if result["winner"] == "Problem A":
                    actual_winner = "Reference"
                elif result["winner"] == "Problem B":
                    actual_winner = "Generated"
                else:
                    actual_winner = "Tie"

            result["actual_winner"] = actual_winner
            comparisons.append(result)

            # 統計勝負
            if actual_winner == "Generated":
                wins += 1
                print(f"   ✓ Generated勝出")
            elif actual_winner == "Reference":
                losses += 1
                print(f"   ✗ Reference勝出")
            else:
                ties += 1
                print(f"   = 平局")
        
        # 計算勝率
        win_rate = wins / num_comparisons if num_comparisons > 0 else 0
        loss_rate = losses / num_comparisons if num_comparisons > 0 else 0
        tie_rate = ties / num_comparisons if num_comparisons > 0 else 0
        
        metrics = {
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "tie_rate": tie_rate,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total_comparisons": num_comparisons
        }
        
        print(f"\n[INFO] Win Rate評估結果:")
        print(f"   勝率: {win_rate:.2%}")
        print(f"   敗率: {loss_rate:.2%}")
        print(f"   平局率: {tie_rate:.2%}")
        
        return {
            "comparisons": comparisons,
            "metrics": metrics,
            "evaluation_date": datetime.now().isoformat(),
            "judge_model": self.judge_model
        }
    
    def _build_comparison_prompt(
        self,
        problem_a: Dict[str, Any],
        problem_b: Dict[str, Any],
        label_a: str,
        label_b: str
    ) -> str:
        """建構對比提示詞"""
        # 檢查是否有solution字段
        has_solution_a = bool(problem_a.get('solution', '').strip())
        has_solution_b = bool(problem_b.get('solution', '').strip())

        # 建構題目展示
        problem_a_text = f"""**{label_a}**
Problem: {problem_a.get('problem', '')}
Answer: {problem_a.get('answer', '')}"""

        if has_solution_a:
            problem_a_text += f"\nSolution: {problem_a.get('solution', '')}"

        problem_b_text = f"""**{label_b}**
Problem: {problem_b.get('problem', '')}
Answer: {problem_b.get('answer', '')}"""

        if has_solution_b:
            problem_b_text += f"\nSolution: {problem_b.get('solution', '')}"

        # 根據是否有solution調整評估維度
        if has_solution_a and has_solution_b:
            criteria = """**Evaluation Criteria:**
Please evaluate comprehensively from the following dimensions:
1. **Mathematical Correctness**: Are the problem, solution, and answer mathematically correct?
2. **Clarity**: Is the problem statement clear and unambiguous?
3. **Difficulty Appropriateness**: Does the difficulty match AIME standards (challenging but solvable)?
4. **Solution Completeness**: Is the solution complete with clear reasoning steps?"""
        else:
            criteria = """**Evaluation Criteria:**
Please evaluate comprehensively from the following dimensions:
1. **Mathematical Correctness**: Are the problem and answer mathematically correct and reasonable?
2. **Clarity**: Is the problem statement clear and unambiguous?
3. **Difficulty Appropriateness**: Does the difficulty match AIME standards (challenging but solvable)?
4. **Problem Quality**: Is the problem well-designed with appropriate complexity?

Note: Some problems may not have solutions provided. Focus on the problem statement and answer quality."""

        prompt = f"""You are a professional mathematics problem evaluator. Please compare the following two AIME-style math problems and determine which one has higher quality.

{problem_a_text}

{problem_b_text}

{criteria}

**Important Guidelines:**
- Be objective and fair in your evaluation
- Consider all dimensions equally
- If both problems are of similar quality, choose "Tie"
- Do not favor one problem just because it appears first or second
- If one problem has a solution and the other doesn't, focus on the problem statement and answer quality

Please output your judgment in the following JSON format:
```json
{{
    "winner": "{label_a}",  // or "{label_b}" or "Tie"
    "reason": "Detailed explanation of why you chose this answer, covering the evaluation dimensions..."
}}
```
"""
        return prompt
    
    def _parse_comparison_response(
        self,
        response: str,
        label_a: str,
        label_b: str
    ) -> Tuple[str, str]:
        """解析對比回應"""
        try:
            # 提取JSON部分
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            # 修復LaTeX轉義問題
            import re
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # 修復LaTeX轉義：將 \frac 轉為 \\frac
                fixed_json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)
                data = json.loads(fixed_json_str)
            
            winner = data.get("winner", "Tie")
            reason = data.get("reason", "No reason provided")
            
            # 驗證winner是否有效
            if winner not in [label_a, label_b, "Tie"]:
                winner = "Tie"
            
            return winner, reason
            
        except Exception as e:
            print(f"[WARN] 解析對比回應失敗: {e}")
            return "Tie", "Failed to parse response"
    
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """匯出評估結果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Win Rate結果已保存: {output_path}")
