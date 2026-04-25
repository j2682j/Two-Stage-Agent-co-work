import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.llm import HelloAgentsLLM


class LLMJudgeEvaluator:
    """LLM Judge評估器"""
    
    # 評估維度
    EVALUATION_DIMENSIONS = [
        "correctness",      # 正確性
        "clarity",          # 清晰度
        "difficulty_match", # 難度匹配
        "completeness"      # 完整性
    ]
    
    def __init__(
        self,
        llm: Optional[HelloAgentsLLM] = None,
        judge_model: str = "gpt-4o"
    ):
        """
        初始化LLM Judge評估器
        
        Args:
            llm: LLM實例，如果為None則建立新實例
            judge_model: 評委模型名稱
        """
        self.llm = llm or HelloAgentsLLM(model=judge_model)
        self.judge_model = judge_model
        
    def evaluate_single(
        self,
        problem: Dict[str, Any],
        reference: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        評估單個問題
        
        Args:
            problem: 待評估的問題
            reference: 參考問題（可選，用於對比）
        
        Returns:
            評估結果，包含各維度評分和總分
        """
        start_time = time.time()
        
        # 建構評估提示詞
        prompt = self._build_evaluation_prompt(problem, reference)

        # 呼叫LLM進行評估
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages)
        
        # 解析評估結果
        scores = self._parse_evaluation_response(response)
        
        # 計算總分
        total_score = sum(scores.values()) / len(scores)
        
        execution_time = time.time() - start_time
        
        return {
            "problem_id": problem.get("problem_id", "unknown"),
            "scores": scores,
            "total_score": total_score,
            "evaluation_text": response,
            "execution_time": execution_time
        }
    
    def evaluate_batch(
        self,
        problems: List[Dict[str, Any]],
        references: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        批量評估問題
        
        Args:
            problems: 待評估的問題列表
            references: 參考問題列表（可選）
        
        Returns:
            評估結果匯總
        """
        print(f"\n[INFO] 開始LLM Judge評估")
        print(f"   評委模型: {self.judge_model}")
        print(f"   評估數量: {len(problems)}")
        print(f"   評估維度: {', '.join(self.EVALUATION_DIMENSIONS)}")
        
        results = []
        for idx, problem in enumerate(problems):
            print(f"\n   評估進度: {idx + 1}/{len(problems)}")
            
            reference = references[idx] if references and idx < len(references) else None
            result = self.evaluate_single(problem, reference)
            results.append(result)
            
            # 顯示評分
            print(f"   ✓ {problem.get('problem_id', 'unknown')}: {result['total_score']:.2f}/5.0")
        
        # 計算統計資訊
        metrics = self._compute_metrics(results)
        
        return {
            "results": results,
            "metrics": metrics,
            "evaluation_date": datetime.now().isoformat(),
            "judge_model": self.judge_model,
            "num_problems": len(problems)
        }
    
    def _build_evaluation_prompt(
        self,
        problem: Dict[str, Any],
        reference: Optional[Dict[str, Any]] = None
    ) -> str:
        """建構評估提示詞"""
        prompt = f"""你是一位專業的數學題目評估專家。請評估以下AIME風格數學題目的品質。

【待評估題目】
問題: {problem.get('problem', '')}
答案: {problem.get('answer', '')}
解答: {problem.get('solution', '')}
"""
        
        if reference:
            prompt += f"""
【參考題目（AIME真題）】
問題: {reference.get('problem', '')}
答案: {reference.get('answer', '')}
解答: {reference.get('solution', '')}
"""
        
        prompt += """
請從以下四個維度評估題目品質（每個維度1-5分）：

1. **正確性 (Correctness)**: 數學邏輯是否正確，答案是否準確
2. **清晰度 (Clarity)**: 問題表述是否清晰，解答是否易懂
3. **難度匹配 (Difficulty Match)**: 難度是否符合AIME標準（6-9/15）
4. **完整性 (Completeness)**: 解答步驟是否完整，是否包含必要的推理

請按以下JSON格式輸出評分：
```json
{
    "correctness": 5,
    "clarity": 4,
    "difficulty_match": 4,
    "completeness": 5,
    "comments": "詳細評價..."
}
```
"""
        return prompt
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, float]:
        """解析LLM評估回應"""
        try:
            # 提取JSON部分
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 提取評分
            scores = {}
            for dim in self.EVALUATION_DIMENSIONS:
                scores[dim] = float(data.get(dim, 3.0))  # 預設3分
            
            return scores
            
        except Exception as e:
            print(f"[WARN] 解析評估回應失敗: {e}")
            # 回傳預設評分
            return {dim: 3.0 for dim in self.EVALUATION_DIMENSIONS}
    
    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """計算評估指標"""
        if not results:
            return {}
        
        # 計算各維度平均分
        dimension_scores = {dim: [] for dim in self.EVALUATION_DIMENSIONS}
        total_scores = []
        
        for result in results:
            total_scores.append(result["total_score"])
            for dim in self.EVALUATION_DIMENSIONS:
                dimension_scores[dim].append(result["scores"][dim])
        
        metrics = {
            "average_total_score": sum(total_scores) / len(total_scores),
            "dimension_averages": {
                dim: sum(scores) / len(scores)
                for dim, scores in dimension_scores.items()
            },
            "pass_rate": sum(1 for s in total_scores if s >= 3.5) / len(total_scores),
            "excellent_rate": sum(1 for s in total_scores if s >= 4.5) / len(total_scores)
        }
        
        return metrics
    
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """匯出評估結果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 評估結果已保存: {output_path}")
