"""
BFCL 評估指標模組

計算 BFCL 相關的評估指標
"""

from typing import Dict, Any, List, Optional
import json
import ast
import numpy as np


class BFCLMetrics:
    """BFCL 評估指標計算器

    計算工具呼叫相關的評估指標:
    - 正確率 (Accuracy): 完全正確的比例
    - AST 匹配度 (AST Match): 抽象語法樹匹配度
    - 參數正確率 (Parameter Accuracy): 參數正確的比例
    - F1分數: 精確率和召回率的調和平均
    - 執行成功率: 可執行函式呼叫的成功率
    """

    @staticmethod
    def calculate_accuracy(predictions: List[Any], references: List[Any]) -> float:
        """計算正確率

        Args:
            predictions: 預測結果列表
            references: 參考答案列表

        Returns:
            正確率 (0-1)
        """
        if not predictions or not references:
            return 0.0

        min_len = min(len(predictions), len(references))
        correct = sum(1 for p, r in zip(predictions[:min_len], references[:min_len]) if p == r)
        return correct / min_len

    @staticmethod
    def calculate_ast_match(predicted: str, expected: str) -> float:
        """計算 AST 匹配度

        Args:
            predicted: 預測的函式呼叫
            expected: 期望的函式呼叫

        Returns:
            匹配度 (0-1)
        """
        try:
            # 嘗試解析為AST
            pred_ast = ast.parse(predicted, mode='eval')
            exp_ast = ast.parse(expected, mode='eval')

            # 比較AST結構
            pred_dump = ast.dump(pred_ast)
            exp_dump = ast.dump(exp_ast)

            if pred_dump == exp_dump:
                return 1.0

            # 計算結構相似度
            similarity = BFCLMetrics._calculate_string_similarity(pred_dump, exp_dump)
            return similarity

        except SyntaxError:
            # 如果無法解析，使用字串相似度
            return BFCLMetrics._calculate_string_similarity(predicted, expected)

    @staticmethod
    def _calculate_string_similarity(s1: str, s2: str) -> float:
        """計算字串相似度（簡化版Levenshtein距離）"""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # 使用集合交集計算相似度
        set1 = set(s1.split())
        set2 = set(s2.split())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def calculate_parameter_accuracy(
        predicted_params: Dict[str, Any],
        expected_params: Dict[str, Any]
    ) -> float:
        """計算參數正確率

        Args:
            predicted_params: 預測的參數
            expected_params: 期望的參數

        Returns:
            參數正確率 (0-1)
        """
        if not expected_params:
            return 1.0 if not predicted_params else 0.0

        if not predicted_params:
            return 0.0

        correct = 0
        for key, expected_value in expected_params.items():
            if key in predicted_params:
                predicted_value = predicted_params[key]
                if BFCLMetrics._values_match(predicted_value, expected_value):
                    correct += 1

        return correct / len(expected_params)

    @staticmethod
    def _values_match(v1: Any, v2: Any) -> bool:
        """比較兩個值是否匹配"""
        # 處理數值類型
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return abs(v1 - v2) < 1e-6

        # 處理字串類型
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.strip().lower() == v2.strip().lower()

        # 處理列表類型
        if isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                return False
            return all(BFCLMetrics._values_match(a, b) for a, b in zip(v1, v2))

        # 處理字典類型
        if isinstance(v1, dict) and isinstance(v2, dict):
            if set(v1.keys()) != set(v2.keys()):
                return False
            return all(BFCLMetrics._values_match(v1[k], v2[k]) for k in v1.keys())

        # 預設使用相等比較
        return v1 == v2

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """計算綜合指標

        Args:
            results: 評估結果列表

        Returns:
            指標字典，包含各種評估指標
        """
        if not results:
            return self._empty_metrics()

        total = len(results)

        # 基礎指標
        success_count = sum(1 for r in results if r.get("success", False))
        accuracy = success_count / total

        # 分數統計
        scores = [r.get("score", 0.0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # 執行時間統計
        execution_times = [r.get("execution_time", 0.0) for r in results if "execution_time" in r]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

        # 按類別統計
        category_metrics = self._compute_category_metrics(results)

        # 函式呼叫統計
        function_call_stats = self._compute_function_call_stats(results)

        return {
            "total_samples": total,
            "success_count": success_count,
            "accuracy": accuracy,
            "average_score": avg_score,
            "average_execution_time": avg_execution_time,
            "category_metrics": category_metrics,
            "function_call_stats": function_call_stats,
            "score_distribution": self._compute_score_distribution(scores)
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """回傳空指標"""
        return {
            "total_samples": 0,
            "success_count": 0,
            "accuracy": 0.0,
            "average_score": 0.0,
            "average_execution_time": 0.0,
            "category_metrics": {},
            "function_call_stats": {},
            "score_distribution": {}
        }

    def _compute_category_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """計算分類別指標"""
        categories = {}

        for result in results:
            category = result.get("category", "unknown")
            if category not in categories:
                categories[category] = {
                    "total": 0,
                    "success": 0,
                    "scores": []
                }

            categories[category]["total"] += 1
            if result.get("success", False):
                categories[category]["success"] += 1
            categories[category]["scores"].append(result.get("score", 0.0))

        # 計算每個類別的統計資訊
        category_metrics = {}
        for category, stats in categories.items():
            accuracy = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0

            category_metrics[category] = {
                "total": stats["total"],
                "success": stats["success"],
                "accuracy": accuracy,
                "average_score": avg_score
            }

        return category_metrics

    def _compute_function_call_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """計算函式呼叫統計"""
        total_calls = 0
        successful_calls = 0
        function_names = set()

        for result in results:
            predicted = result.get("predicted", [])
            if isinstance(predicted, list):
                total_calls += len(predicted)
                for call in predicted:
                    if isinstance(call, dict) and "name" in call:
                        function_names.add(call["name"])
                        if result.get("success", False):
                            successful_calls += 1

        return {
            "total_function_calls": total_calls,
            "successful_calls": successful_calls,
            "unique_functions": len(function_names),
            "function_names": sorted(list(function_names)),
            "avg_calls_per_sample": total_calls / len(results) if results else 0.0
        }

    def _compute_score_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """計算分數分布"""
        if not scores:
            return {}

        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2],
            "std": np.std(scores) if len(scores) > 1 else 0.0,
            "quartiles": {
                "q1": sorted(scores)[len(scores) // 4],
                "q2": sorted(scores)[len(scores) // 2],
                "q3": sorted(scores)[3 * len(scores) // 4]
            }
        }

    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """計算F1分數

        Args:
            precision: 精確率
            recall: 召回率

        Returns:
            F1分數
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_precision_recall(
        predicted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> tuple[float, float]:
        """計算精確率和召回率

        Args:
            predicted: 預測的函式呼叫列表
            expected: 期望的函式呼叫列表

        Returns:
            (precision, recall) 元組
        """
        if not expected:
            return 1.0 if not predicted else 0.0, 1.0

        if not predicted:
            return 0.0, 0.0

        # 簡化版本：基於函式名匹配
        pred_names = set(call.get("name", "") for call in predicted if isinstance(call, dict))
        exp_names = set(call.get("name", "") for call in expected if isinstance(call, dict))

        true_positives = len(pred_names & exp_names)

        precision = true_positives / len(pred_names) if pred_names else 0.0
        recall = true_positives / len(exp_names) if exp_names else 0.0

        return precision, recall

 