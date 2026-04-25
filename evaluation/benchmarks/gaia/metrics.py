"""
GAIA 評估指標模組

計算 GAIA 相關的評估指標
"""

from typing import Dict, Any, List, Optional
import numpy as np


class GAIAMetrics:
    """GAIA 評估指標計算器

    計算通用AI 助理相關的評估指標:
    - 精確匹配率 (Exact Match Rate): 答案完全正確的比例
    - 部分匹配率 (Partial Match Rate): 答案部分正確的比例
    - 按難度等級的成功率: Level 1/2/3 的表現
    - 平均推理步數: 解決問題所需的平均步數
    - 執行時間統計: 回應時間分析
    """

    @staticmethod
    def calculate_exact_match_rate(results: List[Dict[str, Any]]) -> float:
        """計算精確匹配率

        Args:
            results: 評估結果列表

        Returns:
            精確匹配率 (0-1)
        """
        if not results:
            return 0.0

        exact_matches = sum(1 for r in results if r.get("exact_match", False))
        return exact_matches / len(results)

    @staticmethod
    def calculate_partial_match_rate(results: List[Dict[str, Any]]) -> float:
        """計算部分匹配率

        Args:
            results: 評估結果列表

        Returns:
            部分匹配率 (0-1)
        """
        if not results:
            return 0.0

        partial_matches = sum(1 for r in results if r.get("partial_match", False))
        return partial_matches / len(results)

    @staticmethod
    def calculate_level_metrics(
        results: List[Dict[str, Any]],
        level: int
    ) -> Dict[str, float]:
        """計算特定難度等級的指標

        Args:
            results: 評估結果列表
            level: 難度等級 (1-3)

        Returns:
            該等級的指標字典
        """
        level_results = [r for r in results if r.get("level") == level]

        if not level_results:
            return {
                "total": 0,
                "exact_match_rate": 0.0,
                "partial_match_rate": 0.0,
                "average_score": 0.0
            }

        exact_matches = sum(1 for r in level_results if r.get("exact_match", False))
        partial_matches = sum(1 for r in level_results if r.get("partial_match", False))
        scores = [r.get("score", 0.0) for r in level_results]

        return {
            "total": len(level_results),
            "exact_match_rate": exact_matches / len(level_results),
            "partial_match_rate": partial_matches / len(level_results),
            "average_score": sum(scores) / len(scores) if scores else 0.0
        }

    @staticmethod
    def calculate_average_execution_time(results: List[Dict[str, Any]]) -> float:
        """計算平均執行時間

        Args:
            results: 評估結果列表

        Returns:
            平均執行時間(秒)
        """
        execution_times = [r.get("execution_time", 0.0) for r in results if "execution_time" in r]
        return sum(execution_times) / len(execution_times) if execution_times else 0.0

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """計算綜合指標

        Args:
            results: 評估結果列表

        Returns:
            完整的指標字典
        """
        if not results:
            return self._empty_metrics()

        # 基礎指標
        total = len(results)
        exact_match_rate = self.calculate_exact_match_rate(results)
        partial_match_rate = self.calculate_partial_match_rate(results)
        avg_execution_time = self.calculate_average_execution_time(results)

        # 分級指標
        level_metrics = {
            "Level_1": self.calculate_level_metrics(results, 1),
            "Level_2": self.calculate_level_metrics(results, 2),
            "Level_3": self.calculate_level_metrics(results, 3)
        }

        # 分數統計
        scores = [r.get("score", 0.0) for r in results]
        score_stats = self._compute_score_statistics(scores)

        # 性能分析
        performance_analysis = self._analyze_performance(results)

        return {
            "total_samples": total,
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "average_execution_time": avg_execution_time,
            "level_metrics": level_metrics,
            "score_statistics": score_stats,
            "performance_analysis": performance_analysis
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """回傳空指標"""
        return {
            "total_samples": 0,
            "exact_match_rate": 0.0,
            "partial_match_rate": 0.0,
            "average_execution_time": 0.0,
            "level_metrics": {
                "Level_1": {"total": 0, "exact_match_rate": 0.0, "partial_match_rate": 0.0, "average_score": 0.0},
                "Level_2": {"total": 0, "exact_match_rate": 0.0, "partial_match_rate": 0.0, "average_score": 0.0},
                "Level_3": {"total": 0, "exact_match_rate": 0.0, "partial_match_rate": 0.0, "average_score": 0.0}
            },
            "score_statistics": {},
            "performance_analysis": {}
        }

    def _compute_score_statistics(self, scores: List[float]) -> Dict[str, float]:
        """計算分數統計資訊"""
        if not scores:
            return {}

        return {
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores),
            "min": min(scores),
            "max": max(scores),
            "q1": np.percentile(scores, 25),
            "q3": np.percentile(scores, 75)
        }

    def _analyze_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析性能表現"""
        if not results:
            return {}

        # 按等級分組分析
        level_performance = {}
        for level in [1, 2, 3]:
            level_results = [r for r in results if r.get("level") == level]
            if level_results:
                exact_matches = sum(1 for r in level_results if r.get("exact_match", False))
                level_performance[f"Level_{level}"] = {
                    "sample_count": len(level_results),
                    "success_count": exact_matches,
                    "success_rate": exact_matches / len(level_results)
                }

        # 計算難度遞進表現
        difficulty_progression = self._analyze_difficulty_progression(level_performance)

        # 錯誤分析
        error_analysis = self._analyze_errors(results)

        return {
            "level_performance": level_performance,
            "difficulty_progression": difficulty_progression,
            "error_analysis": error_analysis
        }

    def _analyze_difficulty_progression(self, level_performance: Dict[str, Any]) -> Dict[str, Any]:
        """分析難度遞進表現"""
        progression = {}

        levels = ["Level_1", "Level_2", "Level_3"]
        for i in range(len(levels) - 1):
            current_level = levels[i]
            next_level = levels[i + 1]

            if current_level in level_performance and next_level in level_performance:
                current_rate = level_performance[current_level]["success_rate"]
                next_rate = level_performance[next_level]["success_rate"]

                progression[f"{current_level}_to_{next_level}"] = {
                    "drop_rate": current_rate - next_rate,
                    "relative_drop": (current_rate - next_rate) / current_rate if current_rate > 0 else 0
                }

        return progression

    def _analyze_errors(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析錯誤情況"""
        total_errors = sum(1 for r in results if not r.get("exact_match", False))
        partial_correct = sum(1 for r in results if r.get("partial_match", False) and not r.get("exact_match", False))
        complete_wrong = sum(1 for r in results if not r.get("partial_match", False) and not r.get("exact_match", False))

        return {
            "total_errors": total_errors,
            "partial_correct": partial_correct,
            "complete_wrong": complete_wrong,
            "error_rate": total_errors / len(results) if results else 0,
            "partial_correct_rate": partial_correct / total_errors if total_errors > 0 else 0
        }

    @staticmethod
    def compare_results(results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """比較兩個評估結果

        Args:
            results1: 第一個評估結果
            results2: 第二個評估結果

        Returns:
            比較結果字典
        """
        comparison = {
            "exact_match_rate_diff": results1.get("exact_match_rate", 0) - results2.get("exact_match_rate", 0),
            "partial_match_rate_diff": results1.get("partial_match_rate", 0) - results2.get("partial_match_rate", 0),
            "execution_time_diff": results1.get("average_execution_time", 0) - results2.get("average_execution_time", 0)
        }

        # 按等級比較
        level_comparison = {}
        for level in ["Level_1", "Level_2", "Level_3"]:
            if level in results1.get("level_metrics", {}) and level in results2.get("level_metrics", {}):
                level1 = results1["level_metrics"][level]
                level2 = results2["level_metrics"][level]
                level_comparison[level] = {
                    "exact_match_rate_diff": level1.get("exact_match_rate", 0) - level2.get("exact_match_rate", 0),
                    "score_diff": level1.get("average_score", 0) - level2.get("average_score", 0)
                }

        comparison["level_comparison"] = level_comparison

        return comparison

 