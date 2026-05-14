"""
GAIA 評估指標模組

計算 GAIA 相關的評估指標
"""

from typing import Dict, Any, List, Optional
import numpy as np


class GAIAMetrics:
    """
    負責在 evaluation.benchmarks.gaia.metrics 中封裝 GAIAMetrics，封裝 benchmark 評估、答案判定、分數計算或報告資料整理流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    @staticmethod
    def calculate_exact_match_rate(results: List[Dict[str, Any]]) -> float:
        """
        負責執行 GAIAMetrics 中的 calculate_exact_match_rate 流程，依照 GAIAMetrics 的流程需求處理 calculate_exact_match_rate 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not results:
            return 0.0

        exact_matches = sum(1 for r in results if r.get("exact_match", False))
        return exact_matches / len(results)

    @staticmethod
    def calculate_partial_match_rate(results: List[Dict[str, Any]]) -> float:
        """
        負責執行 GAIAMetrics 中的 calculate_partial_match_rate 流程，依照 GAIAMetrics 的流程需求處理 calculate_partial_match_rate 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 GAIAMetrics 中的 calculate_level_metrics 流程，依照 GAIAMetrics 的流程需求處理 calculate_level_metrics 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
            level: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 GAIAMetrics 中的 calculate_average_execution_time 流程，依照 GAIAMetrics 的流程需求處理 calculate_average_execution_time 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        execution_times = [r.get("execution_time", 0.0) for r in results if "execution_time" in r]
        return sum(execution_times) / len(execution_times) if execution_times else 0.0

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        負責執行 GAIAMetrics 中的 compute_metrics 流程，依照 GAIAMetrics 的流程需求處理 compute_metrics 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 GAIAMetrics 中的 _empty_metrics 流程，依照 GAIAMetrics 的流程需求處理 _empty_metrics 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 GAIAMetrics 中的 _compute_score_statistics 流程，依照 GAIAMetrics 的流程需求處理 _compute_score_statistics 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            scores: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 GAIAMetrics 中的 _analyze_performance 流程，依照 GAIAMetrics 的流程需求處理 _analyze_performance 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 GAIAMetrics 中的 _analyze_difficulty_progression 流程，依照 GAIAMetrics 的流程需求處理 _analyze_difficulty_progression 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            level_performance: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 GAIAMetrics 中的 _analyze_errors 流程，依照 GAIAMetrics 的流程需求處理 _analyze_errors 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 GAIAMetrics 中的 compare_results 流程，依照 GAIAMetrics 的流程需求處理 compare_results 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results1: 此流程需要使用的輸入資料。
            results2: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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

 