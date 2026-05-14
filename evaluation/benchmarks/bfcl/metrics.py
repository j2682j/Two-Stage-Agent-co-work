"""
BFCL 評估指標模組

計算 BFCL 相關的評估指標
"""

from typing import Dict, Any, List, Optional
import json
import ast
import numpy as np


class BFCLMetrics:
    """
    負責在 evaluation.benchmarks.bfcl.metrics 中封裝 BFCLMetrics，封裝 benchmark 評估、答案判定、分數計算或報告資料整理流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    @staticmethod
    def calculate_accuracy(predictions: List[Any], references: List[Any]) -> float:
        """
        負責執行 BFCLMetrics 中的 calculate_accuracy 流程，依照 BFCLMetrics 的流程需求處理 calculate_accuracy 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            predictions: 此流程需要使用的輸入資料。
            references: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not predictions or not references:
            return 0.0

        min_len = min(len(predictions), len(references))
        correct = sum(1 for p, r in zip(predictions[:min_len], references[:min_len]) if p == r)
        return correct / min_len

    @staticmethod
    def calculate_ast_match(predicted: str, expected: str) -> float:
        """
        負責執行 BFCLMetrics 中的 calculate_ast_match 流程，依照 BFCLMetrics 的流程需求處理 calculate_ast_match 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            predicted: 此流程需要使用的輸入資料。
            expected: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 BFCLMetrics 中的 _calculate_string_similarity 流程，依照 BFCLMetrics 的流程需求處理 _calculate_string_similarity 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            s1: 此流程需要使用的輸入資料。
            s2: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 BFCLMetrics 中的 calculate_parameter_accuracy 流程，依照 BFCLMetrics 的流程需求處理 calculate_parameter_accuracy 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            predicted_params: 此流程需要使用的輸入資料。
            expected_params: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 BFCLMetrics 中的 _values_match 流程，依照 BFCLMetrics 的流程需求處理 _values_match 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            v1: 此流程需要使用的輸入資料。
            v2: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 BFCLMetrics 中的 compute_metrics 流程，依照 BFCLMetrics 的流程需求處理 compute_metrics 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 BFCLMetrics 中的 _empty_metrics 流程，依照 BFCLMetrics 的流程需求處理 _empty_metrics 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 BFCLMetrics 中的 _compute_category_metrics 流程，依照 BFCLMetrics 的流程需求處理 _compute_category_metrics 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 BFCLMetrics 中的 _compute_function_call_stats 流程，依照 BFCLMetrics 的流程需求處理 _compute_function_call_stats 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 BFCLMetrics 中的 _compute_score_distribution 流程，依照 BFCLMetrics 的流程需求處理 _compute_score_distribution 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            scores: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 BFCLMetrics 中的 calculate_f1_score 流程，依照 BFCLMetrics 的流程需求處理 calculate_f1_score 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            precision: 評估、推理或工具執行後產生的結果與分數資料。
            recall: 評估、推理或工具執行後產生的結果與分數資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 float。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_precision_recall(
        predicted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> tuple[float, float]:
        """
        負責執行 BFCLMetrics 中的 calculate_precision_recall 流程，依照 BFCLMetrics 的流程需求處理 calculate_precision_recall 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            predicted: 此流程需要使用的輸入資料。
            expected: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 tuple[float, float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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

 