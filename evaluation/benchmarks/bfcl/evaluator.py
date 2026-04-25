"""
BFCL 評估器模組

負責評估智慧代理在 BFCL 基準測試上的表現
"""

from typing import Dict, Any, List, Optional, Union
import json
import ast
import re
import time
from pathlib import Path

from .dataset import BFCLDataset
from .metrics import BFCLMetrics


class BFCLEvaluator:
    """BFCL 評估器

    評估智慧代理的工具呼叫能力,包括:
    - 簡單函式呼叫
    - 多函式呼叫
    - 平行函式呼叫
    - 無關檢測

    支援兩種評估模式:
    - AST評估: 抽象語法樹匹配
    - 執行評估: 實際函式執行結果對比

    Attributes:
        dataset: BFCL 資料集
        metrics: 評估指標計算器
        evaluation_mode: 評估模式 ('ast' 或 'execution')
    """

    def __init__(
        self,
        dataset: Optional[BFCLDataset] = None,
        category: Optional[str] = None,
        evaluation_mode: str = "ast",
        local_data_dir: Optional[str] = None
    ):
        """初始化 BFCL 評估器

        Args:
            dataset: BFCL 資料集,如果為 None 則自動建立
            category: 評估類別
            evaluation_mode: 評估模式 ('ast' 或 'execution')
            local_data_dir: 本地資料目錄
        """
        self.dataset = dataset or BFCLDataset(
            category=category,
            local_data_dir=local_data_dir
        )
        self.metrics = BFCLMetrics()
        self.evaluation_mode = evaluation_mode
        self.category = category
        
    def evaluate(self, agent: Any, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """評估智慧代理

        Args:
            agent: 要評估的智慧代理
            max_samples: 最大評估樣本數,None表示評估全部

        Returns:
            評估結果字典,包含各項指標
        """
        print(f"\n🔧 開始 BFCL 評估...")
        print(f"   智慧代理: {getattr(agent, 'name', 'Unknown')}")
        print(f"   評估模式: {self.evaluation_mode}")
        print(f"   類別: {self.category or '全部'}")

        # 載入資料集
        dataset = self.dataset.load()
        if not dataset:
            print("   [WARN] 資料集為空,跳過評估")
            return self._create_empty_results(agent)

        # 限制樣本數量
        if max_samples:
            dataset = dataset[:max_samples]

        print(f"   樣本數量: {len(dataset)}")

        # 執行評估
        results = []
        categories = {}

        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                print(f"   進度: {i+1}/{len(dataset)}")

            try:
                sample_result = self.evaluate_sample(agent, sample)
                results.append(sample_result)

                # 按類別統計（使用評估器的category，而不是樣本的category）
                category = self.category if self.category else sample.get("category", "unknown")
                if category not in categories:
                    categories[category] = {"total": 0, "correct": 0, "results": []}

                categories[category]["total"] += 1
                if sample_result["success"]:
                    categories[category]["correct"] += 1
                categories[category]["results"].append(sample_result)

            except Exception as e:
                print(f"   [WARN] 樣本 {i} 評估失敗: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "predicted": None,
                    "expected": sample.get("ground_truth"),
                    "score": 0.0
                })

        # 計算總體指標
        total_samples = len(results)
        correct_samples = sum(1 for r in results if r["success"])
        overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        # 計算分類指標
        category_metrics = {}
        for cat, cat_data in categories.items():
            accuracy = cat_data["correct"] / cat_data["total"] if cat_data["total"] > 0 else 0.0
            category_metrics[cat] = {
                "total": cat_data["total"],
                "correct": cat_data["correct"],
                "accuracy": accuracy
            }

        final_results = {
            "benchmark": "BFCL",
            "agent_name": getattr(agent, 'name', 'Unknown'),
            "evaluation_mode": self.evaluation_mode,
            "category": self.category,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "overall_accuracy": overall_accuracy,
            "category_metrics": category_metrics,
            "detailed_results": results
        }

        print(f"[OK] BFCL 評估完成")
        print(f"   總體正確率: {overall_accuracy:.2%}")
        for cat, metrics in category_metrics.items():
            print(f"   {cat}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

        return final_results
    
    def evaluate_sample(self, agent: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        """評估單個樣本

        Args:
            agent: 要評估的智慧代理
            sample: 樣本資料

        Returns:
            單個樣本的評估結果
        """
        try:
            # 準備輸入
            question = sample.get("question", "")
            functions = sample.get("function", [])
            ground_truth = sample.get("ground_truth", [])

            # 建構函式呼叫提示
            prompt = self._build_function_calling_prompt(question, functions)

            # 呼叫智慧代理
            start_time = time.time()
            response = agent.run(prompt)
            execution_time = time.time() - start_time

            # 解析回應中的函式呼叫
            predicted_calls = self._extract_function_calls(response)

            # 評估結果
            if self.evaluation_mode == "ast":
                success, score = self._evaluate_ast_matching(predicted_calls, ground_truth)
            else:
                success, score = self._evaluate_execution(predicted_calls, ground_truth, functions)

            return {
                "success": success,
                "score": score,
                "predicted": predicted_calls,
                "expected": ground_truth,
                "response": response,
                "question": question,  # 添加question字段用於匯出
                "execution_time": execution_time,
                "sample_id": sample.get("id", ""),
                "category": self.category if self.category else sample.get("category", "unknown")
            }

        except Exception as e:
            return {
                "success": False,
                "score": 0.0,
                "predicted": None,
                "expected": sample.get("ground_truth", []),
                "question": sample.get("question", ""),  # 添加question字段
                "error": str(e),
                "sample_id": sample.get("id", ""),
                "category": self.category if self.category else sample.get("category", "unknown")
            }

    def _create_empty_results(self, agent: Any) -> Dict[str, Any]:
        """建立空的評估結果"""
        return {
            "benchmark": "BFCL",
            "agent_name": getattr(agent, 'name', 'Unknown'),
            "evaluation_mode": self.evaluation_mode,
            "category": self.category,
            "total_samples": 0,
            "correct_samples": 0,
            "overall_accuracy": 0.0,
            "category_metrics": {},
            "detailed_results": []
        }

    def _build_function_calling_prompt(self, question: str, functions: List[Dict]) -> str:
        """建構函式呼叫提示"""
        if not functions:
            return question

        prompt = f"你是一個智慧助理，可以呼叫以下函式來幫助回答問題：\n\n"

        # 添加函式定義
        for i, func in enumerate(functions, 1):
            func_name = func.get("name", f"function_{i}")
            func_desc = func.get("description", "")
            func_params = func.get("parameters", {})

            prompt += f"函式 {i}: {func_name}\n"
            prompt += f"描述: {func_desc}\n"

            if func_params:
                prompt += f"參數: {json.dumps(func_params, ensure_ascii=False, indent=2)}\n"

            prompt += "\n"

        prompt += f"請根據以下問題，選擇合適的函式進行呼叫：\n{question}\n\n"
        prompt += "請以JSON格式回傳函式呼叫，例如：\n"
        prompt += '[{"name": "function_name", "arguments": {"param1": "value1"}}]'

        return prompt

    def _extract_function_calls(self, response: str) -> List[Dict[str, Any]]:
        """從回應中提取函式呼叫"""
        try:
            # 嘗試直接解析JSON
            if response.strip().startswith('[') and response.strip().endswith(']'):
                return json.loads(response.strip())

            # 使用正則表達式查找JSON數組
            json_pattern = r'\[.*?\]'
            matches = re.findall(json_pattern, response, re.DOTALL)

            for match in matches:
                try:
                    calls = json.loads(match)
                    if isinstance(calls, list):
                        return calls
                except json.JSONDecodeError:
                    continue

            # 查找單個函式呼叫
            single_call_pattern = r'\{.*?"name".*?\}'
            matches = re.findall(single_call_pattern, response, re.DOTALL)

            calls = []
            for match in matches:
                try:
                    call = json.loads(match)
                    if "name" in call:
                        calls.append(call)
                except json.JSONDecodeError:
                    continue

            return calls

        except Exception:
            return []

    def _evaluate_ast_matching(self, predicted: List[Dict], expected: List) -> tuple[bool, float]:
        """AST匹配評估

        支援兩種ground truth格式：
        1. BFCL v4格式：[{"func_name": {"param": [value1, value2]}}]
        2. 字串格式：["func_name(param=value)"]
        """
        if not expected:
            return len(predicted) == 0, 1.0 if len(predicted) == 0 else 0.0

        try:
            # 檢測ground truth格式
            if expected and isinstance(expected[0], dict):
                # BFCL v4格式
                return self._evaluate_bfcl_v4_format(predicted, expected)
            else:
                # 字串格式（舊版）
                return self._evaluate_string_format(predicted, expected)

        except Exception as e:
            print(f"   [WARN] 評估出錯: {e}")
            return False, 0.0

    def _evaluate_bfcl_v4_format(self, predicted: List[Dict], expected: List[Dict]) -> tuple[bool, float]:
        """評估BFCL v4格式的ground truth

        BFCL v4格式：
        predicted: [{"name": "func_name", "arguments": {"param": value}}]
        expected: [{"func_name": {"param": [value1, value2]}}]
        """
        if len(predicted) != len(expected):
            return False, 0.0

        matches = 0
        for pred_call in predicted:
            if not isinstance(pred_call, dict) or "name" not in pred_call:
                continue

            pred_func_name = pred_call["name"]
            pred_args = pred_call.get("arguments", {})

            # 在expected中查找匹配的函式呼叫
            for exp_call in expected:
                if not isinstance(exp_call, dict):
                    continue

                # expected格式：{"func_name": {"param": [values]}}
                for exp_func_name, exp_params in exp_call.items():
                    if exp_func_name != pred_func_name:
                        continue

                    # 比較參數
                    if self._compare_parameters(pred_args, exp_params):
                        matches += 1
                        break

        success = matches == len(expected)
        score = matches / len(expected) if expected else 0.0
        return success, score

    def _compare_parameters(self, pred_params: Dict, exp_params: Dict) -> bool:
        """比較預測參數和期望參數

        Args:
            pred_params: {"param": value}
            exp_params: {"param": [value1, value2]}  # 數組表示多個可接受的值
        """
        # 檢查所有必需參數
        for param_name, expected_values in exp_params.items():
            if param_name not in pred_params:
                # 參數缺失，檢查是否有空字串作為預設值
                if not isinstance(expected_values, list) or "" not in expected_values:
                    return False
                continue

            pred_value = pred_params[param_name]

            # expected_values是數組，包含所有可接受的值
            if isinstance(expected_values, list):
                # 檢查pred_value是否在可接受的值列表中
                if pred_value not in expected_values:
                    # 嘗試類型轉換后比較
                    if str(pred_value) not in [str(v) for v in expected_values]:
                        return False
            else:
                # 單個值比較
                if pred_value != expected_values and str(pred_value) != str(expected_values):
                    return False

        return True

    def _evaluate_string_format(self, predicted: List[Dict], expected: List[str]) -> tuple[bool, float]:
        """評估字串格式的ground truth（舊版）"""
        # 將預測結果轉換為字串形式
        predicted_strs = []
        for call in predicted:
            if isinstance(call, dict) and "name" in call:
                func_name = call["name"]
                args = call.get("arguments", {})
                # 建構函式呼叫字串
                if args:
                    args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
                    call_str = f"{func_name}({args_str})"
                else:
                    call_str = f"{func_name}()"
                predicted_strs.append(call_str)

        # 簡單的字串匹配評估
        if len(predicted_strs) != len(expected):
            return False, 0.0

        # 檢查每個函式呼叫是否匹配
        matches = 0
        for pred_str in predicted_strs:
            for exp_str in expected:
                if self._ast_strings_match(pred_str, exp_str):
                    matches += 1
                    break

        success = matches == len(expected)
        score = matches / len(expected) if expected else 0.0

        return success, score

    def _ast_strings_match(self, pred: str, expected: str) -> bool:
        """比較兩個函式呼叫字串是否在AST層面匹配"""
        try:
            # 嘗試解析為AST並比較
            pred_ast = ast.parse(pred, mode='eval')
            exp_ast = ast.parse(expected, mode='eval')
            return ast.dump(pred_ast) == ast.dump(exp_ast)
        except:
            # 如果AST解析失敗，使用字串相似度
            return pred.strip() == expected.strip()

    def _evaluate_execution(self, predicted: List[Dict], expected: List[str], functions: List[Dict]) -> tuple[bool, float]:
        """執行評估（簡化版本）"""
        # 這裡實現簡化的執行評估
        # 在實際應用中，需要安全的代碼執行環境
        return self._evaluate_ast_matching(predicted, expected)

    def export_to_bfcl_format(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        include_inference_log: bool = True
    ) -> None:
        """匯出評估結果為BFCL官方格式

        BFCL官方格式範例：
        {
            "id": "simple_python_0",
            "model_result": [
                {
                    "name": "calculate_triangle_area",
                    "arguments": {"base": 10, "height": 5, "unit": "units"}
                }
            ],
            "inference_log": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

        Args:
            results: evaluate()方法回傳的評估結果
            output_path: 輸出檔案路徑
            include_inference_log: 是否包含推理日誌
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 轉換為BFCL格式
        bfcl_results = []

        for detail in results.get("detailed_results", []):
            # 將predicted轉換為字串格式的函式呼叫
            predicted = detail.get("predicted", [])
            result_string = ""

            if predicted:
                call = predicted[0]  # 通常只有一個函式呼叫
                if isinstance(call, dict) and "name" in call:
                    func_name = call["name"]
                    args = call.get("arguments", {})

                    # 建構函式呼叫字串
                    if args:
                        args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
                        result_string = f"{func_name}({args_str})"
                    else:
                        result_string = f"{func_name}()"

            bfcl_item = {
                "id": detail.get("sample_id", ""),
                "result": result_string  # BFCL期望的是單個字串
            }

            # 添加推理日誌（如果需要）
            if include_inference_log:
                question = detail.get("question", "")
                response = detail.get("response", "")

                bfcl_item["inference_log"] = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]

            bfcl_results.append(bfcl_item)

        # 寫入JSONL格式（每行一個JSON對象）
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in bfcl_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n[OK] BFCL格式結果已匯出")
        print(f"   輸出檔案: {output_path}")
        print(f"   樣本數: {len(bfcl_results)}")
        print(f"   包含推理日誌: {include_inference_log}")

        # 提示如何使用BFCL官方評估
        print(f"\n📝 使用BFCL官方評估工具：")
        print(f"   1. 安裝: pip install bfcl-eval")
        print(f"   2. 設定環境變數: export BFCL_PROJECT_ROOT=.")
        print(f"   3. 將結果檔案復制到: result/HelloAgents/")
        print(f"   4. 執行評估: bfcl evaluate --model HelloAgents --test-category {self.category}")

