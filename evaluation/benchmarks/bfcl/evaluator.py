"""
BFCL 評估器。

這個模組負責載入 Berkeley Function Calling Leaderboard 樣本，呼叫系統產生
function call 答案，並用 BFCL 的 AST/格式比對邏輯計算正確率。
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any, Optional, Union

from parser.bfcl_tool_call_parser import BFCLToolCallParser
from prompt.bfcl_prompt_builder import BFCLPromptBuilder

from .dataset import BFCLDataset
from .metrics import BFCLMetrics


class BFCLEvaluator:
    """
    負責執行 BFCL benchmark 評估流程，將資料集樣本轉成 prompt、呼叫 Agent 或 BFCLAdapter，
    再把模型輸出解析成 function call 並與 ground truth 比對。

    Attributes:
        dataset: BFCLDataset 實例，用來載入指定 category 的 BFCL 樣本。
        metrics: BFCLMetrics 實例，保留給後續彙整指標使用。
        evaluation_mode: 評估模式，目前主要支援 ast；execution 會先 fallback 到 AST 比對。
        category: 目前評估的 BFCL category。
        prompt_builder: 負責建立 BFCL function calling prompt。
        tool_call_parser: 負責解析模型輸出的 function call。
    """

    def __init__(
        self,
        dataset: Optional[BFCLDataset] = None,
        category: Optional[str] = None,
        evaluation_mode: str = "ast",
        local_data_dir: Optional[str] = None,
    ):
        """
        負責初始化 BFCL 評估器與其需要的 dataset、prompt builder、parser。

        Args:
            dataset: 已建立的 BFCLDataset；若為 None，會依 local_data_dir 與 category 建立。
            category: 要評估的 BFCL 類別；若為 None，交由 BFCLDataset 使用預設類別。
            evaluation_mode: 評估模式，建議使用 ast。
            local_data_dir: 本機 BFCL data 目錄。

        Returns:
            None。

        Side Effects:
            若未傳入 dataset，會建立 BFCLDataset，後續 evaluate() 會從指定資料目錄讀取資料。
        """
        self.dataset = dataset or BFCLDataset(
            bfcl_data_dir=local_data_dir or "./temp_gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
            category=category,
        )
        self.metrics = BFCLMetrics()
        self.evaluation_mode = evaluation_mode
        self.category = category
        self.prompt_builder = BFCLPromptBuilder()
        self.tool_call_parser = BFCLToolCallParser()

    def evaluate(self, agent: Any, max_samples: Optional[int] = None) -> dict[str, Any]:
        """
        負責對整個 BFCL dataset 執行逐題評估，並彙整整體與各 category 的正確率。

        Args:
            agent: 可執行 BFCL 任務的物件；建議傳入 BFCLAdapter，或至少提供 run(prompt)。
            max_samples: 最多評估幾筆樣本；若為 None，會評估全部載入樣本。

        Returns:
            包含 benchmark 名稱、agent 名稱、總題數、答對題數、整體正確率、分類指標與詳細結果的 dict。

        Side Effects:
            會呼叫 agent 執行推理，可能觸發 LLM/API/工具與 GraphMemory retrieval。
        """
        print("\n[INFO] 開始 BFCL 評估...")
        print(f"   Agent: {getattr(agent, 'name', 'Unknown')}")
        print(f"   評估模式: {self.evaluation_mode}")
        print(f"   類別: {self.category or '全部'}")

        dataset = self.dataset.load()
        if not dataset:
            print("   [WARN] 沒有載入任何 BFCL 樣本，回傳空結果。")
            return self._create_empty_results(agent)

        if max_samples:
            dataset = dataset[:max_samples]

        print(f"   評估樣本數: {len(dataset)}")

        results: list[dict[str, Any]] = []
        categories: dict[str, dict[str, Any]] = {}

        for index, sample in enumerate(dataset):
            if index % 10 == 0:
                print(f"   進度: {index + 1}/{len(dataset)}")

            try:
                sample_result = self.evaluate_sample(agent, sample)
            except Exception as exc:
                print(f"   [WARN] 樣本 {index} 評估失敗: {exc}")
                sample_result = {
                    "success": False,
                    "error": str(exc),
                    "predicted": None,
                    "expected": sample.get("ground_truth"),
                    "score": 0.0,
                    "sample_id": sample.get("id", ""),
                    "category": self.category if self.category else sample.get("category", "unknown"),
                }

            results.append(sample_result)
            category = self.category if self.category else sample.get("category", "unknown")
            if category not in categories:
                categories[category] = {"total": 0, "correct": 0, "results": []}

            categories[category]["total"] += 1
            if sample_result["success"]:
                categories[category]["correct"] += 1
            categories[category]["results"].append(sample_result)

        total_samples = len(results)
        correct_samples = sum(1 for result in results if result["success"])
        overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        category_metrics = {}
        for category, category_data in categories.items():
            accuracy = category_data["correct"] / category_data["total"] if category_data["total"] > 0 else 0.0
            category_metrics[category] = {
                "total": category_data["total"],
                "correct": category_data["correct"],
                "accuracy": accuracy,
            }

        final_results = {
            "benchmark": "BFCL",
            "agent_name": getattr(agent, "name", "Unknown"),
            "evaluation_mode": self.evaluation_mode,
            "category": self.category,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "overall_accuracy": overall_accuracy,
            "category_metrics": category_metrics,
            "detailed_results": results,
        }

        print("[OK] BFCL 評估完成")
        print(f"   整體正確率: {overall_accuracy:.2%}")
        for category, metrics in category_metrics.items():
            print(f"   {category}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

        return final_results

    def evaluate_sample(self, agent: Any, sample: dict[str, Any]) -> dict[str, Any]:
        """
        負責評估單筆 BFCL 樣本，優先使用 BFCLAdapter.run_sample() 取得完整 workflow trace。

        Args:
            agent: BFCLAdapter 或具備 run(prompt) 的 agent。
            sample: BFCL 單筆樣本，通常包含 question、function、ground_truth、id、category。

        Returns:
            包含 success、score、predicted、expected、response、parse_metadata、stage trace 與耗時的 dict。

        Side Effects:
            會呼叫 agent 執行一次任務；若使用 BFCLAdapter，會進入系統的 two-stage workflow。
        """
        question = sample.get("question", "")
        functions = sample.get("function", sample.get("functions", [])) or []
        ground_truth = sample.get("ground_truth", [])
        category = self.category if self.category else sample.get("category", "unknown")

        try:
            start_time = time.time()
            workflow = self._run_agent_on_sample(agent, sample, question, functions)
            execution_time = time.time() - start_time

            response = str(workflow.get("final_response", workflow.get("raw_response", "")) or "")
            parse_metadata = workflow.get("parse_metadata") or self.tool_call_parser.parse_with_metadata(response)
            predicted_calls = workflow.get("predicted_calls", parse_metadata.get("calls", [])) or []

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
                "question": question,
                "execution_time": execution_time,
                "sample_id": sample.get("id", ""),
                "category": category,
                "prompt": workflow.get("prompt", ""),
                "workflow": workflow,
                "parse_metadata": parse_metadata,
                "parse_source": parse_metadata.get("source"),
                "parse_error": parse_metadata.get("parse_error"),
                "expected_call_count": len(ground_truth or []),
                "predicted_call_count": len(predicted_calls),
                "stage1_result": workflow.get("stage1_result", ""),
                "stage2_outputs": workflow.get("stage2_outputs", []),
                "final_decision": workflow.get("final_decision", {}),
                "top_k_indices": workflow.get("top_k_indices", []),
            }

        except Exception as exc:
            return {
                "success": False,
                "score": 0.0,
                "predicted": None,
                "expected": ground_truth,
                "response": "",
                "question": question,
                "execution_time": 0.0,
                "error": str(exc),
                "sample_id": sample.get("id", ""),
                "category": category,
                "parse_metadata": {},
                "parse_source": None,
                "parse_error": str(exc),
                "expected_call_count": len(ground_truth or []),
                "predicted_call_count": 0,
                "stage1_result": "",
                "stage2_outputs": [],
                "final_decision": {},
                "top_k_indices": [],
            }

    def _run_agent_on_sample(
        self,
        agent: Any,
        sample: dict[str, Any],
        question: str,
        functions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        負責統一呼叫 BFCLAdapter 或舊式 agent.run(prompt)，並回傳評估器可使用的 workflow dict。

        Args:
            agent: BFCLAdapter 或具備 run(prompt) 的 agent。
            sample: 原始 BFCL 樣本。
            question: 樣本中的使用者問題。
            functions: 樣本中的 function schema 清單。

        Returns:
            包含 prompt、final_response、predicted_calls、parse_metadata 與 workflow trace 的 dict。

        Side Effects:
            會觸發 agent 推理；若 agent 是 BFCLAdapter，會保留 stage1/stage2/final decision 資訊。
        """
        if hasattr(agent, "run_sample"):
            return agent.run_sample(
                sample,
                prompt_builder=self.prompt_builder,
                parser=self.tool_call_parser,
            )

        prompt = self._build_function_calling_prompt(question, functions)
        response = str(agent.run(prompt))
        parse_metadata = self.tool_call_parser.parse_with_metadata(response)
        return {
            "prompt": prompt,
            "raw_response": response,
            "final_response": response,
            "predicted_calls": parse_metadata.get("calls", []),
            "parse_metadata": parse_metadata,
            "stage1_result": "",
            "stage2_outputs": [],
            "final_decision": {},
            "top_k_indices": [],
        }

    def _create_empty_results(self, agent: Any) -> dict[str, Any]:
        """
        負責在 dataset 沒有樣本時建立空的 BFCL 評估結果。

        Args:
            agent: 被評估的 agent 或 adapter。

        Returns:
            total_samples 與 correct_samples 皆為 0 的結果 dict。
        """
        return {
            "benchmark": "BFCL",
            "agent_name": getattr(agent, "name", "Unknown"),
            "evaluation_mode": self.evaluation_mode,
            "category": self.category,
            "total_samples": 0,
            "correct_samples": 0,
            "overall_accuracy": 0.0,
            "category_metrics": {},
            "detailed_results": [],
        }

    def _build_function_calling_prompt(self, question: str, functions: list[dict[str, Any]]) -> str:
        """
        負責用 BFCLPromptBuilder 建立單階段 function calling prompt。

        Args:
            question: BFCL 使用者問題。
            functions: 可使用的 function schema 清單。

        Returns:
            給舊式 agent.run(prompt) 使用的 prompt 字串。
        """
        return self.prompt_builder.build_function_calling_prompt(question, functions)

    def _extract_function_calls(self, response: str) -> list[dict[str, Any]]:
        """
        負責從模型輸出中解析 BFCL function call。

        Args:
            response: 模型或 agent 的文字輸出。

        Returns:
            標準化後的 function call 清單，每個元素包含 name 與 arguments。
        """
        return self.tool_call_parser.parse(response)

    def _evaluate_ast_matching(self, predicted: list[dict], expected: list) -> tuple[bool, float]:
        """
        負責依 BFCL ground truth 格式選擇 v4 dict 或字串 AST 比對。

        Args:
            predicted: 解析後的 function call 清單。
            expected: BFCL ground truth，可能是 v4 dict 格式或舊字串格式。

        Returns:
            (success, score)，success 表示是否完全正確，score 表示部分匹配比例。
        """
        if not expected:
            return len(predicted) == 0, 1.0 if len(predicted) == 0 else 0.0

        try:
            if expected and isinstance(expected[0], dict):
                return self._evaluate_bfcl_v4_format(predicted, expected)
            return self._evaluate_string_format(predicted, expected)
        except Exception as exc:
            print(f"   [WARN] 評估比對失敗: {exc}")
            return False, 0.0

    def _evaluate_bfcl_v4_format(self, predicted: list[dict], expected: list[dict]) -> tuple[bool, float]:
        """
        負責比對 BFCL v4 dict 格式的 function call ground truth。

        Args:
            predicted: Agent 預測出的 function call 清單。
            expected: BFCL v4 ground truth，例如 {"function_name": {"param": [allowed_values]}}。

        Returns:
            (success, score)，score 為成功匹配的 call 數除以 expected call 數。

        Notes:
            每個 expected call 只能被匹配一次，避免重複預測同一個 call 時被重複計分。
        """
        if len(predicted) != len(expected):
            return False, 0.0

        matches = 0
        used_expected: set[int] = set()
        for pred_call in predicted:
            if not isinstance(pred_call, dict) or "name" not in pred_call:
                continue

            pred_func_name = pred_call["name"]
            pred_args = pred_call.get("arguments", {})

            for exp_index, exp_call in enumerate(expected):
                if exp_index in used_expected or not isinstance(exp_call, dict):
                    continue

                for exp_func_name, exp_params in exp_call.items():
                    if exp_func_name != pred_func_name:
                        continue
                    if self._compare_parameters(pred_args, exp_params):
                        matches += 1
                        used_expected.add(exp_index)
                        break

                if exp_index in used_expected:
                    break

        success = matches == len(expected)
        score = matches / len(expected) if expected else 0.0
        return success, score

    def _compare_parameters(self, pred_params: dict, exp_params: dict) -> bool:
        """
        負責比對單個 function call 的 predicted arguments 是否符合 expected parameters。

        Args:
            pred_params: Agent 預測的 arguments。
            exp_params: ground truth 中允許的參數值；值可能是單值或 allowed values list。

        Returns:
            若所有 expected parameters 都被滿足，回傳 True；否則回傳 False。
        """
        if not isinstance(pred_params, dict) or not isinstance(exp_params, dict):
            return pred_params == exp_params or str(pred_params) == str(exp_params)

        for param_name, expected_values in exp_params.items():
            if param_name not in pred_params:
                if not isinstance(expected_values, list) or "" not in expected_values:
                    return False
                continue

            pred_value = pred_params[param_name]
            if isinstance(expected_values, list):
                if pred_value not in expected_values and str(pred_value) not in [str(value) for value in expected_values]:
                    return False
            elif pred_value != expected_values and str(pred_value) != str(expected_values):
                return False

        return True

    def _evaluate_string_format(self, predicted: list[dict], expected: list[str]) -> tuple[bool, float]:
        """
        負責比對舊版 BFCL 字串格式的 function call ground truth。

        Args:
            predicted: Agent 預測出的 function call 清單。
            expected: 字串格式 ground truth，例如 foo(x=1)。

        Returns:
            (success, score)，score 為成功匹配比例。
        """
        predicted_strs = []
        for call in predicted:
            if isinstance(call, dict) and "name" in call:
                func_name = call["name"]
                args = call.get("arguments", {})
                if args:
                    args_str = ", ".join([f"{key}={repr(value)}" for key, value in args.items()])
                    call_str = f"{func_name}({args_str})"
                else:
                    call_str = f"{func_name}()"
                predicted_strs.append(call_str)

        if len(predicted_strs) != len(expected):
            return False, 0.0

        matches = 0
        used_expected: set[int] = set()
        for pred_str in predicted_strs:
            for exp_index, exp_str in enumerate(expected):
                if exp_index in used_expected:
                    continue
                if self._ast_strings_match(pred_str, exp_str):
                    matches += 1
                    used_expected.add(exp_index)
                    break

        success = matches == len(expected)
        score = matches / len(expected) if expected else 0.0
        return success, score

    def _ast_strings_match(self, pred: str, expected: str) -> bool:
        """
        負責用 Python AST 比對兩個 function call 字串是否結構相同。

        Args:
            pred: Agent 預測的 function call 字串。
            expected: ground truth function call 字串。

        Returns:
            AST 結構相同時回傳 True；若 AST 解析失敗，改用去頭尾空白後的字串比對。
        """
        try:
            pred_ast = ast.parse(pred, mode="eval")
            exp_ast = ast.parse(expected, mode="eval")
            return ast.dump(pred_ast) == ast.dump(exp_ast)
        except Exception:
            return pred.strip() == expected.strip()

    def _evaluate_execution(self, predicted: list[dict], expected: list[str], functions: list[dict]) -> tuple[bool, float]:
        """
        負責執行 execution mode 的評估入口，目前先沿用 AST matching。

        Args:
            predicted: Agent 預測出的 function call 清單。
            expected: BFCL ground truth。
            functions: 可用 function schema 清單。

        Returns:
            (success, score)，目前與 _evaluate_ast_matching 相同。

        Limitations:
            尚未實作真正的 function execution sandbox，因此 execution mode 目前是相容入口。
        """
        return self._evaluate_ast_matching(predicted, expected)

    def export_to_bfcl_format(
        self,
        results: dict[str, Any],
        output_path: Union[str, Path],
        include_inference_log: bool = True,
    ) -> None:
        """
        負責把本系統的評估結果輸出成 BFCL 官方可讀的 JSONL 格式。

        Args:
            results: evaluate() 回傳的完整結果。
            output_path: 要輸出的 JSONL 檔案路徑。
            include_inference_log: 是否附上 user prompt 與 assistant response。

        Returns:
            None。結果會寫入 output_path。

        Side Effects:
            會建立 output_path 的父目錄並覆寫同名檔案。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bfcl_results = []
        for detail in results.get("detailed_results", []):
            predicted = detail.get("predicted", []) or []
            result_string = self._format_predicted_calls_for_export(predicted)

            bfcl_item = {
                "id": detail.get("sample_id", ""),
                "result": result_string,
            }

            if include_inference_log:
                bfcl_item["inference_log"] = [
                    {"role": "user", "content": detail.get("question", "")},
                    {"role": "assistant", "content": detail.get("response", "")},
                ]

            bfcl_results.append(bfcl_item)

        with open(output_path, "w", encoding="utf-8") as file:
            for item in bfcl_results:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

        print("\n[OK] BFCL result exported.")
        print(f"   輸出檔案: {output_path}")
        print(f"   樣本數: {len(bfcl_results)}")
        print(f"   包含 inference log: {include_inference_log}")

    def _format_predicted_calls_for_export(self, predicted: list[dict[str, Any]]) -> str:
        """
        負責把 parsed function calls 轉成 BFCL export 使用的字串。

        Args:
            predicted: 解析後的 function call 清單。

        Returns:
            單一或多個 function call 字串；若沒有預測 call，回傳空字串。
        """
        call_strings: list[str] = []
        for call in predicted:
            if not isinstance(call, dict) or "name" not in call:
                continue
            func_name = call["name"]
            args = call.get("arguments", {})
            if args:
                args_str = ", ".join([f"{key}={repr(value)}" for key, value in args.items()])
                call_strings.append(f"{func_name}({args_str})")
            else:
                call_strings.append(f"{func_name}()")
        return "\n".join(call_strings)
