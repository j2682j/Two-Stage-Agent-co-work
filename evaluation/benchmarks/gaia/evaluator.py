"""
GAIA 評估器模組

負責評估智慧代理在 GAIA 基準測試上的表現
"""

from typing import Dict, Any, List, Optional, Union
import time
import re
import json
from pathlib import Path

from .dataset import GAIADataset
from .metrics import GAIAMetrics


class GAIAEvaluator:
    """
    負責在 evaluation.benchmarks.gaia.evaluator 中封裝 GAIAEvaluator，封裝 benchmark 評估、答案判定、分數計算或報告資料整理流程。
    
    Args:
        dataset: 此流程需要使用的輸入資料。
        level: 此流程需要使用的輸入資料。
        local_data_dir: 此流程需要使用的輸入資料。
        strict_mode: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        dataset: Optional[GAIADataset] = None,
        level: Optional[int] = None,
        local_data_dir: Optional[str] = None,
        strict_mode: bool = True
    ):
        """
        負責執行 GAIAEvaluator 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            dataset: 此流程需要使用的輸入資料。
            level: 此流程需要使用的輸入資料。
            local_data_dir: 此流程需要使用的輸入資料。
            strict_mode: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.dataset = dataset if dataset is not None else GAIADataset(
            level=level,
            local_data_dir=local_data_dir
        )
        self.metrics = GAIAMetrics()
        self.level = level
        self.strict_mode = strict_mode
        
    def evaluate(self, agent: Any, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        負責執行 GAIAEvaluator 中的 evaluate 流程，評估 benchmark 任務的回答結果並產生分數、判定或分析資料。
        
        Args:
            agent: 此流程需要使用的輸入資料。
            max_samples: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        print(f"\n[INFO] 開始 GAIA 評估...")
        print(f"   智慧代理: {getattr(agent, 'name', 'Unknown')}")
        print(f"   難度等級: {self.level or '全部'}")
        print(f"   匹配模式: {'嚴格' if self.strict_mode else '寬松'}")

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
        level_stats = {1: {"total": 0, "correct": 0, "partial": 0},
                      2: {"total": 0, "correct": 0, "partial": 0},
                      3: {"total": 0, "correct": 0, "partial": 0}}

        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                print(f"   進度: {i+1}/{len(dataset)}")

            try:
                sample_result = self.evaluate_sample(agent, sample)
                results.append(sample_result)

                # 按等級統計
                level = sample.get("level", 1)
                if level in level_stats:
                    level_stats[level]["total"] += 1
                    if sample_result["exact_match"]:
                        level_stats[level]["correct"] += 1
                    if sample_result["partial_match"]:
                        level_stats[level]["partial"] += 1

            except Exception as e:
                print(f"   [WARN] 樣本 {i} 評估失敗: {e}")
                results.append({
                    "exact_match": False,
                    "partial_match": False,
                    "predicted": None,
                    "expected": sample.get("final_answer"),
                    "error": str(e),
                    "score": 0.0
                })

        # 計算總體指標
        total_samples = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        partial_matches = sum(1 for r in results if r["partial_match"])

        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0.0
        partial_match_rate = partial_matches / total_samples if total_samples > 0 else 0.0

        # 計算分級指標
        level_metrics = {}
        for level, stats in level_stats.items():
            if stats["total"] > 0:
                level_metrics[f"Level_{level}"] = {
                    "total": stats["total"],
                    "exact_matches": stats["correct"],
                    "partial_matches": stats["partial"],
                    "exact_match_rate": stats["correct"] / stats["total"],
                    "partial_match_rate": stats["partial"] / stats["total"]
                }

        final_results = {
            "benchmark": "GAIA",
            "agent_name": getattr(agent, 'name', 'Unknown'),
            "strict_mode": self.strict_mode,
            "level_filter": self.level,
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "level_metrics": level_metrics,
            "detailed_results": results
        }

        print(f"[OK] GAIA 評估完成")
        print(f"   精確匹配率: {exact_match_rate:.2%}")
        print(f"   部分匹配率: {partial_match_rate:.2%}")
        for level_name, metrics in level_metrics.items():
            print(f"   {level_name}: {metrics['exact_match_rate']:.2%} 精確 / {metrics['partial_match_rate']:.2%} 部分")

        return final_results
    
    def evaluate_sample(self, agent: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        負責執行 GAIAEvaluator 中的 evaluate_sample 流程，評估 benchmark 任務的回答結果並產生分數、判定或分析資料。
        
        Args:
            agent: 此流程需要使用的輸入資料。
            sample: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        response = ""
        predicted_answer = None
        try:
            # 準備輸入
            question = sample.get("question", "")
            expected_answer = sample.get("final_answer", "")
            level = sample.get("level", 1)
            task_id = sample.get("task_id", "")

            # 建構提示
            prompt = self._build_prompt(question, sample)

            # 呼叫智慧代理
            start_time = time.time()
            if hasattr(agent, "run_sample"):
                response = agent.run_sample(prompt, sample)
            else:
                response = agent.run(prompt)
            execution_time = time.time() - start_time

            # 提取答案
            predicted_answer = self._extract_answer(response)

            # 評估答案
            exact_match = self._check_exact_match(predicted_answer, expected_answer)
            partial_match = self._check_partial_match(predicted_answer, expected_answer)

            # 計算分數
            if exact_match:
                score = 1.0
            elif partial_match:
                score = 0.5
            else:
                score = 0.0

            return {
                "task_id": task_id,
                "level": level,
                "exact_match": exact_match,
                "partial_match": partial_match,
                "score": score,
                "predicted": predicted_answer,
                "expected": expected_answer,
                "response": response,
                "execution_time": execution_time
            }

        except Exception as e:
            print("   [ERROR] evaluate_sample 失敗")
            print(f"   task_id: {sample.get('task_id', '')}")
            print(f"   exception_type: {type(e).__name__}")
            print(f"   exception: {e}")
            print(f"   predicted_so_far: {predicted_answer!r}")
            print(f"   response_len: {len(response) if response else 0}")
            if response:
                preview = response[:500].replace("\r", "\\r").replace("\n", "\\n")
                print(f"   response_preview: {preview}")
            return {
                "task_id": sample.get("task_id", ""),
                "level": sample.get("level", 1),
                "exact_match": False,
                "partial_match": False,
                "score": 0.0,
                "predicted": predicted_answer,
                "expected": sample.get("final_answer", ""),
                "response": response,
                "error": str(e)
            }

    def _create_empty_results(self, agent: Any) -> Dict[str, Any]:
        """
        負責執行 GAIAEvaluator 中的 _create_empty_results 流程，依照 GAIAEvaluator 的流程需求處理 _create_empty_results 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            agent: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "benchmark": "GAIA",
            "agent_name": getattr(agent, 'name', 'Unknown'),
            "strict_mode": self.strict_mode,
            "level_filter": self.level,
            "total_samples": 0,
            "exact_matches": 0,
            "partial_matches": 0,
            "exact_match_rate": 0.0,
            "partial_match_rate": 0.0,
            "level_metrics": {},
            "detailed_results": []
        }

    def _build_prompt(self, question: str, sample: Dict[str, Any]) -> str:
        """
        負責執行 GAIAEvaluator 中的 _build_prompt 流程，依照 GAIAEvaluator 的流程需求處理 _build_prompt 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            sample: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return str(question or "").strip()

    def _extract_answer(self, response: str) -> str:
        """
        負責執行 GAIAEvaluator 中的 _extract_answer 流程，依照 GAIAEvaluator 的流程需求處理 _extract_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            response: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 首先嘗試提取GAIA官方格式的答案
        final_answer_pattern = r'FINAL ANSWER:\s*(.+?)(?:\n|$)'
        match = re.search(final_answer_pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # 移除可能的方括號
            answer = answer.strip('[]')
            return answer

        # 備用方案：查找其他答案標記
        answer_patterns = [
            r'答案[：:]\s*(.+)',
            r'最終答案[：:]\s*(.+)',
            r'Final answer[：:]\s*(.+)',
            r'Answer[：:]\s*(.+)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 如果沒有找到標記，回傳最後一個非空行
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                return line

        return response.strip()

    def _check_exact_match(self, predicted: str, expected: str) -> bool:
        """
        負責執行 GAIAEvaluator 中的 _check_exact_match 流程，依照 GAIAEvaluator 的流程需求處理 _check_exact_match 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            predicted: 此流程需要使用的輸入資料。
            expected: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not predicted or not expected:
            return False

        # 標準化字串
        pred_normalized = self._normalize_answer(predicted)
        exp_normalized = self._normalize_answer(expected)

        return pred_normalized == exp_normalized

    def _check_partial_match(self, predicted: str, expected: str) -> bool:
        """
        負責執行 GAIAEvaluator 中的 _check_partial_match 流程，依照 GAIAEvaluator 的流程需求處理 _check_partial_match 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            predicted: 此流程需要使用的輸入資料。
            expected: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not predicted or not expected:
            return False

        # 標準化字串
        pred_normalized = self._normalize_answer(predicted)
        exp_normalized = self._normalize_answer(expected)

        # 檢查包含關系
        if exp_normalized in pred_normalized or pred_normalized in exp_normalized:
            return True

        # 檢查關鍵詞匹配
        pred_words = set(pred_normalized.split())
        exp_words = set(exp_normalized.split())

        if not exp_words:
            return False

        # 如果超過70%的期望詞匯出現在預測中，認為部分匹配
        overlap = len(pred_words & exp_words)
        return overlap / len(exp_words) >= 0.7

    def _normalize_answer(self, answer: str) -> str:
        """
        負責執行 GAIAEvaluator 中的 _normalize_answer 流程，依照 GAIAEvaluator 的流程需求處理 _normalize_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            answer: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not answer:
            return ""

        answer = answer.strip()

        # 檢查是否是逗號分隔的列表
        if ',' in answer:
            # 分隔並標準化每個元素
            parts = [self._normalize_single_answer(p.strip()) for p in answer.split(',')]
            # 按字母順序排序（GAIA要求）
            parts.sort()
            return ','.join(parts)
        else:
            return self._normalize_single_answer(answer)

    def _normalize_single_answer(self, answer: str) -> str:
        """
        負責執行 GAIAEvaluator 中的 _normalize_single_answer 流程，依照 GAIAEvaluator 的流程需求處理 _normalize_single_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            answer: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        answer = answer.strip().lower()

        # 移除常見的冠詞
        articles = ['the', 'a', 'an']
        words = answer.split()
        if words and words[0] in articles:
            words = words[1:]
            answer = ' '.join(words)

        # 移除貨幣符號和百分號
        answer = answer.replace('$', '').replace('%', '').replace('€', '').replace('£', '')

        # 移除數字中的逗號分隔符（如 1,000 -> 1000）
        # 但保留小數點
        answer = re.sub(r'(\d),(\d)', r'\1\2', answer)

        # 移除多余空格
        answer = ' '.join(answer.split())

        # 移除末尾的標點符號
        answer = answer.rstrip('.,;:!?')

        return answer

    def export_to_gaia_format(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        include_reasoning: bool = True
    ) -> None:
        """
        負責執行 GAIAEvaluator 中的 export_to_gaia_format 流程，依照 GAIAEvaluator 的流程需求處理 export_to_gaia_format 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            results: 此流程需要使用的輸入資料。
            output_path: 要讀取或寫入的檔案或目錄路徑。
            include_reasoning: 控制是否啟用此項資料、功能或處理分支的布林開關。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detailed_results = results.get("detailed_results", [])

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in detailed_results:
                gaia_result = {
                    "task_id": result.get("task_id", ""),
                    "model_answer": result.get("predicted", "")
                }

                if include_reasoning:
                    gaia_result["reasoning_trace"] = result.get("response", "")

                f.write(json.dumps(gaia_result, ensure_ascii=False) + '\n')

        print(f"[OK] GAIA格式結果已匯出")
        print(f"   輸出檔案: {output_path}")
        print(f"   樣本數: {len(detailed_results)}")
        print(f"   包含推理軌跡: {include_reasoning}")

