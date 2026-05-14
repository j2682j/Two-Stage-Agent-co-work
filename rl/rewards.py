"""RL訓練獎勵函式"""

import re
from typing import List, Callable, Dict, Any, Optional


class MathRewardFunction:
    """
    負責在 rl.rewards 中封裝 MathRewardFunction，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        tolerance: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, tolerance: float = 1e-4):
        """
        負責執行 MathRewardFunction 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            tolerance: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.tolerance = tolerance
        self.__name__ = "MathRewardFunction"  # 添加__name__屬性
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        負責執行 MathRewardFunction 中的 extract_answer 流程，解析模型輸出並取出答案、決策、排序或 JSON 結構。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 嘗試多種答案格式
        patterns = [
            r"Final Answer:\s*([^\n]+)",
            r"####\s*([^\n]+)",
            r"答案是?\s*[:：]?\s*([^\n]+)",
            r"Therefore,?\s*(?:the answer is)?\s*([^\n]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 如果沒有找到特定格式，嘗試提取最後一行的數字
        lines = text.strip().split('\n')
        for line in reversed(lines):
            numbers = re.findall(r'-?\d+\.?\d*', line)
            if numbers:
                return numbers[-1]
        
        return None
    
    def normalize_answer(self, answer: str) -> Optional[float]:
        """
        負責執行 MathRewardFunction 中的 normalize_answer 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
        
        Args:
            answer: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if answer is None:
            return None
        
        # 移除常見的非數字字符
        answer = answer.strip()
        answer = answer.replace(',', '')  # 移除千位分隔符
        answer = answer.replace('$', '')  # 移除美元符號
        answer = answer.replace('%', '')  # 移除百分號
        
        # 提取數字
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if not numbers:
            return None
        
        try:
            return float(numbers[0])
        except ValueError:
            return None
    
    def compare_answers(self, pred: str, truth: str) -> bool:
        """
        負責執行 MathRewardFunction 中的 compare_answers 流程，依照 MathRewardFunction 的流程需求處理 compare_answers 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            pred: 此流程需要使用的輸入資料。
            truth: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        pred_num = self.normalize_answer(pred)
        truth_num = self.normalize_answer(truth)
        
        if pred_num is None or truth_num is None:
            # 如果無法轉換為數字，進行字串比較
            return pred.strip().lower() == truth.strip().lower()
        
        # 數值比較（考慮容差）
        return abs(pred_num - truth_num) < self.tolerance
    
    def __call__(
        self,
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        負責執行 MathRewardFunction 中的 __call__ 流程，依照 MathRewardFunction 的流程需求處理 __call__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            completions: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 從kwargs中取得ground_truth
        ground_truths = kwargs.get("ground_truth", [])
        if not ground_truths:
            raise ValueError("ground_truth必須在資料集中提供")

        rewards = []

        for completion, truth in zip(completions, ground_truths):
            # 提取預測答案
            pred_answer = self.extract_answer(completion)

            # 比較答案
            if pred_answer and self.compare_answers(pred_answer, truth):
                reward = 1.0
            else:
                reward = 0.0

            rewards.append(reward)

        return rewards


def create_accuracy_reward(tolerance: float = 1e-4) -> Callable:
    """
    負責執行 rl.rewards 中的 create_accuracy_reward 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        tolerance: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Callable。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    reward_fn = MathRewardFunction(tolerance=tolerance)
    return reward_fn


def create_length_penalty_reward(
    base_reward_fn: Callable,
    max_length: int = 1024,
    penalty_weight: float = 0.1
) -> Callable:
    """
    負責執行 rl.rewards 中的 create_length_penalty_reward 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        base_reward_fn: 此流程需要使用的輸入資料。
        max_length: 控制檢索、篩選或輸出數量的數值參數。
        penalty_weight: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Callable。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        # 計算基礎獎勵
        """
        負責執行 rl.rewards 中的 reward_fn 流程，依照 rl.rewards 的流程需求處理 reward_fn 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            completions: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        base_rewards = base_reward_fn(completions, **kwargs)
        
        # 添加長度懲罰
        final_rewards = []
        for reward, completion in zip(base_rewards, completions):
            length = len(completion)
            if length > max_length:
                penalty = penalty_weight * (length - max_length) / max_length
                reward = max(0.0, reward - penalty)
            final_rewards.append(reward)
        
        return final_rewards
    
    return reward_fn


def create_step_reward(
    base_reward_fn: Callable,
    step_bonus: float = 0.1
) -> Callable:
    """
    負責執行 rl.rewards 中的 create_step_reward 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        base_reward_fn: 此流程需要使用的輸入資料。
        step_bonus: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Callable。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        # 計算基礎獎勵
        """
        負責執行 rl.rewards 中的 reward_fn 流程，依照 rl.rewards 的流程需求處理 reward_fn 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            completions: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[float]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        base_rewards = base_reward_fn(completions, **kwargs)
        
        # 添加步驟獎勵
        final_rewards = []
        for reward, completion in zip(base_rewards, completions):
            # 統計推理步驟（簡單地統計換行符數量）
            num_steps = completion.count('\n')
            step_reward = min(step_bonus * num_steps, 0.5)  # 最多0.5的額外獎勵
            final_rewards.append(reward + step_reward)
        
        return final_rewards
    
    return reward_fn


def evaluate_rewards(
    completions: List[str],
    ground_truths: List[str],
    reward_fn: Callable
) -> Dict[str, Any]:
    """
    負責執行 rl.rewards 中的 evaluate_rewards 流程，評估候選結果是否符合任務需求並回傳判定資訊。
    
    Args:
        completions: 此流程需要使用的輸入資料。
        ground_truths: 此流程需要使用的輸入資料。
        reward_fn: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    rewards = reward_fn(completions, ground_truths=ground_truths)
    
    return {
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "accuracy": sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0.0,
        "num_samples": len(rewards)
    }


# 範例使用方式
if __name__ == "__main__":
    # 建立獎勵函式
    reward_fn = create_accuracy_reward()
    
    # 測試樣例
    completions = [
        "Let's solve step by step:\n1 + 1 = 2\nFinal Answer: 2",
        "The answer is 3",
        "I don't know"
    ]
    ground_truths = ["2", "2", "2"]
    
    # 計算獎勵
    rewards = reward_fn(completions, ground_truths)
    print("Rewards:", rewards)
    
    # 評估
    results = evaluate_rewards(completions, ground_truths, reward_fn)
    print("Evaluation:", results)

 