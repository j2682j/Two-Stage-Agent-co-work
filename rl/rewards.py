"""RL訓練獎勵函式"""

import re
from typing import List, Callable, Dict, Any, Optional


class MathRewardFunction:
    """數學問題獎勵函式

    用於評估模型生成的數學答案是否正確。
    """

    def __init__(self, tolerance: float = 1e-4):
        """
        初始化獎勵函式

        Args:
            tolerance: 數值比較的容差
        """
        self.tolerance = tolerance
        self.__name__ = "MathRewardFunction"  # 添加__name__屬性
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        從文字中提取答案
        
        Args:
            text: 生成的文字
            
        Returns:
            提取的答案字串，如果找不到則回傳None
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
        標準化答案為數值
        
        Args:
            answer: 答案字串
            
        Returns:
            標準化後的數值，如果無法轉換則回傳None
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
        比較預測答案和真實答案
        
        Args:
            pred: 預測答案
            truth: 真實答案
            
        Returns:
            是否匹配
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
        計算獎勵

        Args:
            completions: 模型生成的完成文字列表
            **kwargs: 其他參數,必須包含ground_truth列表

        Returns:
            獎勵值列表（1.0表示正確，0.0表示錯誤）
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
    建立準確性獎勵函式（便捷函式）
    
    Args:
        tolerance: 數值比較的容差
        
    Returns:
        獎勵函式
    """
    reward_fn = MathRewardFunction(tolerance=tolerance)
    return reward_fn


def create_length_penalty_reward(
    base_reward_fn: Callable,
    max_length: int = 1024,
    penalty_weight: float = 0.1
) -> Callable:
    """
    建立帶長度懲罰的獎勵函式
    
    Args:
        base_reward_fn: 基礎獎勵函式
        max_length: 最大長度
        penalty_weight: 懲罰權重
        
    Returns:
        帶長度懲罰的獎勵函式
    """
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        # 計算基礎獎勵
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
    建立帶步驟獎勵的函式（鼓勵詳細的推理過程）
    
    Args:
        base_reward_fn: 基礎獎勵函式
        step_bonus: 每個推理步驟的獎勵
        
    Returns:
        帶步驟獎勵的函式
    """
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        # 計算基礎獎勵
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
    評估獎勵函式的性能
    
    Args:
        completions: 生成的完成文字列表
        ground_truths: 真實答案列表
        reward_fn: 獎勵函式
        
    Returns:
        評估結果字典
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

 