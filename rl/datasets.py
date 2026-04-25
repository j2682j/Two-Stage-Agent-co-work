"""RL訓練資料集"""

from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from trl import apply_chat_template


class GSM8KDataset:
    """GSM8K數學推理資料集

    GSM8K (Grade School Math 8K) 是一個包含8500個高品質小學數學問題的資料集。
    每個問題都需要2-8步的推理過程來解決。
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        format_type: str = "sft",  # "sft" or "rl"
        tokenizer = None  # 用於RL格式應用chat template
    ):
        """
        初始化GSM8K資料集

        Args:
            split: 資料集分割 ("train" 或 "test")
            max_samples: 最大樣本數（用於快速測試）
            format_type: 資料格式類型 ("sft" 用於監督學習, "rl" 用於強化學習)
            tokenizer: Tokenizer對象,用於RL格式應用chat template
        """
        self.split = split
        self.max_samples = max_samples
        self.format_type = format_type
        self.tokenizer = tokenizer

        print(f"📥 載入 GSM8K 資料集 (split={split})...")
        self.dataset = load_dataset("openai/gsm8k", "main", split=split)

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            print(f"   使用 {len(self.dataset)} 個樣本（限制：{max_samples}）")
        else:
            print(f"   載入了 {len(self.dataset)} 個樣本")
    
    def format_for_sft(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        格式化為SFT訓練格式
        
        Args:
            example: 原始資料樣本
            
        Returns:
            格式化後的樣本，包含 "prompt" 和 "completion"
        """
        question = example["question"]
        answer = example["answer"]
        
        # 提取最終答案（GSM8K的答案格式為：推理過程\n#### 最終答案）
        if "####" in answer:
            reasoning, final_answer = answer.split("####")
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = answer
            final_answer = ""
        
        # 構造prompt和completion
        prompt = f"Question: {question}\n\nLet's solve this step by step:\n"
        completion = f"{reasoning}\n\nFinal Answer: {final_answer}"
        
        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion  # 用於某些trainer
        }
    
    def format_for_rl(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化為RL訓練格式(Standard Format with Chat Template Applied)

        Args:
            example: 原始資料樣本

        Returns:
            格式化後的樣本，使用standard format (已應用chat template)
            - prompt: 應用chat template後的文字字串
            - ground_truth: 正確答案
            - question: 原始問題
            - full_answer: 完整答案
        """
        question = example["question"]
        answer = example["answer"]

        # 提取最終答案
        if "####" in answer:
            _, final_answer = answer.split("####")
            final_answer = final_answer.strip()
        else:
            final_answer = answer.strip()

        # 構造prompt內容
        prompt_content = f"Question: {question}\n\nLet's solve this step by step:"

        # 如果提供了tokenizer,應用chat template
        if self.tokenizer:
            messages = [{"role": "user", "content": prompt_content}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 如果沒有tokenizer,直接使用原始文字
            prompt_text = prompt_content

        return {
            "prompt": prompt_text,  # Standard format (string)
            "ground_truth": final_answer,
            "question": question,
            "full_answer": answer
        }
    
    def get_dataset(self) -> Dataset:
        """
        取得格式化後的資料集

        Returns:
            HuggingFace Dataset對象
        """
        if self.format_type == "sft":
            formatted_dataset = self.dataset.map(
                self.format_for_sft,
                remove_columns=self.dataset.column_names
            )
        elif self.format_type == "rl":
            formatted_dataset = self.dataset.map(
                self.format_for_rl,
                remove_columns=self.dataset.column_names
            )
        else:
            raise ValueError(f"不支援的格式類型: {self.format_type}")

        return formatted_dataset
    
    def __len__(self) -> int:
        """回傳資料集大小"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """取得單個樣本"""
        example = self.dataset[idx]
        if self.format_type == "sft":
            return self.format_for_sft(example)
        else:
            return self.format_for_rl(example)


def create_math_dataset(
    dataset_name: str = "gsm8k",
    split: str = "train",
    max_samples: Optional[int] = None,
    format_type: str = "sft",
    tokenizer = None
) -> Dataset:
    """
    建立數學推理資料集

    Args:
        dataset_name: 資料集名稱（目前僅支援 "gsm8k"）
        split: 資料集分割
        max_samples: 最大樣本數
        format_type: 資料格式類型
        tokenizer: Tokenizer對象,用於RL格式應用chat template

    Returns:
        格式化後的資料集
    """
    if dataset_name.lower() == "gsm8k":
        dataset_wrapper = GSM8KDataset(
            split=split,
            max_samples=max_samples,
            format_type=format_type,
            tokenizer=tokenizer
        )
        return dataset_wrapper.get_dataset()
    else:
        raise ValueError(f"不支援的資料集: {dataset_name}")


def format_math_dataset(
    dataset: Dataset,
    format_type: str = "sft",
    model_name: str = "Qwen/Qwen3-0.6B"
) -> Dataset:
    """
    將自定義資料集轉換為訓練格式

    Args:
        dataset: 原始資料集,必須包含 'question' 和 'answer' 字段
        format_type: 格式類型 ("sft" 或 "rl")
        model_name: 模型名稱,用於載入tokenizer

    Returns:
        格式化後的資料集
    """
    from transformers import AutoTokenizer

    # 載入tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 定義格式化函式
    def format_sft_sample(example: Dict[str, Any]) -> Dict[str, str]:
        """格式化為SFT格式"""
        question = example["question"]
        answer = example["answer"]

        # 提取最終答案
        if "####" in answer:
            reasoning, final_answer = answer.split("####")
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            reasoning = answer
            final_answer = ""

        # 構造prompt和completion
        prompt = f"Question: {question}\n\nLet's solve this step by step:\n"
        completion = f"{reasoning}\n\nFinal Answer: {final_answer}"

        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion
        }

    def format_rl_sample(example: Dict[str, Any]) -> Dict[str, Any]:
        """格式化為RL格式"""
        question = example["question"]
        answer = example["answer"]

        # 提取最終答案
        if "####" in answer:
            _, final_answer = answer.split("####")
            final_answer = final_answer.strip()
        else:
            final_answer = answer.strip()

        # 構造prompt內容
        prompt_content = f"Question: {question}\n\nLet's solve this step by step:"

        # 應用chat template
        messages = [{"role": "user", "content": prompt_content}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "prompt": prompt_text,
            "ground_truth": final_answer,
            "question": question,
            "full_answer": answer
        }

    # 格式化資料集
    if format_type == "sft":
        formatted_dataset = dataset.map(
            format_sft_sample,
            remove_columns=dataset.column_names
        )
    elif format_type == "rl":
        formatted_dataset = dataset.map(
            format_rl_sample,
            remove_columns=dataset.column_names
        )
    else:
        raise ValueError(f"不支援的格式類型: {format_type}")

    return formatted_dataset


def create_sft_dataset(
    max_samples: Optional[int] = 1000,
    split: str = "train"
) -> Dataset:
    """
    建立SFT訓練資料集（便捷函式）

    Args:
        max_samples: 最大樣本數
        split: 資料集分割

    Returns:
        SFT格式的資料集
    """
    return create_math_dataset(
        dataset_name="gsm8k",
        split=split,
        max_samples=max_samples,
        format_type="sft"
    )


def create_rl_dataset(
    max_samples: Optional[int] = 500,
    split: str = "train",
    model_name: str = "Qwen/Qwen3-0.6B"
) -> Dataset:
    """
    建立RL訓練資料集（便捷函式）

    Args:
        max_samples: 最大樣本數
        split: 資料集分割
        model_name: 模型名稱,用於應用chat template

    Returns:
        RL格式的資料集（已應用chat template）
    """
    # 載入tokenizer
    print(f"📝 載入tokenizer (model={model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return create_math_dataset(
        dataset_name="gsm8k",
        split=split,
        max_samples=max_samples,
        format_type="rl",
        tokenizer=tokenizer
    )


def preview_dataset(dataset: Dataset, num_samples: int = 3) -> None:
    """
    預覽資料集樣本
    
    Args:
        dataset: 資料集
        num_samples: 預覽樣本數
    """
    print(f"\n📋 資料集預覽（前 {num_samples} 個樣本）:")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n樣本 {i+1}:")
        print("-"*80)
        for key, value in sample.items():
            # 限制顯示長度
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            print(f"{key}: {value_str}")
    
    print("="*80 + "\n")


# 範例使用方式
if __name__ == "__main__":
    # 建立SFT資料集
    sft_dataset = create_sft_dataset(max_samples=10)
    preview_dataset(sft_dataset, num_samples=2)
    
    # 建立RL資料集
    rl_dataset = create_rl_dataset(max_samples=10)
    preview_dataset(rl_dataset, num_samples=2)

 