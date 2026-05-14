"""RL訓練資料集"""

from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from trl import apply_chat_template


class GSM8KDataset:
    """
    負責在 rl.datasets 中封裝 GSM8KDataset，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        split: 此流程需要使用的輸入資料。
        max_samples: 控制檢索、篩選或輸出數量的數值參數。
        format_type: 此流程需要使用的輸入資料。
        tokenizer: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        format_type: str = "sft",  # "sft" or "rl"
        tokenizer = None  # 用於RL格式應用chat template
    ):
        """
        負責執行 GSM8KDataset 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            split: 此流程需要使用的輸入資料。
            max_samples: 控制檢索、篩選或輸出數量的數值參數。
            format_type: 此流程需要使用的輸入資料。
            tokenizer: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        負責執行 GSM8KDataset 中的 format_for_sft 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            example: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        負責執行 GSM8KDataset 中的 format_for_rl 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            example: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        負責執行 GSM8KDataset 中的 get_dataset 流程，依照 GSM8KDataset 的流程需求處理 get_dataset 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dataset。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 GSM8KDataset 中的 __len__ 流程，依照 GSM8KDataset 的流程需求處理 __len__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        負責執行 GSM8KDataset 中的 __getitem__ 流程，依照 GSM8KDataset 的流程需求處理 __getitem__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            idx: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
    負責執行 rl.datasets 中的 create_math_dataset 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        dataset_name: 此流程需要使用的輸入資料。
        split: 此流程需要使用的輸入資料。
        max_samples: 控制檢索、篩選或輸出數量的數值參數。
        format_type: 此流程需要使用的輸入資料。
        tokenizer: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dataset。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    負責執行 rl.datasets 中的 format_math_dataset 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
    
    Args:
        dataset: 此流程需要使用的輸入資料。
        format_type: 此流程需要使用的輸入資料。
        model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dataset。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    from transformers import AutoTokenizer

    # 載入tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 定義格式化函式
    def format_sft_sample(example: Dict[str, Any]) -> Dict[str, str]:
        """
        負責執行 rl.datasets 中的 format_sft_sample 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            example: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 rl.datasets 中的 format_rl_sample 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            example: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    負責執行 rl.datasets 中的 create_sft_dataset 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        max_samples: 控制檢索、篩選或輸出數量的數值參數。
        split: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dataset。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    負責執行 rl.datasets 中的 create_rl_dataset 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        max_samples: 控制檢索、篩選或輸出數量的數值參數。
        split: 此流程需要使用的輸入資料。
        model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dataset。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    負責執行 rl.datasets 中的 preview_dataset 流程，依照 rl.datasets 的流程需求處理 preview_dataset 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        dataset: 此流程需要使用的輸入資料。
        num_samples: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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

 