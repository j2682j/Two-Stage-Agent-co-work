"""RL訓練工具函式"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    負責在 rl.utils 中封裝 TrainingConfig，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    # 模型設定
    model_name: str = "Qwen/Qwen3-0.6B"
    model_revision: Optional[str] = None
    
    # 訓練設定
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # RL特定設定
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 硬件設定
    use_fp16: bool = True
    use_bf16: bool = False
    gradient_checkpointing: bool = True
    
    # LoRA設定
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # 監控設定
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    use_tensorboard: bool = True
    
    # 其他設定
    seed: int = 42
    max_length: int = 2048
    
    def to_dict(self) -> Dict[str, Any]:
        """
        負責執行 TrainingConfig 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


def setup_training_environment(config: TrainingConfig) -> None:
    """
    負責執行 rl.utils 中的 setup_training_environment 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
    
    Args:
        config: 控制此流程行為的設定資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    # 建立輸出目錄
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 設定隨機種子
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    except ImportError:
        pass
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # 設定環境變數
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 設定wandb設定
    if config.use_wandb:
        if config.wandb_project:
            os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "false"  # 不上傳模型檔案

    print(f"✅ 訓練環境設定完成")
    print(f"   - 輸出目錄: {config.output_dir}")
    print(f"   - 隨機種子: {config.seed}")
    print(f"   - 模型: {config.model_name}")


def check_trl_installation() -> bool:
    """
    負責執行 rl.utils 中的 check_trl_installation 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    try:
        import trl
        return True
    except ImportError:
        return False


def get_installation_guide() -> str:
    """
    負責執行 rl.utils 中的 get_installation_guide 流程，依照 rl.utils 的流程需求處理 get_installation_guide 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return """
TRL (Transformer Reinforcement Learning) 未安裝。

請使用以下命令安裝：

方式1：安裝HelloAgents的RL功能（推薦）
    pip install hello-agents[rl]

方式2：單獨安裝TRL
    pip install trl

方式3：從源碼安裝最新版本
    pip install git+https://github.com/huggingface/trl.git

安裝完成后，您可以使用以下功能：
- SFT訓練（監督微調）
- GRPO訓練（群體相對策略優化）
- PPO訓練（近端策略優化）
- DPO訓練（直接偏好優化）
- Reward Model訓練

更多資訊請訪問：https://huggingface.co/docs/trl
"""


def format_training_time(seconds: float) -> str:
    """
    負責執行 rl.utils 中的 format_training_time 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
    
    Args:
        seconds: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device_info() -> Dict[str, Any]:
    """
    負責執行 rl.utils 中的 get_device_info 流程，依照 rl.utils 的流程需求處理 get_device_info 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
    }
    
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    return info


def print_training_summary(
    algorithm: str,
    model_name: str,
    dataset_name: str,
    num_epochs: int,
    output_dir: str
) -> None:
    """
    負責執行 rl.utils 中的 print_training_summary 流程，依照 rl.utils 的流程需求處理 print_training_summary 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        algorithm: 此流程需要使用的輸入資料。
        model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        dataset_name: 此流程需要使用的輸入資料。
        num_epochs: 此流程需要使用的輸入資料。
        output_dir: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    device_info = get_device_info()
    
    print("\n" + "="*60)
    print(f"🚀 開始 {algorithm} 訓練")
    print("="*60)
    print(f"📦 模型: {model_name}")
    print(f"📊 資料集: {dataset_name}")
    print(f"🔄 訓練輪數: {num_epochs}")
    print(f"💾 輸出目錄: {output_dir}")
    print(f"🖥️  裝置: {'GPU' if device_info['cuda_available'] else 'CPU'}")
    if device_info['cuda_available']:
        print(f"   - GPU數量: {device_info['cuda_device_count']}")
        print(f"   - GPU型號: {device_info['cuda_device_name']}")
    print("="*60 + "\n")

 