"""RL訓練器封裝

本模組封裝了TRL的各種訓練器，提供統一的介面。
"""

from typing import Optional, Callable, Dict, Any
from pathlib import Path

from .utils import TrainingConfig, check_trl_installation, get_installation_guide

try:
    from transformers import TrainerCallback

    class DetailedLoggingCallback(TrainerCallback):
        """
        負責在 rl.trainers 中封裝 DetailedLoggingCallback，封裝此模組的狀態資料與主要操作流程。
        
        Args:
            total_steps: 此流程需要使用的輸入資料。
            num_epochs: 此流程需要使用的輸入資料。
        
        Returns:
            類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
        
        限制或副作用:
            方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
        """

        def __init__(self, total_steps: int = None, num_epochs: int = None):
            """
            負責執行 DetailedLoggingCallback 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
            
            Args:
                total_steps: 此流程需要使用的輸入資料。
                num_epochs: 此流程需要使用的輸入資料。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            self.total_steps = total_steps
            self.num_epochs = num_epochs
            self.current_epoch = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            """
            負責執行 DetailedLoggingCallback 中的 on_log 流程，依照 DetailedLoggingCallback 的流程需求處理 on_log 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                args: 此流程需要使用的輸入資料。
                state: 目前流程所需的上下文、狀態或附加資訊。
                control: 此流程需要使用的輸入資料。
                logs: 此流程需要使用的輸入資料。
                **kwargs: 此流程需要使用的輸入資料。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            if logs is None:
                return

            # 計算目前epoch
            if state.epoch is not None:
                self.current_epoch = int(state.epoch)

            # 建構日誌消息
            log_parts = []

            # Epoch和Step資訊
            if self.num_epochs:
                log_parts.append(f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

            if state.global_step and self.total_steps:
                log_parts.append(f"Step {state.global_step}/{self.total_steps}")
            elif state.global_step:
                log_parts.append(f"Step {state.global_step}")

            # Loss
            if "loss" in logs:
                log_parts.append(f"Loss: {logs['loss']:.4f}")

            # Learning Rate
            if "learning_rate" in logs:
                log_parts.append(f"LR: {logs['learning_rate']:.2e}")

            # GRPO特定指標
            if "rewards/mean" in logs:
                log_parts.append(f"Reward: {logs['rewards/mean']:.4f}")

            if "objective/kl" in logs:
                log_parts.append(f"KL: {logs['objective/kl']:.4f}")

            # 輸出日誌
            if log_parts:
                print(" | ".join(log_parts))

        def on_epoch_end(self, args, state, control, **kwargs):
            """
            負責執行 DetailedLoggingCallback 中的 on_epoch_end 流程，依照 DetailedLoggingCallback 的流程需求處理 on_epoch_end 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                args: 此流程需要使用的輸入資料。
                state: 目前流程所需的上下文、狀態或附加資訊。
                control: 此流程需要使用的輸入資料。
                **kwargs: 此流程需要使用的輸入資料。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            print(f"{'='*80}")
            print(f"✅ Epoch {self.current_epoch + 1} 完成")
            print(f"{'='*80}\n")

except ImportError:
    # 如果transformers未安裝,建立一個空的回調類
    class DetailedLoggingCallback:
        """
        負責在 rl.trainers 中封裝 DetailedLoggingCallback，封裝此模組的狀態資料與主要操作流程。
        
        Args:
            *args: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
        
        限制或副作用:
            方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
        """
        def __init__(self, *args, **kwargs):
            """
            負責執行 DetailedLoggingCallback 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
            
            Args:
                *args: 此流程需要使用的輸入資料。
                **kwargs: 此流程需要使用的輸入資料。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            pass


class BaseTrainerWrapper:
    """
    負責在 rl.trainers 中封裝 BaseTrainerWrapper，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        config: 控制此流程行為的設定資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        負責執行 BaseTrainerWrapper 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 檢查TRL是否安裝
        if not check_trl_installation():
            raise ImportError(get_installation_guide())
        
        self.config = config or TrainingConfig()
        self.trainer = None
        self.model = None
        self.tokenizer = None
    
    def setup_model(self):
        """
        負責執行 BaseTrainerWrapper 中的 setup_model 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError
    
    def train(self):
        """
        負責執行 BaseTrainerWrapper 中的 train 流程，依照 BaseTrainerWrapper 的流程需求處理 train 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        負責執行 BaseTrainerWrapper 中的 save_model 流程，將目前處理結果、設定或狀態寫入指定儲存位置。
        
        Args:
            output_dir: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        save_dir = output_dir or self.config.output_dir
        if self.trainer:
            self.trainer.save_model(save_dir)
            print(f"✅ 模型已保存到: {save_dir}")
        else:
            print("❌ 訓練器未初始化，無法保存模型")


class SFTTrainerWrapper(BaseTrainerWrapper):
    """
    負責在 rl.trainers 中封裝 SFTTrainerWrapper，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        config: 控制此流程行為的設定資料。
        dataset: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset = None
    ):
        """
        負責執行 SFTTrainerWrapper 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
            dataset: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(config)
        self.dataset = dataset
    
    def setup_model(self):
        """
        負責執行 SFTTrainerWrapper 中的 setup_model 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"📦 載入模型: {self.config.model_name}")
        
        # 載入tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 載入模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            device_map="auto" if self.config.use_fp16 or self.config.use_bf16 else None
        )
        
        print("✅ 模型載入完成")
    
    def train(self):
        """
        負責執行 SFTTrainerWrapper 中的 train 流程，依照 SFTTrainerWrapper 的流程需求處理 train 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        from trl import SFTConfig, SFTTrainer
        
        if self.model is None:
            self.setup_model()
        
        if self.dataset is None:
            raise ValueError("資料集未設定，請提供訓練資料集")
        
        # 設定訓練參數
        # 確定report_to參數
        report_to = []
        if self.config.use_wandb:
            report_to.append("wandb")
        if self.config.use_tensorboard:
            report_to.append("tensorboard")
        if not report_to:
            report_to = ["none"]

        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.use_fp16,
            bf16=self.config.use_bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            max_length=self.config.max_length,  # 修正參數名
            report_to=report_to,
        )
        
        # 計算總步數
        total_steps = (
            len(self.dataset) //
            (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps)
        ) * self.config.num_train_epochs

        # 建立詳細日誌回調
        logging_callback = DetailedLoggingCallback(
            total_steps=total_steps,
            num_epochs=self.config.num_train_epochs
        )

        # 建立訓練器
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,  # 新版TRL使用processing_class
            callbacks=[logging_callback],  # 添加回調
        )

        print("\n🚀 開始SFT訓練...")
        print(f"{'='*80}\n")
        self.trainer.train()
        print(f"\n{'='*80}")
        print("✅ SFT訓練完成")
        
        return self.trainer


class GRPOTrainerWrapper(BaseTrainerWrapper):
    """
    負責在 rl.trainers 中封裝 GRPOTrainerWrapper，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        config: 控制此流程行為的設定資料。
        dataset: 此流程需要使用的輸入資料。
        reward_fn: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset = None,
        reward_fn: Optional[Callable] = None
    ):
        """
        負責執行 GRPOTrainerWrapper 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
            dataset: 此流程需要使用的輸入資料。
            reward_fn: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(config)
        self.dataset = dataset
        self.reward_fn = reward_fn
    
    def setup_model(self):
        """
        負責執行 GRPOTrainerWrapper 中的 setup_model 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"📦 載入模型: {self.config.model_name}")
        
        # 載入tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 載入模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            device_map="auto" if self.config.use_fp16 or self.config.use_bf16 else None
        )
        
        print("✅ 模型載入完成")
    
    def train(self):
        """
        負責執行 GRPOTrainerWrapper 中的 train 流程，依照 GRPOTrainerWrapper 的流程需求處理 train 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        from trl import GRPOConfig, GRPOTrainer
        
        if self.model is None:
            self.setup_model()
        
        if self.dataset is None:
            raise ValueError("資料集未設定，請提供訓練資料集")
        
        if self.reward_fn is None:
            raise ValueError("獎勵函式未設定，請提供reward_fn")
        
        # 確定report_to參數
        report_to = []
        if self.config.use_wandb:
            report_to.append("wandb")
        if self.config.use_tensorboard:
            report_to.append("tensorboard")
        if not report_to:
            report_to = ["none"]

        # 設定訓練參數
        training_args = GRPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.use_fp16,
            bf16=self.config.use_bf16,
            report_to=report_to,
            remove_unused_columns=False,  # 保留所有列,包括ground_truth等
        )
        
        # 計算總步數
        total_steps = (
            len(self.dataset) //
            (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps)
        ) * self.config.num_train_epochs

        # 建立詳細日誌回調
        logging_callback = DetailedLoggingCallback(
            total_steps=total_steps,
            num_epochs=self.config.num_train_epochs
        )

        # 建立訓練器
        self.trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            reward_funcs=self.reward_fn,
            processing_class=self.tokenizer,
            callbacks=[logging_callback],  # 添加回調
        )

        print("\n🚀 開始GRPO訓練...")
        print(f"{'='*80}\n")
        self.trainer.train()
        print(f"\n{'='*80}")
        print("✅ GRPO訓練完成")
        
        return self.trainer


class PPOTrainerWrapper(BaseTrainerWrapper):
    """
    負責在 rl.trainers 中封裝 PPOTrainerWrapper，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        config: 控制此流程行為的設定資料。
        dataset: 此流程需要使用的輸入資料。
        reward_model: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset = None,
        reward_model = None
    ):
        """
        負責執行 PPOTrainerWrapper 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            config: 控制此流程行為的設定資料。
            dataset: 此流程需要使用的輸入資料。
            reward_model: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(config)
        self.dataset = dataset
        self.reward_model = reward_model
    
    def setup_model(self):
        """
        負責執行 PPOTrainerWrapper 中的 setup_model 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"📦 載入模型: {self.config.model_name}")
        
        # 載入tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 載入模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            device_map="auto" if self.config.use_fp16 or self.config.use_bf16 else None
        )
        
        print("✅ 模型載入完成")
    
    def train(self):
        """
        負責執行 PPOTrainerWrapper 中的 train 流程，依照 PPOTrainerWrapper 的流程需求處理 train 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        print("⚠️  PPO訓練器正在開發中...")
        print("   建議使用GRPO訓練器，它更簡單且性能相近")
        raise NotImplementedError("PPO訓練器尚未實現，請使用GRPOTrainerWrapper")
