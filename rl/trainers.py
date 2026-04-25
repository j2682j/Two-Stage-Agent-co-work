"""RL訓練器封裝

本模組封裝了TRL的各種訓練器，提供統一的介面。
"""

from typing import Optional, Callable, Dict, Any
from pathlib import Path

from .utils import TrainingConfig, check_trl_installation, get_installation_guide

try:
    from transformers import TrainerCallback

    class DetailedLoggingCallback(TrainerCallback):
        """詳細日誌回調

        在訓練過程中輸出更詳細的日誌資訊,包括:
        - Epoch/Step進度
        - Loss
        - Learning Rate
        - Reward (GRPO)
        - KL散度 (GRPO)
        """

        def __init__(self, total_steps: int = None, num_epochs: int = None):
            """
            初始化回調

            Args:
                total_steps: 總步數
                num_epochs: 總輪數
            """
            self.total_steps = total_steps
            self.num_epochs = num_epochs
            self.current_epoch = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            """日誌回調"""
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
            """Epoch結束回調"""
            print(f"{'='*80}")
            print(f"✅ Epoch {self.current_epoch + 1} 完成")
            print(f"{'='*80}\n")

except ImportError:
    # 如果transformers未安裝,建立一個空的回調類
    class DetailedLoggingCallback:
        def __init__(self, *args, **kwargs):
            pass


class BaseTrainerWrapper:
    """訓練器基類"""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        初始化訓練器
        
        Args:
            config: 訓練設定
        """
        # 檢查TRL是否安裝
        if not check_trl_installation():
            raise ImportError(get_installation_guide())
        
        self.config = config or TrainingConfig()
        self.trainer = None
        self.model = None
        self.tokenizer = None
    
    def setup_model(self):
        """設定模型和tokenizer"""
        raise NotImplementedError
    
    def train(self):
        """開始訓練"""
        raise NotImplementedError
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        保存模型
        
        Args:
            output_dir: 輸出目錄
        """
        save_dir = output_dir or self.config.output_dir
        if self.trainer:
            self.trainer.save_model(save_dir)
            print(f"✅ 模型已保存到: {save_dir}")
        else:
            print("❌ 訓練器未初始化，無法保存模型")


class SFTTrainerWrapper(BaseTrainerWrapper):
    """SFT (Supervised Fine-Tuning) 訓練器封裝
    
    用於監督微調，讓模型學會遵循指令和基本的推理格式。
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset = None
    ):
        """
        初始化SFT訓練器
        
        Args:
            config: 訓練設定
            dataset: 訓練資料集
        """
        super().__init__(config)
        self.dataset = dataset
    
    def setup_model(self):
        """設定模型和tokenizer"""
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
        """開始SFT訓練"""
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
    """GRPO (Group Relative Policy Optimization) 訓練器封裝
    
    用於強化學習訓練，優化模型的推理能力。
    GRPO相比PPO更簡單，不需要Value Model。
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset = None,
        reward_fn: Optional[Callable] = None
    ):
        """
        初始化GRPO訓練器
        
        Args:
            config: 訓練設定
            dataset: 訓練資料集
            reward_fn: 獎勵函式
        """
        super().__init__(config)
        self.dataset = dataset
        self.reward_fn = reward_fn
    
    def setup_model(self):
        """設定模型和tokenizer"""
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
        """開始GRPO訓練"""
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
    """PPO (Proximal Policy Optimization) 訓練器封裝
    
    用於強化學習訓練，是經典的RL算法。
    相比GRPO，PPO需要額外的Value Model，但可能獲得更好的性能。
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        dataset = None,
        reward_model = None
    ):
        """
        初始化PPO訓練器
        
        Args:
            config: 訓練設定
            dataset: 訓練資料集
            reward_model: 獎勵模型
        """
        super().__init__(config)
        self.dataset = dataset
        self.reward_model = reward_model
    
    def setup_model(self):
        """設定模型和tokenizer"""
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
        """開始PPO訓練"""
        print("⚠️  PPO訓練器正在開發中...")
        print("   建議使用GRPO訓練器，它更簡單且性能相近")
        raise NotImplementedError("PPO訓練器尚未實現，請使用GRPOTrainerWrapper")
