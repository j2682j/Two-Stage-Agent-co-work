"""強化學習訓練模組（第11章：Agentic RL）

本模組提供基於TRL的強化學習訓練功能，包括：
- SFT (Supervised Fine-Tuning): 監督微調
- GRPO (Group Relative Policy Optimization): 群體相對策略優化
- PPO (Proximal Policy Optimization): 近端策略優化
- Reward Modeling: 獎勵模型訓練
"""

# 檢查TRL是否可用
try:
    import trl
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

from .trainers import SFTTrainerWrapper, GRPOTrainerWrapper, PPOTrainerWrapper
from .datasets import (
    GSM8KDataset,
    create_math_dataset,
    create_sft_dataset,
    create_rl_dataset,
    preview_dataset,
    format_math_dataset
)
from .rewards import (
    MathRewardFunction,
    create_accuracy_reward,
    create_length_penalty_reward,
    create_step_reward,
    evaluate_rewards
)
from .utils import TrainingConfig, setup_training_environment

__all__ = [
    # 可用性標志
    "TRL_AVAILABLE",

    # 訓練器
    "SFTTrainerWrapper",
    "GRPOTrainerWrapper",
    "PPOTrainerWrapper",

    # 資料集
    "GSM8KDataset",
    "create_math_dataset",
    "create_sft_dataset",
    "create_rl_dataset",
    "preview_dataset",
    "format_math_dataset",

    # 獎勵函式
    "MathRewardFunction",
    "create_accuracy_reward",
    "create_length_penalty_reward",
    "create_step_reward",
    "evaluate_rewards",

    # 工具函式
    "TrainingConfig",
    "setup_training_environment",
]

