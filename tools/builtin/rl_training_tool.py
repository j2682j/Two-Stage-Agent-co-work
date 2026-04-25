"""RL訓練工具

提供強化學習訓練功能，包括SFT、GRPO、PPO等算法。
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from ..base import Tool, ToolParameter


class RLTrainingTool(Tool):
    """RL訓練工具

    支援的訓練算法：
    - SFT: Supervised Fine-Tuning (監督微調)
    - GRPO: Group Relative Policy Optimization (群體相對策略優化)

    支援的功能：
    - 訓練模型 (train)
    - 載入資料集 (load_dataset)
    - 建立獎勵函式 (create_reward)
    - 評估模型 (evaluate)
    """

    def __init__(self):
        super().__init__(
            name="rl_training",
            description=(
                "強化學習訓練工具。支援SFT、GRPO等算法，"
                "用於訓練和優化語言模型的推理能力。"
                "也支援資料集載入、獎勵函式建立、模型評估等功能。"
                "支援自定義資料集和獎勵函式。"
            )
        )

        # 檢查TRL是否可用
        try:
            from hello_agents.rl import TRL_AVAILABLE
            self.trl_available = TRL_AVAILABLE
        except ImportError:
            self.trl_available = False

        # 儲存自定義資料集和獎勵函式
        self.custom_datasets = {}
        self.custom_reward_functions = {}

    def register_dataset(self, name: str, dataset) -> None:
        """
        註冊自定義資料集

        Args:
            name: 資料集名稱
            dataset: 資料集對象(HuggingFace Dataset)
        """
        self.custom_datasets[name] = dataset
        print(f"[OK] 已註冊自定義資料集: {name}")

    def register_reward_function(self, name: str, reward_fn) -> None:
        """
        註冊自定義獎勵函式

        Args:
            name: 獎勵函式名稱
            reward_fn: 獎勵函式(接受completions和kwargs,回傳rewards列表)
        """
        self.custom_reward_functions[name] = reward_fn
        print(f"[OK] 已註冊自定義獎勵函式: {name}")

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        執行RL相關操作

        Args:
            parameters: 操作參數，包括：
                - action: 操作類型 ("train", "load_dataset", "create_reward", "evaluate")

                訓練參數 (action="train"):
                - algorithm: 訓練算法 ("sft", "grpo")
                - model_name: 模型名稱（預設: "Qwen/Qwen2-0.5B-Instruct"）
                - dataset: 資料集名稱（預設: "gsm8k"）
                - max_samples: 最大樣本數（用於快速測試）
                - num_epochs: 訓練輪數（預設: 3）
                - output_dir: 輸出目錄（預設: "./output"）
                - use_lora: 是否使用LoRA（預設: True）
                - batch_size: 批次大小（預設: 4）

                資料集載入參數 (action="load_dataset"):
                - format: 資料格式 ("sft", "rl")
                - split: 資料集劃分 ("train", "test")
                - max_samples: 最大樣本數

                獎勵函式參數 (action="create_reward"):
                - reward_type: 獎勵類型 ("accuracy", "length_penalty", "step")
                - penalty_weight: 長度懲罰權重（僅length_penalty）
                - step_bonus: 步驟獎勵（僅step）

        Returns:
            操作結果的JSON字串
        """
        # 檢查TRL是否可用
        if not self.trl_available:
            return json.dumps({
                "status": "error",
                "message": (
                    "TRL未安裝。請使用以下命令安裝：\n"
                    "pip install hello-agents[rl]\n"
                    "或\n"
                    "pip install trl"
                )
            }, ensure_ascii=False, indent=2)

        # 取得操作類型
        action = parameters.get("action", "train").lower()

        try:
            if action == "train":
                return self._handle_train(parameters)
            elif action == "load_dataset":
                return self._handle_load_dataset(parameters)
            elif action == "create_reward":
                return self._handle_create_reward(parameters)
            elif action == "evaluate":
                return self._handle_evaluate(parameters)
            else:
                result = {
                    "status": "error",
                    "message": f"不支援的操作: {action}。支援的操作: train, load_dataset, create_reward, evaluate"
                }
                return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            import traceback
            error_result = {
                "status": "error",
                "message": f"操作失敗: {str(e)}",
                "traceback": traceback.format_exc()
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

    def _handle_train(self, parameters: Dict[str, Any]) -> str:
        """處理訓練操作"""
        algorithm = parameters.get("algorithm", "sft").lower()
        model_name = parameters.get("model_name", "Qwen/Qwen2-0.5B-Instruct")
        dataset_name = parameters.get("dataset", "gsm8k")
        max_samples = parameters.get("max_samples", None)
        num_epochs = parameters.get("num_epochs", 3)
        output_dir = parameters.get("output_dir", "./output")
        use_lora = parameters.get("use_lora", True)
        batch_size = parameters.get("batch_size", 4)

        # 支援自定義資料集
        custom_dataset = parameters.get("custom_dataset", None)
        # 支援自定義獎勵函式
        custom_reward = parameters.get("custom_reward", None)

        # 支援訓練監控設定
        use_wandb = parameters.get("use_wandb", False)
        use_tensorboard = parameters.get("use_tensorboard", True)
        wandb_project = parameters.get("wandb_project", None)

        print(f"\n{'='*60}")
        print(f"[INFO] 開始 {algorithm.upper()} 訓練")
        print(f"{'='*60}")
        print(f" 模型: {model_name}")
        if custom_dataset:
            print(f"[INFO] 資料集: 自定義資料集")
        else:
            print(f"[INFO] 資料集: {dataset_name}")
        print(f" 訓練輪數: {num_epochs}")
        print(f" 輸出目錄: {output_dir}")
        print(f"[INFO] 算法: {algorithm.upper()}")
        if custom_reward:
            print(f" 獎勵函式: 自定義獎勵函式")

        # 打印監控設定
        monitoring = []
        if use_wandb:
            monitoring.append(f"wandb (項目: {wandb_project or 'default'})")
        if use_tensorboard:
            monitoring.append("tensorboard")
        if monitoring:
            print(f"[INFO] 訓練監控: {', '.join(monitoring)}")

        print(f"{'='*60}\n")

        if algorithm == "sft":
            result = self._train_sft(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                num_epochs=num_epochs,
                output_dir=output_dir,
                use_lora=use_lora,
                batch_size=batch_size,
                custom_dataset=custom_dataset,
                use_wandb=use_wandb,
                use_tensorboard=use_tensorboard,
                wandb_project=wandb_project
            )
        elif algorithm == "grpo":
            result = self._train_grpo(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                num_epochs=num_epochs,
                output_dir=output_dir,
                use_lora=use_lora,
                batch_size=batch_size,
                custom_dataset=custom_dataset,
                custom_reward=custom_reward,
                use_wandb=use_wandb,
                use_tensorboard=use_tensorboard,
                wandb_project=wandb_project
            )
        else:
            result = {
                "status": "error",
                "message": f"不支援的算法: {algorithm}。支援的算法: sft, grpo"
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_load_dataset(self, parameters: Dict[str, Any]) -> str:
        """處理資料集載入操作"""
        from hello_agents.rl import create_sft_dataset, create_rl_dataset

        format_type = parameters.get("format", "sft").lower()
        split = parameters.get("split", "train")
        max_samples = parameters.get("max_samples", 100)
        model_name = parameters.get("model_name", "Qwen/Qwen3-0.6B")

        if format_type == "sft":
            dataset = create_sft_dataset(split=split, max_samples=max_samples)
        elif format_type == "rl":
            dataset = create_rl_dataset(split=split, max_samples=max_samples, model_name=model_name)
        else:
            return json.dumps({
                "status": "error",
                "message": f"不支援的資料格式: {format_type}。支援的格式: sft, rl"
            }, ensure_ascii=False, indent=2)

        result = {
            "status": "success",
            "format": format_type,
            "split": split,
            "dataset_size": len(dataset),
            "sample_keys": list(dataset[0].keys()) if len(dataset) > 0 else []
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_create_reward(self, parameters: Dict[str, Any]) -> str:
        """處理獎勵函式建立操作"""
        from hello_agents.rl import (
            create_accuracy_reward,
            create_length_penalty_reward,
            create_step_reward
        )

        reward_type = parameters.get("reward_type", "accuracy").lower()

        if reward_type == "accuracy":
            reward_fn = create_accuracy_reward()
            result = {
                "status": "success",
                "reward_type": "accuracy",
                "description": "準確性獎勵函式: 答案正確=1.0, 錯誤=0.0"
            }
        elif reward_type == "length_penalty":
            penalty_weight = parameters.get("penalty_weight", 0.001)
            max_length = parameters.get("max_length", 1024)
            # 建立基礎獎勵函式
            base_reward_fn = create_accuracy_reward()
            reward_fn = create_length_penalty_reward(
                base_reward_fn=base_reward_fn,
                max_length=max_length,
                penalty_weight=penalty_weight
            )
            result = {
                "status": "success",
                "reward_type": "length_penalty",
                "penalty_weight": penalty_weight,
                "max_length": max_length,
                "description": f"長度懲罰獎勵函式: 基礎獎勵 - {penalty_weight} * (長度 / {max_length})"
            }
        elif reward_type == "step":
            step_bonus = parameters.get("step_bonus", 0.1)
            # 建立基礎獎勵函式
            base_reward_fn = create_accuracy_reward()
            reward_fn = create_step_reward(
                base_reward_fn=base_reward_fn,
                step_bonus=step_bonus
            )
            result = {
                "status": "success",
                "reward_type": "step",
                "step_bonus": step_bonus,
                "description": f"步驟獎勵函式: 基礎獎勵 + {step_bonus} * 步驟數"
            }
        else:
            return json.dumps({
                "status": "error",
                "message": f"不支援的獎勵類型: {reward_type}。支援的類型: accuracy, length_penalty, step"
            }, ensure_ascii=False, indent=2)

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_evaluate(self, parameters: Dict[str, Any]) -> str:
        """處理模型評估操作"""
        try:
            from hello_agents.rl import (
                create_rl_dataset,
                create_accuracy_reward,
                evaluate_rewards
            )
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_path = parameters.get("model_path")
            max_samples = parameters.get("max_samples", 100)

            if not model_path:
                return json.dumps({
                    "status": "error",
                    "message": "缺少必需參數: model_path"
                }, ensure_ascii=False, indent=2)

            # 載入測試資料
            print(f"📥 載入測試資料集 (max_samples={max_samples})...")
            dataset = create_rl_dataset(split="test", max_samples=max_samples, model_name=model_path)

            # 載入模型和tokenizer
            print(f"📥 載入模型: {model_path}...")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                model.eval()
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"模型載入失敗: {str(e)}"
                }, ensure_ascii=False, indent=2)

            # 生成預測
            print(" 生成預測...")
            completions = []
            ground_truths = []

            # 匯入tqdm用於進度條
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
                print("  提示: 安裝tqdm可顯示進度條 (pip install tqdm)")

            # 建立迭代器
            iterator = range(min(max_samples, len(dataset)))
            if use_tqdm:
                iterator = tqdm(iterator, desc="  評估進度", unit="樣本")

            for i in iterator:
                prompt = dataset[i]["prompt"]
                ground_truth = dataset[i]["ground_truth"]

                # 生成回答
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,  # 減少生成長度加快速度
                        temperature=0.7,
                        do_sample=False,  # 使用貪婪解碼加快速度
                        pad_token_id=tokenizer.pad_token_id
                    )
                # 只取生成的部分,不包括輸入
                completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                completions.append(completion)
                ground_truths.append(ground_truth)

                # 如果沒有tqdm,每10個樣本打印一次進度
                if not use_tqdm and (i + 1) % 10 == 0:
                    print(f"  進度: {i+1}/{max_samples}")

            # 計算獎勵
            print("[INFO] 計算評估指標...")
            reward_fn = create_accuracy_reward()
            rewards = reward_fn(completions, ground_truth=ground_truths)

            # 計算統計資訊
            avg_reward = sum(rewards) / len(rewards)
            accuracy = avg_reward  # 對於準確性獎勵,平均獎勵就是正確率

            result = {
                "status": "success",
                "model_path": model_path,
                "num_samples": len(completions),
                "accuracy": f"{accuracy:.2%}",
                "average_reward": f"{avg_reward:.4f}",
                "device": device
            }

            print(f"\n[OK] 評估完成!")
            print(f"  正確率: {accuracy:.2%}")
            print(f"  平均獎勵: {avg_reward:.4f}")

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"評估失敗: {str(e)}"
            }, ensure_ascii=False, indent=2)
    
    def _train_sft(
        self,
        model_name: str,
        dataset_name: str,
        max_samples: Optional[int],
        num_epochs: int,
        output_dir: str,
        use_lora: bool,
        batch_size: int,
        custom_dataset = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None
    ) -> Dict[str, Any]:
        """執行SFT訓練"""
        from hello_agents.rl import (
            SFTTrainerWrapper,
            TrainingConfig,
            create_sft_dataset,
            setup_training_environment
        )

        # 建立設定
        config = TrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            use_lora=use_lora,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            wandb_project=wandb_project
        )

        # 設定環境
        setup_training_environment(config)

        # 載入資料集
        if custom_dataset is not None:
            # 使用自定義資料集
            dataset = custom_dataset
            print(f"[OK] 使用自定義資料集: {len(dataset)} 個樣本")
        elif dataset_name in self.custom_datasets:
            # 使用註冊的自定義資料集
            dataset = self.custom_datasets[dataset_name]
            print(f"[OK] 使用註冊的資料集 '{dataset_name}': {len(dataset)} 個樣本")
        else:
            # 使用預設資料集
            dataset = create_sft_dataset(max_samples=max_samples)

        # 建立訓練器
        trainer_wrapper = SFTTrainerWrapper(config=config, dataset=dataset)

        # 開始訓練
        trainer_wrapper.train()

        # 保存模型
        trainer_wrapper.save_model()

        return {
            "status": "success",
            "algorithm": "SFT",
            "model": model_name,
            "output_dir": output_dir,
            "num_epochs": num_epochs,
            "dataset_size": len(dataset)
        }
    
    def _train_grpo(
        self,
        model_name: str,
        dataset_name: str,
        max_samples: Optional[int],
        num_epochs: int,
        output_dir: str,
        use_lora: bool,
        batch_size: int,
        custom_dataset = None,
        custom_reward = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None
    ) -> Dict[str, Any]:
        """執行GRPO訓練"""
        from hello_agents.rl import (
            GRPOTrainerWrapper,
            TrainingConfig,
            create_rl_dataset,
            create_accuracy_reward,
            setup_training_environment
        )

        # 建立設定
        config = TrainingConfig(
            model_name=model_name,
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            use_lora=use_lora,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            wandb_project=wandb_project
        )

        # 設定環境
        setup_training_environment(config)

        # 載入資料集
        if custom_dataset is not None:
            # 使用自定義資料集
            dataset = custom_dataset
            print(f"[OK] 使用自定義資料集: {len(dataset)} 個樣本")
        elif dataset_name in self.custom_datasets:
            # 使用註冊的自定義資料集
            dataset = self.custom_datasets[dataset_name]
            print(f"[OK] 使用註冊的資料集 '{dataset_name}': {len(dataset)} 個樣本")
        else:
            # 使用預設資料集
            dataset = create_rl_dataset(max_samples=max_samples, model_name=model_name)

        # 建立獎勵函式
        if custom_reward is not None:
            # 使用自定義獎勵函式
            reward_fn = custom_reward
            print(f"[OK] 使用自定義獎勵函式")
        elif dataset_name in self.custom_reward_functions:
            # 使用註冊的獎勵函式
            reward_fn = self.custom_reward_functions[dataset_name]
            print(f"[OK] 使用註冊的獎勵函式 '{dataset_name}'")
        else:
            # 使用預設獎勵函式
            reward_fn = create_accuracy_reward()

        # 建立訓練器
        trainer_wrapper = GRPOTrainerWrapper(
            config=config,
            dataset=dataset,
            reward_fn=reward_fn
        )

        # 開始訓練
        trainer_wrapper.train()

        # 保存模型
        trainer_wrapper.save_model()

        return {
            "status": "success",
            "algorithm": "GRPO",
            "model": model_name,
            "output_dir": output_dir,
            "num_epochs": num_epochs,
            "dataset_size": len(dataset)
        }
    
    def get_parameters(self) -> List[ToolParameter]:
        """取得工具參數定義"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作類型: train (訓練), load_dataset (載入資料集), create_reward (建立獎勵函式), evaluate (評估模型)",
                required=False,
                default="train"
            ),
            ToolParameter(
                name="algorithm",
                type="string",
                description="訓練算法 (僅train): sft (監督微調), grpo (群體相對策略優化)",
                required=False,
                default="sft"
            ),
            ToolParameter(
                name="model_name",
                type="string",
                description="模型名稱 (僅train)，例如: Qwen/Qwen2-0.5B-Instruct",
                required=False,
                default="Qwen/Qwen2-0.5B-Instruct"
            ),
            ToolParameter(
                name="dataset",
                type="string",
                description="資料集名稱 (僅train)，目前支援: gsm8k",
                required=False,
                default="gsm8k"
            ),
            ToolParameter(
                name="format",
                type="string",
                description="資料格式 (僅load_dataset): sft, rl",
                required=False,
                default="sft"
            ),
            ToolParameter(
                name="split",
                type="string",
                description="資料集劃分 (僅load_dataset): train, test",
                required=False,
                default="train"
            ),
            ToolParameter(
                name="reward_type",
                type="string",
                description="獎勵類型 (僅create_reward): accuracy, length_penalty, step",
                required=False,
                default="accuracy"
            ),
            ToolParameter(
                name="max_samples",
                type="integer",
                description="最大樣本數（用於快速測試），None表示使用全部資料",
                required=False,
                default=None
            ),
            ToolParameter(
                name="num_epochs",
                type="integer",
                description="訓練輪數 (僅train)",
                required=False,
                default=3
            ),
            ToolParameter(
                name="output_dir",
                type="string",
                description="輸出目錄 (僅train)",
                required=False,
                default="./output"
            ),
            ToolParameter(
                name="use_lora",
                type="boolean",
                description="是否使用LoRA進行參數高效微調 (僅train)",
                required=False,
                default=True
            ),
            ToolParameter(
                name="batch_size",
                type="integer",
                description="批次大小 (僅train)",
                required=False,
                default=4
            ),
        ]


# 便捷函式
def train_with_sft(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    max_samples: Optional[int] = 100,
    num_epochs: int = 3,
    output_dir: str = "./output/sft"
) -> str:
    """
    使用SFT訓練模型（便捷函式）

    Args:
        model_name: 模型名稱
        max_samples: 最大樣本數
        num_epochs: 訓練輪數
        output_dir: 輸出目錄

    Returns:
        訓練結果JSON字串
    """
    tool = RLTrainingTool()
    return tool.run({
        "action": "train",
        "algorithm": "sft",
        "model_name": model_name,
        "max_samples": max_samples,
        "num_epochs": num_epochs,
        "output_dir": output_dir
    })


def train_with_grpo(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    max_samples: Optional[int] = 100,
    num_epochs: int = 3,
    output_dir: str = "./output/grpo"
) -> str:
    """
    使用GRPO訓練模型（便捷函式）

    Args:
        model_name: 模型名稱
        max_samples: 最大樣本數
        num_epochs: 訓練輪數
        output_dir: 輸出目錄

    Returns:
        訓練結果JSON字串
    """
    tool = RLTrainingTool()
    return tool.run({
        "action": "train",
        "algorithm": "grpo",
        "model_name": model_name,
        "max_samples": max_samples,
        "num_epochs": num_epochs,
        "output_dir": output_dir
    })


def load_dataset(
    format_type: str = "sft",
    split: str = "train",
    max_samples: int = 100
) -> str:
    """
    載入資料集（便捷函式）

    Args:
        format_type: 資料格式 ("sft", "rl")
        split: 資料集劃分 ("train", "test")
        max_samples: 最大樣本數

    Returns:
        資料集資訊JSON字串
    """
    tool = RLTrainingTool()
    return tool.run({
        "action": "load_dataset",
        "format": format_type,
        "split": split,
        "max_samples": max_samples
    })


def create_reward_function(
    reward_type: str = "accuracy",
    **kwargs
) -> str:
    """
    建立獎勵函式（便捷函式）

    Args:
        reward_type: 獎勵類型 ("accuracy", "length_penalty", "step")
        **kwargs: 其他參數
            - penalty_weight: 長度懲罰權重（僅length_penalty）
            - step_bonus: 步驟獎勵（僅step）

    Returns:
        獎勵函式資訊JSON字串
    """
    tool = RLTrainingTool()
    params = {
        "action": "create_reward",
        "reward_type": reward_type
    }
    params.update(kwargs)
    return tool.run(params)


def evaluate_model(
    model_path: str,
    max_samples: int = 100
) -> str:
    """
    評估模型性能（便捷函式）

    Args:
        model_path: 模型路徑
        max_samples: 評估樣本數

    Returns:
        評估結果JSON字串
    """
    tool = RLTrainingTool()
    return tool.run({
        "action": "evaluate",
        "model_path": model_path,
        "max_samples": max_samples
    })
 
