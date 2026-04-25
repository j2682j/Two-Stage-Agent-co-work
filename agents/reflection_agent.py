"""??????????????"""

from typing import Optional, List, Dict, Any
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

# 預設提示詞模板
DEFAULT_PROMPTS = {
    "initial": """
請根據以下要求完成任務：

任務: {task}

請提供一個完整、準確的回答。
""",
    "reflect": """
請仔細審查以下回答，並找出可能的問題或改進空間：

# 原始任務:
{task}

# 目前回答:
{content}

請分析這個回答的品質，指出不足之處，並提出具體的改進建議。
如果回答已經很好，請回答"無需改進"。
""",
    "refine": """
請根據反饋意見改進你的回答：

# 原始任務:
{task}

# 上一輪回答:
{last_attempt}

# 反饋意見:
{feedback}

請提供一個改進後的回答。
"""
}

class Memory:
    """
    簡單的短期記憶模組，用於儲存智慧代理的行動與反思軌跡。
    """
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """向記憶中添加一條新紀錄"""
        self.records.append({"type": record_type, "content": content})
        print(f"📝 記憶已更新，新增一條 '{record_type}' 紀錄。")

    def get_trajectory(self) -> str:
        """將所有記憶紀錄格式化為一個連貫的字串文字"""
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一輪嘗試 (代碼) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 評審員反饋 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """取得最近一次的執行結果"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""

class ReflectionAgent(Agent):
    """
    Reflection Agent - 自我反思與迭代優化的智慧代理

    這個Agent能夠：
    1. 執行初始任務
    2. 對結果進行自我反思
    3. 根據反思結果進行優化
    4. 迭代改進直到滿意

    特別適合代碼生成、文檔寫作、分析報告等需要迭代優化的任務。

    支援多種專業領域的提示詞模板，使用者可以自定義或使用內建模板。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化ReflectionAgent

        Args:
            name: Agent名稱
            llm: LLM實例
            system_prompt: 系統提示詞
            config: 設定對象
            max_iterations: 最大迭代次數
            custom_prompts: 自定義提示詞模板 {"initial": "", "reflect": "", "refine": ""}
        """
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory = Memory()

        # 設定提示詞模板：使用者自定義優先，否則使用預設模板
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        執行Reflection Agent

        Args:
            input_text: 任務描述
            **kwargs: 其他參數

        Returns:
            最終優化後的結果
        """
        print(f"\n🤖 {self.name} 開始處理任務: {input_text}")

        # 重置記憶
        self.memory = Memory()

        # 1. 初始執行
        print("\n--- 正在進行初始嘗試 ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        # 2. 迭代循環：反思與優化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 輪迭代 ---")

            # a. 反思
            print("\n-> 正在進行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # b. 檢查是否需要停止
            if "無需改進" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思認為結果已無需改進，任務完成。")
                break

            # c. 優化
            print("\n-> 正在進行優化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- 任務完成 ---\n最終結果:\n{final_result}")

        # 保存到歷史紀錄
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        return final_result
    
    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """呼叫LLM並取得完整回應"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
