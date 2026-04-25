"""????????????"""

import ast
from typing import Optional, List, Dict
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

# 預設規劃器提示詞模板
DEFAULT_PLANNER_PROMPT = """
你是一個頂級的AI規劃專家。你的任務是將使用者提出的複雜問題分解成一個由多個簡單步驟組成的行動計劃。
請確保計劃中的每個步驟都是一個獨立的、可執行的子任務，並且嚴格按照邏輯順序排列。
你的輸出必須是一個Python列表，其中每個元素都是一個描述子任務的字串。

問題: {question}

請嚴格按照以下格式輸出你的計劃:
```python
["步驟1", "步驟2", "步驟3", ...]
```
"""

# 預設執行器提示詞模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位頂級的AI執行專家。你的任務是嚴格按照給定的計劃，一步步地解決問題。
你將收到原始問題、完整的計劃、以及到目前為止已經完成的步驟和結果。
請你專注於解決"目前步驟"，並僅輸出該步驟的最終答案，不要輸出任何額外的解釋或對話。

# 原始問題:
{question}

# 完整計劃:
{plan}

# 歷史步驟與結果:
{history}

# 目前步驟:
{current_step}

請僅輸出針對"目前步驟"的回答:
"""

class Planner:
    """規劃器 - 負責將複雜問題分解為簡單步驟"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成執行計劃

        Args:
            question: 要解決的問題
            **kwargs: LLM呼叫參數

        Returns:
            步驟列表
        """
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成計劃 ---")
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"✅ 計劃已生成:\n{response_text}")

        try:
            # 提取Python代碼塊中的列表
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析計劃時出錯: {e}")
            print(f"原始回應: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析計劃時發生未知錯誤: {e}")
            return []

class Executor:
    """執行器 - 負責按計劃逐步執行"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按計劃執行任務

        Args:
            question: 原始問題
            plan: 執行計劃
            **kwargs: LLM呼叫參數

        Returns:
            最終答案
        """
        history = ""
        final_answer = ""

        print("\n--- 正在執行計劃 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在執行步驟 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "無",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.invoke(messages, **kwargs) or ""

            history += f"步驟 {i}: {step}\n結果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步驟 {i} 已完成，結果: {final_answer}")

        return final_answer

class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent - 分解規劃與逐步執行的智慧代理
    
    這個Agent能夠：
    1. 將複雜問題分解為簡單步驟
    2. 按照計劃逐步執行
    3. 維護執行歷史和上下文
    4. 得出最終答案
    
    特別適合多步驟推理、數學問題、複雜分析等任務。
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化PlanAndSolveAgent

        Args:
            name: Agent名稱
            llm: LLM實例
            system_prompt: 系統提示詞
            config: 設定對象
            custom_prompts: 自定義提示詞模板 {"planner": "", "executor": ""}
        """
        super().__init__(name, llm, system_prompt, config)

        # 設定提示詞模板：使用者自定義優先，否則使用預設模板
        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        執行Plan and Solve Agent
        
        Args:
            input_text: 要解決的問題
            **kwargs: 其他參數
            
        Returns:
            最終答案
        """
        print(f"\n🤖 {self.name} 開始處理問題: {input_text}")
        
        # 1. 生成計劃
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "無法生成有效的行動計劃，任務終止。"
            print(f"\n--- 任務終止 ---\n{final_answer}")
            
            # 保存到歷史紀錄
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            
            return final_answer
        
        # 2. 執行計劃
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任務完成 ---\n最終答案: {final_answer}")
        
        # 保存到歷史紀錄
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer
