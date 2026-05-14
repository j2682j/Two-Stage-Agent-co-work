

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
    """
    負責在 agents.plan_solve_agent 中封裝 Planner，封裝代理節點的推理、工具使用、訊息傳遞或協作控制邏輯。
    
    Args:
        llm_client: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        prompt_template: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        """
        負責執行 Planner 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            llm_client: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            prompt_template: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        負責執行 Planner 中的 plan 流程，依照 Planner 的流程需求處理 plan 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    """
    負責在 agents.plan_solve_agent 中封裝 Executor，封裝代理節點的推理、工具使用、訊息傳遞或協作控制邏輯。
    
    Args:
        llm_client: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        prompt_template: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        """
        負責執行 Executor 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            llm_client: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            prompt_template: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        負責執行 Executor 中的 execute 流程，依照 Executor 的流程需求處理 execute 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            plan: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    負責在 agents.plan_solve_agent 中封裝 PlanAndSolveAgent，封裝代理節點的推理、工具使用、訊息傳遞或協作控制邏輯。
    
    Args:
        name: 此流程需要使用的輸入資料。
        llm: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        system_prompt: 此流程需要使用的輸入資料。
        config: 控制此流程行為的設定資料。
        custom_prompts: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
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
        負責執行 PlanAndSolveAgent 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            llm: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            system_prompt: 此流程需要使用的輸入資料。
            config: 控制此流程行為的設定資料。
            custom_prompts: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        負責執行 PlanAndSolveAgent 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            input_text: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
