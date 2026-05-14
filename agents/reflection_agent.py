

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
    負責在 agents.reflection_agent 中封裝 Memory，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self):
        """
        負責執行 Memory 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        負責執行 Memory 中的 add_record 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            record_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            content: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.records.append({"type": record_type, "content": content})
        print(f"📝 記憶已更新，新增一條 '{record_type}' 紀錄。")

    def get_trajectory(self) -> str:
        """
        負責執行 Memory 中的 get_trajectory 流程，依照 Memory 的流程需求處理 get_trajectory 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一輪嘗試 (代碼) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 評審員反饋 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """
        負責執行 Memory 中的 get_last_execution 流程，依照 Memory 的流程需求處理 get_last_execution 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""

class ReflectionAgent(Agent):
    """
    負責在 agents.reflection_agent 中封裝 ReflectionAgent，封裝代理節點的推理、工具使用、訊息傳遞或協作控制邏輯。
    
    Args:
        name: 此流程需要使用的輸入資料。
        llm: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        system_prompt: 此流程需要使用的輸入資料。
        config: 控制此流程行為的設定資料。
        max_iterations: 控制檢索、篩選或輸出數量的數值參數。
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
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        負責執行 ReflectionAgent 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            llm: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            system_prompt: 此流程需要使用的輸入資料。
            config: 控制此流程行為的設定資料。
            max_iterations: 控制檢索、篩選或輸出數量的數值參數。
            custom_prompts: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory = Memory()

        # 設定提示詞模板：使用者自定義優先，否則使用預設模板
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        負責執行 ReflectionAgent 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            input_text: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        """
        負責執行 ReflectionAgent 中的 _get_llm_response 流程，依照 ReflectionAgent 的流程需求處理 _get_llm_response 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            prompt: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            **kwargs: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
