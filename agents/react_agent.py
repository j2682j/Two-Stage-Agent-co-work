

import re
from typing import Optional, List, Tuple
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# 預設ReAct提示詞模板
DEFAULT_REACT_PROMPT = """你是一個具備推理和行動能力的AI 助理。你可以通過思考分析問題，然後呼叫合適的工具來取得資訊，最終給出準確的答案。

## 可用工具
{tools}

## 工作流程
請嚴格按照以下格式進行回應，每次只能執行一個步驟：

Thought: 分析問題，確定需要什么資訊，制定研究策略。
Action: 選擇合適的工具取得資訊，格式為：
- `{{tool_name}}[{{tool_input}}]`：呼叫工具取得資訊。
- `Finish[研究結論]`：當你有足夠資訊得出結論時。

## 重要提醒
1. 每次回應必須包含Thought和Action兩部分
2. 工具呼叫的格式必須嚴格遵循：工具名[參數]
3. 只有當你確信有足夠資訊回答問題時，才使用Finish
4. 如果工具回傳的資訊不夠，繼續使用其他工具或相同工具的不同參數

## 目前任務
**Question:** {question}

## 執行歷史
{history}

現在開始你的推理和行動："""

class ReActAgent(Agent):
    """
    負責在 agents.react_agent 中封裝 ReActAgent，封裝代理節點的推理、工具使用、訊息傳遞或協作控制邏輯。
    
    Args:
        name: 此流程需要使用的輸入資料。
        llm: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        tool_registry: 此流程需要使用的輸入資料。
        system_prompt: 此流程需要使用的輸入資料。
        config: 控制此流程行為的設定資料。
        max_steps: 控制檢索、篩選或輸出數量的數值參數。
        custom_prompt: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        """
        負責執行 ReActAgent 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            llm: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            tool_registry: 此流程需要使用的輸入資料。
            system_prompt: 此流程需要使用的輸入資料。
            config: 控制此流程行為的設定資料。
            max_steps: 控制檢索、篩選或輸出數量的數值參數。
            custom_prompt: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(name, llm, system_prompt, config)

        # 如果沒有提供tool_registry，建立一個空的
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry

        self.max_steps = max_steps
        self.current_history: List[str] = []

        # 設定提示詞模板：使用者自定義優先，否則使用預設模板
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

    def add_tool(self, tool):
        """
        負責執行 ReActAgent 中的 add_tool 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            tool: 可呼叫的工具、工具名稱或工具註冊表。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 檢查是否是MCP工具
        if hasattr(tool, 'auto_expand') and tool.auto_expand:
            # MCP工具會自動展開為多個工具
            if hasattr(tool, '_available_tools') and tool._available_tools:
                for mcp_tool in tool._available_tools:
                    # 建立包裝工具
                    from ..tools.base import Tool
                    wrapped_tool = Tool(
                        name=f"{tool.name}_{mcp_tool['name']}",
                        description=mcp_tool.get('description', ''),
                        func=lambda input_text, t=tool, tn=mcp_tool['name']: t.run({
                            "action": "call_tool",
                            "tool_name": tn,
                            "arguments": {"input": input_text}
                        })
                    )
                    self.tool_registry.register_tool(wrapped_tool)
                print(f"✅ MCP工具 '{tool.name}' 已展開為 {len(tool._available_tools)} 個獨立工具")
            else:
                self.tool_registry.register_tool(tool)
        else:
            self.tool_registry.register_tool(tool)

    def run(self, input_text: str, **kwargs) -> str:
        """
        負責執行 ReActAgent 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            input_text: 此流程需要使用的輸入資料。
            **kwargs: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.current_history = []
        current_step = 0
        
        print(f"\n🤖 {self.name} 開始處理問題: {input_text}")
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")
            
            # 建構提示詞
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )
            
            # 呼叫LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)
            
            if not response_text:
                print("❌ 錯誤：LLM未能回傳有效回應。")
                break
            
            # 解析輸出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"🤔 思考: {thought}")
            
            if not action:
                print("⚠️ 警告：未能解析出有效的Action，流程終止。")
                break
            
            # 檢查是否完成
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 最終答案: {final_answer}")
                
                # 保存到歷史紀錄
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                
                return final_answer
            
            # 執行工具呼叫
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: 無效的Action格式，請檢查。")
                continue
            
            print(f"🎬 行動: {tool_name}[{tool_input}]")
            
            # 呼叫工具
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 觀察: {observation}")
            
            # 更新歷史
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")
        
        print("⏰ 已達到最大步數，流程終止。")
        final_answer = "抱歉，我無法在限定步數內完成這個任務。"
        
        # 保存到歷史紀錄
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer
    
    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        負責執行 ReActAgent 中的 _parse_output 流程，依照 ReActAgent 的流程需求處理 _parse_output 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Tuple[Optional[str], Optional[str]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        
        return thought, action
    
    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        負責執行 ReActAgent 中的 _parse_action 流程，依照 ReActAgent 的流程需求處理 _parse_action 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            action_text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Tuple[Optional[str], Optional[str]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def _parse_action_input(self, action_text: str) -> str:
        """
        負責執行 ReActAgent 中的 _parse_action_input 流程，依照 ReActAgent 的流程需求處理 _parse_action_input 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            action_text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""
