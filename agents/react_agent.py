"""???????? ReAct ?????"""

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
    ReAct (Reasoning and Acting) Agent
    
    結合推理和行動的智慧代理，能夠：
    1. 分析問題並制定行動計劃
    2. 呼叫外部工具取得資訊
    3. 基於觀察結果進行推理
    4. 迭代執行直到得出最終答案
    
    這是一個經典的Agent范式，特別適合需要外部資訊的任務。
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
        初始化ReActAgent

        Args:
            name: Agent名稱
            llm: LLM實例
            tool_registry: 工具註冊表（可選，如果不提供則建立空的工具註冊表）
            system_prompt: 系統提示詞
            config: 設定對象
            max_steps: 最大執行步數
            custom_prompt: 自定義提示詞模板
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
        添加工具到工具註冊表
        支援MCP工具的自動展開

        Args:
            tool: 工具實例(可以是普通Tool或MCPTool)
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
        執行ReAct Agent
        
        Args:
            input_text: 使用者問題
            **kwargs: 其他參數
            
        Returns:
            最終答案
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
        """解析LLM輸出，提取思考和行動"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        
        return thought, action
    
    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析行動文字，提取工具名稱和輸入"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def _parse_action_input(self, action_text: str) -> str:
        """解析行動輸入"""
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""
