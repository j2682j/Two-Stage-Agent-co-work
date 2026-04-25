"""工具鏈管理器 - HelloAgents工具鏈式呼叫支援"""

from typing import List, Dict, Any, Optional
from .registry import ToolRegistry


class ToolChain:
    """工具鏈 - 支援多個工具的順序執行"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: str = None):
        """
        添加工具執行步驟
        
        Args:
            tool_name: 工具名稱
            input_template: 輸入模板，支援變量替換，如 "{input}" 或 "{search_result}"
            output_key: 輸出結果的鍵名，用於後續步驟引用
        """
        step = {
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result"
        }
        self.steps.append(step)
        print(f"[OK] 工具鏈 '{self.name}' 添加步驟: {tool_name}")

    def execute(self, registry: ToolRegistry, input_data: str, context: Dict[str, Any] = None) -> str:
        """
        執行工具鏈
        
        Args:
            registry: 工具註冊表
            input_data: 初始輸入資料
            context: 執行上下文，用於變量替換
            
        Returns:
            最終執行結果
        """
        if not self.steps:
            return "[ERROR] 工具鏈為空，無法執行"

        print(f"[INFO] 開始執行工具鏈: {self.name}")
        
        # 初始化上下文
        if context is None:
            context = {}
        context["input"] = input_data
        
        final_result = input_data
        
        for i, step in enumerate(self.steps):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]
            
            print(f"[INFO] 執行步驟 {i+1}/{len(self.steps)}: {tool_name}")
            
            # 替換模板中的變量
            try:
                actual_input = input_template.format(**context)
            except KeyError as e:
                return f"[ERROR] 模板變量替換失敗: {e}"
            
            # 執行工具
            try:
                result = registry.execute_tool(tool_name, actual_input)
                context[output_key] = result
                final_result = result
                print(f"[OK] 步驟 {i+1} 完成")
            except Exception as e:
                return f"[ERROR] 工具 '{tool_name}' 執行失敗: {e}"
        
        print(f"🎉 工具鏈 '{self.name}' 執行完成")
        return final_result


class ToolChainManager:
    """工具鏈管理器"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        """註冊工具鏈"""
        self.chains[chain.name] = chain
        print(f"[OK] 工具鏈 '{chain.name}' 已註冊")

    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        """執行指定的工具鏈"""
        if chain_name not in self.chains:
            return f"[ERROR] 工具鏈 '{chain_name}' 不存在"

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """列出所有已註冊的工具鏈"""
        return list(self.chains.keys())

    def get_chain_info(self, chain_name: str) -> Optional[Dict[str, Any]]:
        """取得工具鏈資訊"""
        if chain_name not in self.chains:
            return None
        
        chain = self.chains[chain_name]
        return {
            "name": chain.name,
            "description": chain.description,
            "steps": len(chain.steps),
            "step_details": [
                {
                    "tool_name": step["tool_name"],
                    "input_template": step["input_template"],
                    "output_key": step["output_key"]
                }
                for step in chain.steps
            ]
        }


# 便捷函式
def create_research_chain() -> ToolChain:
    """建立一個研究工具鏈：搜尋 -> 計算 -> 總結"""
    chain = ToolChain(
        name="research_and_calculate",
        description="搜尋資訊並進行相關計算"
    )

    # 步驟1：搜尋資訊
    chain.add_step(
        tool_name="search",
        input_template="{input}",
        output_key="search_result"
    )

    # 步驟2：基於搜尋結果進行計算
    chain.add_step(
        tool_name="my_calculator",
        input_template="2 + 2",  # 簡單的計算範例
        output_key="calc_result"
    )

    return chain


def create_simple_chain() -> ToolChain:
    """建立一個簡單的工具鏈範例"""
    chain = ToolChain(
        name="simple_demo",
        description="簡單的工具鏈演示"
    )

    # 只包含一個計算步驟
    chain.add_step(
        tool_name="my_calculator",
        input_template="{input}",
        output_key="result"
    )

    return chain
