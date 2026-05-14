"""工具鏈管理器 - HelloAgents工具鏈式呼叫支援"""

from typing import List, Dict, Any, Optional
from .registry import ToolRegistry


class ToolChain:
    """
    負責在 tools.chain 中封裝 ToolChain，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        name: 此流程需要使用的輸入資料。
        description: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, name: str, description: str):
        """
        負責執行 ToolChain 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: str = None):
        """
        負責執行 ToolChain 中的 add_step 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            input_template: 此流程需要使用的輸入資料。
            output_key: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
        負責執行 ToolChain 中的 execute 流程，依照 ToolChain 的流程需求處理 execute 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            registry: 可呼叫的工具、工具名稱或工具註冊表。
            input_data: 此流程需要使用的輸入資料。
            context: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    """
    負責在 tools.chain 中封裝 ToolChainManager，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        registry: 可呼叫的工具、工具名稱或工具註冊表。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, registry: ToolRegistry):
        """
        負責執行 ToolChainManager 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            registry: 可呼叫的工具、工具名稱或工具註冊表。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        """
        負責執行 ToolChainManager 中的 register_chain 流程，依照 ToolChainManager 的流程需求處理 register_chain 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            chain: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.chains[chain.name] = chain
        print(f"[OK] 工具鏈 '{chain.name}' 已註冊")

    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        """
        負責執行 ToolChainManager 中的 execute_chain 流程，依照 ToolChainManager 的流程需求處理 execute_chain 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            chain_name: 此流程需要使用的輸入資料。
            input_data: 此流程需要使用的輸入資料。
            context: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if chain_name not in self.chains:
            return f"[ERROR] 工具鏈 '{chain_name}' 不存在"

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """
        負責執行 ToolChainManager 中的 list_chains 流程，依照 ToolChainManager 的流程需求處理 list_chains 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return list(self.chains.keys())

    def get_chain_info(self, chain_name: str) -> Optional[Dict[str, Any]]:
        """
        負責執行 ToolChainManager 中的 get_chain_info 流程，依照 ToolChainManager 的流程需求處理 get_chain_info 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            chain_name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
    """
    負責執行 tools.chain 中的 create_research_chain 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 ToolChain。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
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
    """
    負責執行 tools.chain 中的 create_simple_chain 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 ToolChain。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
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
