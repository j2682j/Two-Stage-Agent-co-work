"""計算器工具"""

import ast
import operator
import math
from typing import Dict, Any

from ..base import Tool

class CalculatorTool(Tool):
    """
    負責在 tools.builtin.calculator 中封裝 CalculatorTool，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    # 支援的操作符
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }
    
    # 支援的函式
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }
    
    def __init__(self):
        """
        負責執行 CalculatorTool 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(
            name="python_calculator",
            description="執行數學計算。支援基本運算、數學函式等。例如：2+3*4, sqrt(16), sin(pi/2)等。"
        )
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        負責執行 CalculatorTool 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            parameters: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 支援兩種參數格式：input 和 expression
        expression = parameters.get("input", "") or parameters.get("expression", "")
        if not expression:
            return "錯誤：計算表達式不能為空"

        print(f" 正在計算: {expression}")

        try:
            # 解析表達式
            node = ast.parse(expression, mode='eval')
            result = self._eval_node(node.body)
            result_str = str(result)
            print(f"[OK] 計算結果: {result_str}")
            return result_str
        except Exception as e:
            error_msg = f"計算失敗: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    
    def _eval_node(self, node):
        """
        負責執行 CalculatorTool 中的 _eval_node 流程，依照 CalculatorTool 的流程需求處理 _eval_node 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            node: 圖結構中的節點、邊或相關識別資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.OPERATORS[type(node.op)](
                self._eval_node(node.left), 
                self._eval_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            else:
                raise ValueError(f"不支援的函式: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            else:
                raise ValueError(f"未定義的變量: {node.id}")
        else:
            raise ValueError(f"不支援的表達式類型: {type(node)}")
    
    def get_parameters(self):
        """
        負責執行 CalculatorTool 中的 get_parameters 流程，依照 CalculatorTool 的流程需求處理 get_parameters 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        from ..base import ToolParameter
        return [
            ToolParameter(
                name="input",
                type="string",
                description="要計算的數學表達式，支援基本運算和數學函式",
                required=True
            )
        ]

# 便捷函式
def calculate(expression: str) -> str:
    """
    負責執行 tools.builtin.calculator 中的 calculate 流程，依照 tools.builtin.calculator 的流程需求處理 calculate 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        expression: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    tool = CalculatorTool()
    return tool.run({"input": expression})
 
