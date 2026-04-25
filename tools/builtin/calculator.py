"""計算器工具"""

import ast
import operator
import math
from typing import Dict, Any

from ..base import Tool

class CalculatorTool(Tool):
    """Python計算器工具"""
    
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
        super().__init__(
            name="python_calculator",
            description="執行數學計算。支援基本運算、數學函式等。例如：2+3*4, sqrt(16), sin(pi/2)等。"
        )
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        執行計算

        Args:
            parameters: 包含input參數的字典

        Returns:
            計算結果
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
        """遞歸計算AST節點"""
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
        """取得工具參數定義"""
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
    執行數學計算

    Args:
        expression: 數學表達式

    Returns:
        計算結果字串
    """
    tool = CalculatorTool()
    return tool.run({"input": expression})
 
