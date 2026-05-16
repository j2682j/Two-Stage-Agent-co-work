"""evaluation.benchmarks.bfcl.__init__ 模組。

提供此模組相關的資料結構、流程輔助或整合邏輯。
"""


from importlib import import_module

__all__ = [
    "BFCLDataset",
    "BFCLEvaluator",
    "BFCLMetrics",
    "BFCLIntegration",
    "run_bfcl_evaluation",
]

_EXPORT_MAP = {
    "BFCLDataset": ("evaluation.benchmarks.bfcl.dataset", "BFCLDataset"),
    "BFCLEvaluator": ("evaluation.benchmarks.bfcl.evaluator", "BFCLEvaluator"),
    "BFCLMetrics": ("evaluation.benchmarks.bfcl.metrics", "BFCLMetrics"),
    "BFCLIntegration": ("evaluation.benchmarks.bfcl.bfcl_integration", "BFCLIntegration"),
    "run_bfcl_evaluation": ("evaluation.benchmarks.bfcl.bfcl_runner", "run_bfcl_evaluation"),
}


def __getattr__(name: str):
    """
    負責執行 evaluation.benchmarks.bfcl.__init__ 中的 __getattr__ 流程，依照 evaluation.benchmarks.bfcl.__init__ 的流程需求處理 __getattr__ 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        name: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
