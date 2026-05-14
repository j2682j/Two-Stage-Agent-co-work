"""Benchmark 評估模組。"""

from importlib import import_module

__all__ = [
    "BFCLAdapter",
    "BFCLEvaluator",
    "GAIAAdapter",
    "GAIAEvaluator",
    "LLMJudgeEvaluator",
    "WinRateEvaluator",
]

_EXPORT_MAP = {
    "BFCLAdapter": ("evaluation.benchmarks.bfcl_adapter", "BFCLAdapter"),
    "BFCLEvaluator": ("evaluation.benchmarks.bfcl.evaluator", "BFCLEvaluator"),
    "GAIAAdapter": ("evaluation.benchmarks.gaia_adapter", "GAIAAdapter"),
    "GAIAEvaluator": ("evaluation.benchmarks.gaia.evaluator", "GAIAEvaluator"),
    "LLMJudgeEvaluator": ("evaluation.benchmarks.data_generation.llm_judge", "LLMJudgeEvaluator"),
    "WinRateEvaluator": ("evaluation.benchmarks.data_generation.win_rate", "WinRateEvaluator"),
}


def __getattr__(name: str):
    """
    負責延遲載入 evaluation.benchmarks 對外暴露的 benchmark 類別。

    Args:
        name: 呼叫端要求的屬性名稱。

    Returns:
        對應的類別或函式物件。

    限制或副作用:
        第一次存取時會 import 對應模組，並把結果快取到 globals()。
    """
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
