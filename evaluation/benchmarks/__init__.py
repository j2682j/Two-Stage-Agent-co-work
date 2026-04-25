"""???????????"""


from importlib import import_module

__all__ = [
    "BFCLEvaluator",
    "GAIAEvaluator",
    "LLMJudgeEvaluator",
    "WinRateEvaluator",
]

_EXPORT_MAP = {
    "BFCLEvaluator": ("evaluation.benchmarks.bfcl.evaluator", "BFCLEvaluator"),
    "GAIAEvaluator": ("evaluation.benchmarks.gaia.evaluator", "GAIAEvaluator"),
    "LLMJudgeEvaluator": ("evaluation.benchmarks.data_generation.llm_judge", "LLMJudgeEvaluator"),
    "WinRateEvaluator": ("evaluation.benchmarks.data_generation.win_rate", "WinRateEvaluator"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
