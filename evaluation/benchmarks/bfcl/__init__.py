"""BFCL ???????"""


from importlib import import_module

__all__ = [
    "BFCLDataset",
    "BFCLEvaluator",
    "BFCLMetrics",
    "BFCLIntegration",
]

_EXPORT_MAP = {
    "BFCLDataset": ("evaluation.benchmarks.bfcl.dataset", "BFCLDataset"),
    "BFCLEvaluator": ("evaluation.benchmarks.bfcl.evaluator", "BFCLEvaluator"),
    "BFCLMetrics": ("evaluation.benchmarks.bfcl.metrics", "BFCLMetrics"),
    "BFCLIntegration": ("evaluation.benchmarks.bfcl.bfcl_integration", "BFCLIntegration"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
