from __future__ import annotations

from typing import Any

from .bfcl_contract import BFCLPromptContract
from .gaia_contract import GAIAPromptContract
from .generic_contract import GenericPromptContract


def _context_value(context: Any, key: str) -> str:
    if context is None:
        return ""
    if isinstance(context, dict):
        return str(context.get(key, "") or "")
    return str(getattr(context, key, "") or "")


def _looks_like_bfcl(question: str) -> bool:
    text = str(question or "")
    markers = (
        "BFCL function calling",
        "Return JSON only. The output must be a JSON list of function calls",
        "Available functions:",
        "Workflow requirements:",
    )
    return sum(1 for marker in markers if marker in text) >= 2


def resolve_prompt_contract(context: Any = None, *, question: str = ""):
    benchmark = _context_value(context, "benchmark").lower()
    source = _context_value(context, "source").lower()
    if "bfcl" in benchmark or "bfcl" in source or _looks_like_bfcl(question):
        return BFCLPromptContract(benchmark="bfcl", task_context=context)
    if "gaia" in benchmark or "gaia" in source:
        return GAIAPromptContract(benchmark="gaia", task_context=context)
    return GenericPromptContract(benchmark="generic", task_context=context)
