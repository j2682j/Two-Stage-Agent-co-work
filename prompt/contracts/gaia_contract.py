from __future__ import annotations

from .base import PromptContract


class GAIAPromptContract(PromptContract):
    def stage1_output_contract(self, expected_weight_count: int, has_formers: bool) -> str:
        base = super().stage1_output_contract(expected_weight_count, has_formers)
        return (
            base
            + "\n- GAIA answers should be concise final answers, not JSON wrappers.\n"
            + "- When the question asks for a specific unit, number format, or entity name, FINAL_ANSWER must match it exactly."
        )

    def stage2_output_contract(self) -> str:
        return (
            super().stage2_output_contract()
            + "\n\nGAIA rules:\n"
            + "- final_answer must be a concise natural-language answer string.\n"
            + "- Do not return function-call JSON or tool-call wrappers."
        )
