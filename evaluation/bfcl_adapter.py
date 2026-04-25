from __future__ import annotations
from typing import Any
from .benchmark_adapter import BaseBenchmarkAdapter


class BFCLAdapter(BaseBenchmarkAdapter):
    """BFCL benchmark 的轉接器。"""

    def __init__(
        self,
        agent: Any,
        use_two_stage: bool = True,
        include_reasoning: bool = False,
        name: str | None = None,
    ):
        super().__init__(agent=agent, name=name or "AgentNetwork")
        self.use_two_stage = use_two_stage
        self.include_reasoning = include_reasoning

    def normalize_question(self, question: str) -> str:
        return question.strip() if question else ""

    def run(self, prompt: str) -> str:
        normalized_prompt = self.normalize_question(prompt)

        if self.use_two_stage:
            result = self.agent.forward_two_stage(normalized_prompt)
            final_answer = result.get("final_result", "")
            reasoning = result.get("stage1_result", "")
        else:
            final_answer, *_ = self.agent.forward(normalized_prompt)
            reasoning = ""

        if self.include_reasoning and reasoning:
            return f"{reasoning}\n{final_answer}"

        return str(final_answer)