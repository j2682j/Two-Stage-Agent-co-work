from __future__ import annotations

import ast
import json
from typing import Any

from evaluation.benchmark_adapter import BaseBenchmarkAdapter
from parser.bfcl_tool_call_parser import BFCLToolCallParser
from prompt.bfcl_prompt_builder import BFCLPromptBuilder


class BFCLAdapter(BaseBenchmarkAdapter):
    """
    負責把 BFCL function calling 樣本轉成 AgentNetwork 可執行的任務，並整理回評估器可使用的結構化結果。

    Args:
        agent: 實際執行推理的 AgentNetwork 或相容物件。
        use_two_stage: 是否使用 AgentNetwork.forward_two_stage()。
        include_reasoning: run() 相容路徑是否附加 stage1 reasoning。
        name: 對外顯示的 adapter 名稱。

    Returns:
        BFCLAdapter 實例。

    限制或副作用:
        run_sample() 會呼叫模型與 AgentNetwork workflow，並可能觸發 GraphMemory retrieval、token usage 紀錄與工具 trace。
    """

    def __init__(
        self,
        agent: Any,
        use_two_stage: bool = True,
        include_reasoning: bool = False,
        name: str | None = None,
    ):
        """
        負責初始化 BFCLAdapter 的 AgentNetwork、prompt builder 與 tool call parser。

        Args:
            agent: 實際執行推理的 AgentNetwork 或相容物件。
            use_two_stage: 是否使用 two-stage workflow。
            include_reasoning: run() 是否把 stage1 reasoning 附加到回覆中。
            name: adapter 名稱。

        Returns:
            無。

        限制或副作用:
            只初始化內部依賴，不會立即呼叫模型。
        """
        super().__init__(agent=agent, name=name or "AgentNetwork")
        self.use_two_stage = use_two_stage
        self.include_reasoning = include_reasoning
        self.prompt_builder = BFCLPromptBuilder()
        self.tool_call_parser = BFCLToolCallParser()

    def normalize_question(self, question: str) -> str:
        """
        負責正規化 BFCL 題目文字或 evaluator 傳入的 prompt。

        Args:
            question: 原始題目或 prompt。

        Returns:
            去除頭尾空白後的字串；若輸入為空則回傳空字串。

        限制或副作用:
            不會修改題目內容中的 function schema 或 JSON 指令。
        """
        return question.strip() if question else ""

    def run(self, prompt: str) -> str:
        """
        負責提供舊版 evaluator 相容入口，將單一 prompt 交給 AgentNetwork 執行並回傳最終答案文字。

        Args:
            prompt: 已組好的 BFCL function calling prompt。

        Returns:
            AgentNetwork 的 final_result 字串；若 include_reasoning=True，會附加 stage1_result。

        限制或副作用:
            會呼叫模型；此方法不會額外解析 function call，正式 BFCL 評估應優先使用 run_sample()。
        """
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

    def run_sample(
        self,
        sample: dict[str, Any],
        *,
        prompt_builder: BFCLPromptBuilder | None = None,
        parser: BFCLToolCallParser | None = None,
    ) -> dict[str, Any]:
        """
        負責執行單一 BFCL 樣本的 two-stage workflow，並回傳 function call 解析結果與完整 trace。

        Args:
            sample: BFCL dataset 的單題資料，包含 question、function、id、category 等欄位。
            prompt_builder: 可選的 BFCLPromptBuilder；若未提供會使用 adapter 內建 builder。
            parser: 可選的 BFCLToolCallParser；若未提供會使用 adapter 內建 parser。

        Returns:
            包含 final_response、predicted_calls、parse_metadata、stage1_result、stage2_outputs、final_decision 與 context 的字典。

        限制或副作用:
            會呼叫 AgentNetwork.forward_two_stage() 或 forward()；現階段不修改 AgentNetwork 核心 prompt 流程，而是把 BFCL 指令包進輸入 prompt。
        """
        builder = prompt_builder or self.prompt_builder
        tool_parser = parser or self.tool_call_parser
        question = self.normalize_question(str(sample.get("question", "") or ""))
        functions = sample.get("function", sample.get("functions", [])) or []
        task_id = str(sample.get("id", sample.get("task_id", "")) or "")

        prompt = self._build_workflow_prompt(builder, question=question, functions=functions)
        context = {
            "benchmark": "BFCL",
            "task_id": task_id,
            "task_type": "function_calling",
            "category": sample.get("category", ""),
            "functions": functions,
        }

        if self.use_two_stage:
            result = self.agent.forward_two_stage(prompt, context=context)
            final_response = str(result.get("final_result", "") or "")
        else:
            final_response, *_ = self.agent.forward(prompt)
            result = {
                "final_result": final_response,
                "stage1_result": "",
                "top_k_indices": [],
                "stage2_outputs": [],
            }

        final_response = self._normalize_bfcl_json_response(final_response)
        parse_metadata = tool_parser.parse_with_metadata(final_response)
        stage2_outputs = list(result.get("stage2_outputs", []) or [])
        parsed_stage2_outputs = self._parse_stage2_outputs(stage2_outputs, tool_parser)
        final_decision = getattr(self.agent, "last_final_decision", None) or {}

        return {
            "prompt": prompt,
            "context": context,
            "raw_result": result,
            "raw_response": final_response,
            "final_response": final_response,
            "predicted_calls": parse_metadata.get("calls", []),
            "parse_metadata": parse_metadata,
            "stage1_result": result.get("stage1_result", getattr(self.agent, "last_stage1_result", "")),
            "top_k_indices": list(result.get("top_k_indices", getattr(self.agent, "last_top_k_indices", [])) or []),
            "stage2_outputs": parsed_stage2_outputs,
            "final_decision": final_decision,
        }

    def _build_workflow_prompt(self, builder: BFCLPromptBuilder, *, question: str, functions: list[dict[str, Any]]) -> str:
        """
        負責建立目前 AgentNetwork 可直接執行的 BFCL function calling prompt。

        Args:
            builder: BFCLPromptBuilder。
            question: BFCL 題目文字。
            functions: BFCL function schema 列表。

        Returns:
            要交給 AgentNetwork 的 prompt 字串。

        限制或副作用:
            目前使用 single function-calling prompt 包住完整 BFCL 指令，讓現有 two-stage network 產生最終 JSON function call list。
        """
        prompt = builder.build_function_calling_prompt(question, functions)
        return (
            f"{prompt}\n\n"
            "Workflow requirements:\n"
            "- Stage1 may reason about which function should be used.\n"
            "- Stage2 and final decision must preserve a valid BFCL function-call answer.\n"
            "- The final answer must be JSON only: a list of calls with name and arguments.\n"
            "- If no function should be called, the final answer must be []."
        )

    def _normalize_bfcl_json_response(self, response: str) -> str:
        """
        Normalize BFCL final answers into strict JSON without changing non-JSON text.

        Some final decision paths preserve a Python literal representation such as
        [{'name': 'tool', 'arguments': {'x': 1}}]. BFCL expects JSON, so convert
        structured list/dict literals with json.dumps instead of replacing quotes.
        """
        text = str(response or "").strip()
        if not text:
            return text

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        else:
            if isinstance(parsed, (list, dict)):
                return json.dumps(parsed, ensure_ascii=False)
            return text

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text

        if isinstance(parsed, (list, dict)):
            return json.dumps(parsed, ensure_ascii=False)
        return text

    def _parse_stage2_outputs(
        self,
        stage2_outputs: list[dict[str, Any]],
        parser: BFCLToolCallParser,
    ) -> list[dict[str, Any]]:
        """
        負責為 stage2 每個候選輸出補上 BFCL tool call 解析資訊。

        Args:
            stage2_outputs: AgentNetwork 產生的 stage2 output 列表。
            parser: BFCLToolCallParser。

        Returns:
            附加 parsed_calls、parse_source 與 parse_error 的 stage2 output 列表。

        限制或副作用:
            只複製並整理資料，不會修改原始 network.last_stage2_outputs。
        """
        parsed_outputs: list[dict[str, Any]] = []
        for output in stage2_outputs:
            item = dict(output)
            response = str(item.get("answer", item.get("reply", "")) or "")
            metadata = parser.parse_with_metadata(response)
            item["parsed_calls"] = metadata.get("calls", [])
            item["parse_source"] = metadata.get("source")
            item["parse_error"] = metadata.get("parse_error")
            parsed_outputs.append(item)
        return parsed_outputs
