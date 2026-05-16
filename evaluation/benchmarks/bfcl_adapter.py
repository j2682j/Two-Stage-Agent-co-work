from __future__ import annotations

import ast
import json
from typing import Any

from evaluation.benchmark_adapter import BaseBenchmarkAdapter
from network.core.task_context import TaskContext
from parser.bfcl_tool_call_parser import BFCLToolCallParser
from prompt.bfcl_prompt_builder import BFCLPromptBuilder


class BFCLAdapter(BaseBenchmarkAdapter):
    """BFCLAdapter 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    def __init__(
        self,
        agent: Any,
        use_two_stage: bool = True,
        include_reasoning: bool = False,
        name: str | None = None,
    ):
        """初始化 BFCLAdapter 實例。
        
        參數:
            agent: 此流程需要使用的輸入資料。
            use_two_stage: 此流程需要使用的輸入資料。
            include_reasoning: 此流程需要使用的輸入資料。
            name: 此流程需要使用的輸入資料。
        """
        super().__init__(agent=agent, name=name or "AgentNetwork")
        self.use_two_stage = use_two_stage
        self.include_reasoning = include_reasoning
        self.prompt_builder = BFCLPromptBuilder()
        self.tool_call_parser = BFCLToolCallParser()

    def normalize_question(self, question: str) -> str:
        """處理 normalize_question 流程並回傳結果。
        
        參數:
            question: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        return question.strip() if question else ""

    def run(self, prompt: str) -> str:
        """執行 run 流程並回傳結果。
        
        參數:
            prompt: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
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
        """執行 run_sample 流程並回傳結果。
        
        參數:
            sample: 此流程需要使用的輸入資料。
            prompt_builder: 此流程需要使用的輸入資料。
            parser: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        builder = prompt_builder or self.prompt_builder
        tool_parser = parser or self.tool_call_parser
        question = self.normalize_question(str(sample.get("question", "") or ""))
        functions = sample.get("function", sample.get("functions", [])) or []
        task_id = str(sample.get("id", sample.get("task_id", "")) or "")

        prompt = self._build_workflow_prompt(builder, question=question, functions=functions)
        context = TaskContext(
            benchmark="BFCL",
            task_id=task_id,
            task_type="function_calling",
            category=str(sample.get("category", "") or ""),
            question=question,
            functions=functions,
        )

        if self.use_two_stage:
            result = self.agent.forward_two_stage(prompt, context=context)
            final_response = str(result.get("final_result", "") or "")
        else:
            if hasattr(self.agent, "set_task_context"):
                self.agent.set_task_context(context)
            final_response, *_ = self.agent.forward(prompt)
            result = {
                "final_result": final_response,
                "stage1_result": "",
                "top_k_indices": [],
                "stage2_outputs": [],
            }

        final_decision = getattr(self.agent, "last_final_decision", None) or {}
        final_response, parse_metadata = self._select_best_bfcl_response(
            final_response,
            final_decision=final_decision,
            parser=tool_parser,
        )
        stage2_outputs = list(result.get("stage2_outputs", []) or [])
        parsed_stage2_outputs = self._parse_stage2_outputs(stage2_outputs, tool_parser)

        return {
            "prompt": prompt,
            "context": context.to_dict(),
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
        """建立 build_workflow_prompt 所需的資料或輸出。
        
        參數:
            builder: 此流程需要使用的輸入資料。
            question: 此流程需要使用的輸入資料。
            functions: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        functions_text = builder._format_functions(functions)
        return (
            "You are solving a BFCL function-calling task inside a multi-agent workflow.\n\n"
            "Available functions:\n"
            f"{functions_text}\n\n"
            "User question:\n"
            f"{question}\n\n"
            "BFCL final answer contract:\n"
            '- The final answer value must be a JSON list of function calls: '
            '[{"name": "function_name", "arguments": {"param1": "value1"}}]\n'
            "- If no function should be called, the final answer value must be [].\n"
            "- Use exactly one of the function names listed in Available functions; preserve dotted module prefixes.\n"
            "- Do not shorten, rename, or invent function names.\n"
            "- Put only function parameters inside arguments; do not put computed return values, explanations, or extra keys.\n"
            "- Use only parameter names from that function's schema; include required parameters and optional parameters only when needed.\n"
            '- Do not wrap calls in {"function_call": ...}, {"final_answer": ...}, or any other object.\n'
            "- For mathematical expression string arguments, use Python-style exponentiation **, never ^.\n"
            "- Use double quotes inside the BFCL JSON list.\n\n"
            "Workflow requirements:\n"
            "- Stage1 may reason about which function should be used, but it must still follow the AgentNetwork reply format.\n"
            "- Stage2 and final decision must preserve a valid BFCL function-call answer.\n"
            "- When the AgentNetwork prompt asks for FINAL_ANSWER, put only the BFCL JSON list as the FINAL_ANSWER value.\n"
            "- Do not output a bare JSON list as the whole reply during Stage1; include REASONING, FINAL_ANSWER, and WEIGHTS when requested."
        )

    def _normalize_bfcl_json_response(self, response: str) -> str:
        """處理 normalize_bfcl_json_response 流程並回傳結果。
        
        參數:
            response: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
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

    def _select_best_bfcl_response(
        self,
        response: str,
        *,
        final_decision: dict[str, Any],
        parser: BFCLToolCallParser,
    ) -> tuple[str, dict[str, Any]]:
        """處理 select_best_bfcl_response 流程並回傳結果。
        
        參數:
            response: 此流程需要使用的輸入資料。
            final_decision: 此流程需要使用的輸入資料。
            parser: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
        """
        candidates = [
            response,
            final_decision.get("final_result", ""),
            final_decision.get("final_reply", ""),
        ]
        seen: set[str] = set()
        first_response = self._normalize_bfcl_json_response(str(response or ""))
        first_metadata = parser.parse_with_metadata(first_response)

        for candidate in candidates:
            normalized = self._normalize_bfcl_json_response(str(candidate or ""))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            metadata = parser.parse_with_metadata(normalized)
            if metadata.get("calls"):
                selected = json.dumps(metadata["calls"], ensure_ascii=False)
                return selected, parser.parse_with_metadata(selected)

        return first_response, first_metadata

    def _parse_stage2_outputs(
        self,
        stage2_outputs: list[dict[str, Any]],
        parser: BFCLToolCallParser,
    ) -> list[dict[str, Any]]:
        """解析輸入內容並回傳結構化結果。
        
        參數:
            stage2_outputs: 此流程需要使用的輸入資料。
            parser: 此流程需要使用的輸入資料。
        
        回傳:
            此函式的處理結果。
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
