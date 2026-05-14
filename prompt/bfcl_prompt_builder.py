from __future__ import annotations

import json
from typing import Any

from .builder import PromptBuilder, PromptPacket


class BFCLPromptBuilder(PromptBuilder):
    """
    負責在 prompt.bfcl_prompt_builder 中封裝 BFCLPromptBuilder，組裝 BFCL function calling 評估需要的 stage1、stage2 與 final decision prompt。

    Args:
        config: 控制 prompt 壓縮、選取與渲染行為的設定資料。

    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法產生 BFCL 專用提示詞。

    限制或副作用:
        只負責產生 prompt 文字或 chat messages，不負責執行模型呼叫，也不驗證模型輸出的 tool call 是否正確。
    """

    def gather(self, **kwargs) -> list[PromptPacket]:
        """
        負責執行 BFCLPromptBuilder 中的 gather 流程，收集 BFCL 題目、function schema、記憶與候選 tool call 等 prompt 片段。

        Args:
            **kwargs: 目前 BFCL prompt 建構需要的 question、functions、memory_context、stage1_result 或 candidates。

        Returns:
            依重要性標記的 PromptPacket 清單。

        限制或副作用:
            只整理輸入資料，不會壓縮 function schema 內部欄位，也不會修改呼叫端傳入的原始資料。
        """
        packets = [
            PromptPacket(
                content=self._normalize_text(kwargs.get("question", "")),
                packet_type="question",
                priority=10.0,
            ),
            PromptPacket(
                content=self._format_functions(kwargs.get("functions", []) or []),
                packet_type="functions",
                priority=9.0,
            ),
        ]

        memory_context = self._normalize_text(kwargs.get("memory_context", ""))
        if memory_context:
            packets.append(PromptPacket(content=memory_context, packet_type="memory_context", priority=8.0))

        stage1_result = self._normalize_text(kwargs.get("stage1_result", ""))
        if stage1_result:
            packets.append(PromptPacket(content=stage1_result, packet_type="stage1_result", priority=7.0))

        candidates = kwargs.get("candidates", []) or []
        if candidates:
            packets.append(
                PromptPacket(
                    content=json.dumps(candidates, ensure_ascii=False, indent=2),
                    packet_type="candidates",
                    priority=7.0,
                )
            )

        return packets

    def select(self, packets: list[PromptPacket], **kwargs) -> list[PromptPacket]:
        """
        負責執行 BFCLPromptBuilder 中的 select 流程，保留 BFCL prompt 必要片段並依重要性排序。

        Args:
            packets: gather 產生的 PromptPacket 清單。
            **kwargs: 呼叫端提供的額外選取設定。

        Returns:
            排序後的 PromptPacket 清單。

        限制或副作用:
            BFCL function schema 是必要資訊，因此目前不會因長度自動移除 functions packet。
        """
        return sorted(packets, key=lambda packet: packet.priority, reverse=True)

    def structure(self, packets: list[PromptPacket], **kwargs) -> dict[str, Any]:
        """
        負責執行 BFCLPromptBuilder 中的 structure 流程，將 prompt 片段整理成 BFCL 渲染所需的固定欄位。

        Args:
            packets: 已選取的 PromptPacket 清單。
            **kwargs: 呼叫端提供的額外結構化設定。

        Returns:
            包含 question、functions、memory_context、stage1_result 與 candidates 的字典。

        限制或副作用:
            若同一 packet_type 重複出現，後面的內容會覆蓋前面的內容。
        """
        structured = {
            "question": "",
            "functions": "",
            "memory_context": "",
            "stage1_result": "",
            "candidates": "",
        }
        for packet in packets:
            if packet.packet_type in structured:
                structured[packet.packet_type] = packet.content
        return structured

    def compress(self, structured: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        負責執行 BFCLPromptBuilder 中的 compress 流程，控制記憶與候選內容長度，避免 prompt 過長。

        Args:
            structured: structure 產生的 BFCL prompt 欄位。
            **kwargs: 呼叫端提供的壓縮設定。

        Returns:
            壓縮後的 BFCL prompt 欄位。

        限制或副作用:
            不壓縮 functions 欄位，避免破壞 BFCL function schema 的完整性。
        """
        structured["memory_context"] = self._compress_multiline_text(
            structured.get("memory_context", ""),
            max_lines=8,
            max_chars=1200,
        )
        structured["candidates"] = self._compress_multiline_text(
            structured.get("candidates", ""),
            max_lines=80,
            max_chars=5000,
        )
        return structured

    def render(self, compressed: dict[str, Any], **kwargs) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 render 流程，依指定 target 產生 BFCL prompt 文字。

        Args:
            compressed: compress 產生的 BFCL prompt 欄位。
            **kwargs: 包含 target 的額外渲染設定，target 可為 stage1、stage2、final 或 single。

        Returns:
            可直接送入模型的 BFCL prompt 文字。

        限制或副作用:
            預設 target 為 single，會產生相容舊版 BFCLEvaluator 的單段 function calling prompt。
        """
        target = str(kwargs.get("target", "single") or "single").lower()
        if target == "stage1":
            return self._render_stage1(compressed)
        if target == "stage2":
            return self._render_stage2(compressed)
        if target in {"final", "final_decision"}:
            return self._render_final_decision(compressed)
        return self._render_single_function_calling(compressed)

    def build_function_calling_prompt(self, question: str, functions: list[dict[str, Any]]) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 build_function_calling_prompt 流程，產生相容舊版 BFCL evaluator 的單段 function calling prompt。

        Args:
            question: 目前要處理的 BFCL 問題文字。
            functions: 題目提供的可用 function schema 清單。

        Returns:
            要求模型直接輸出 BFCL tool call JSON list 的 prompt 文字。

        限制或副作用:
            若 functions 為空，會直接回傳原始 question，維持舊版 evaluator 行為。
        """
        if not functions:
            return question
        return self.build(question=question, functions=functions, target="single")

    def build_stage1_tool_reasoning_prompt(
        self,
        *,
        question: str,
        functions: list[dict[str, Any]],
        memory_context: str = "",
    ) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 build_stage1_tool_reasoning_prompt 流程，產生 stage1 只推理工具使用策略的 prompt。

        Args:
            question: 目前要處理的 BFCL 問題文字。
            functions: 題目提供的可用 function schema 清單。
            memory_context: GraphMemory 取回的相關任務、錯誤經驗或策略提醒。

        Returns:
            要求模型輸出工具選擇推理而不輸出最終 tool call 的 prompt 文字。

        限制或副作用:
            此 prompt 不要求模型產生可評分的 BFCL tool call，只供 stage2 參考。
        """
        return self.build(
            question=question,
            functions=functions,
            memory_context=memory_context,
            target="stage1",
        )

    def build_stage2_tool_call_prompt(
        self,
        *,
        question: str,
        functions: list[dict[str, Any]],
        stage1_result: str = "",
        memory_context: str = "",
    ) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 build_stage2_tool_call_prompt 流程，產生 stage2 top_k agent 輸出 BFCL tool call 的 prompt。

        Args:
            question: 目前要處理的 BFCL 問題文字。
            functions: 題目提供的可用 function schema 清單。
            stage1_result: stage1 對工具選擇、參數與不呼叫工具可能性的推理結果。
            memory_context: GraphMemory 取回的相關任務、錯誤經驗或策略提醒。

        Returns:
            要求模型只輸出 BFCL tool call JSON list 的 prompt 文字。

        限制或副作用:
            prompt 會要求 JSON only，但仍需後續 parser 驗證模型是否真的遵守格式。
        """
        return self.build(
            question=question,
            functions=functions,
            stage1_result=stage1_result,
            memory_context=memory_context,
            target="stage2",
        )

    def build_final_decision_prompt(
        self,
        *,
        question: str,
        functions: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        stage1_result: str = "",
        memory_context: str = "",
    ) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 build_final_decision_prompt 流程，產生 final decision 比較候選 tool call 的 prompt。

        Args:
            question: 目前要處理的 BFCL 問題文字。
            functions: 題目提供的可用 function schema 清單。
            candidates: stage2 top_k agent 產生的候選 tool call 與相關推理。
            stage1_result: stage1 的工具使用策略推理結果。
            memory_context: GraphMemory 取回的相關任務、錯誤經驗或策略提醒。

        Returns:
            要求模型選出最終 BFCL tool call JSON list 的 prompt 文字。

        限制或副作用:
            此 prompt 只做候選選擇與修正，不會執行候選 tool call。
        """
        return self.build(
            question=question,
            functions=functions,
            candidates=candidates,
            stage1_result=stage1_result,
            memory_context=memory_context,
            target="final",
        )

    def _format_functions(self, functions: list[dict[str, Any]]) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 _format_functions 流程，將 BFCL function schema 整理成模型可讀的文字區塊。

        Args:
            functions: 題目提供的可用 function schema 清單。

        Returns:
            包含 function 名稱、描述與 parameters JSON 的文字區塊。

        限制或副作用:
            只讀取 name、description 與 parameters 欄位；未知欄位不會額外展開。
        """
        if not functions:
            return "No functions are available."

        lines: list[str] = []
        for index, func in enumerate(functions, 1):
            func_name = func.get("name", f"function_{index}")
            func_desc = func.get("description", "")
            func_params = func.get("parameters", {})
            lines.append(f"Function {index}: {func_name}")
            lines.append(f"Description: {func_desc}")
            if func_params:
                lines.append("Parameters:")
                lines.append(json.dumps(func_params, ensure_ascii=False, indent=2))
            else:
                lines.append("Parameters: {}")
            lines.append("")
        return "\n".join(lines).strip()

    def _json_output_contract(self) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 _json_output_contract 流程，產生 BFCL tool call JSON 輸出格式規範。

        Args:
            無。

        Returns:
            描述 BFCL tool call JSON list 格式的文字。

        限制或副作用:
            只描述輸出格式，不會檢查模型是否遵守此格式。
        """
        return (
            "Return JSON only. The output must be a JSON list of function calls:\n"
            '[{"name": "function_name", "arguments": {"param1": "value1"}}]\n'
            "If no function should be called, return an empty JSON list: []"
        )

    def _render_single_function_calling(self, compressed: dict[str, Any]) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 _render_single_function_calling 流程，產生單段 BFCL function calling prompt。

        Args:
            compressed: 已整理好的 question 與 functions 欄位。

        Returns:
            相容舊版 BFCLEvaluator 的 BFCL prompt 文字。

        限制或副作用:
            這個 prompt 會要求模型直接輸出 tool call JSON list，不包含 stage1/stage2 拆分語意。
        """
        return f"""
You are a function calling assistant. Choose the correct function calls to answer the user question.

Available functions:
{compressed["functions"]}

User question:
{compressed["question"]}

{self._json_output_contract()}
        """.strip()

    def _render_stage1(self, compressed: dict[str, Any]) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 _render_stage1 流程，產生 BFCL stage1 工具使用推理 prompt。

        Args:
            compressed: 已整理好的 BFCL prompt 欄位。

        Returns:
            要求模型輸出工具使用策略 JSON 的 prompt 文字。

        限制或副作用:
            stage1 不輸出最終 BFCL tool call，避免在第一階段實際決定工具呼叫。
        """
        memory_block = f"\nRelevant memory:\n{compressed['memory_context']}\n" if compressed["memory_context"] else ""
        return f"""
You are in stage1 of a BFCL function calling task.
Your job is to reason about tool use only. Do not produce the final function call list yet.

Available functions:
{compressed["functions"]}
{memory_block}
User question:
{compressed["question"]}

Return JSON only with:
{{
  "reasoning": "brief explanation of which function(s) may be needed and why",
  "candidate_functions": ["function_name"],
  "required_arguments": {{"function_name": ["arg1", "arg2"]}},
  "no_call_likely": false
}}
        """.strip()

    def _render_stage2(self, compressed: dict[str, Any]) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 _render_stage2 流程，產生 BFCL stage2 tool call 生成 prompt。

        Args:
            compressed: 已整理好的 BFCL prompt 欄位。

        Returns:
            要求模型輸出 BFCL tool call JSON list 的 prompt 文字。

        限制或副作用:
            stage2 可參考 stage1 與 memory，但最終仍需 parser 和 evaluator 驗證格式與正確性。
        """
        memory_block = f"\nRelevant memory:\n{compressed['memory_context']}\n" if compressed["memory_context"] else ""
        stage1_block = f"\nStage1 tool-use reasoning:\n{compressed['stage1_result']}\n" if compressed["stage1_result"] else ""
        return f"""
You are in stage2 of a BFCL function calling task.
Generate the function call list that best answers the user question.

Available functions:
{compressed["functions"]}
{memory_block}{stage1_block}
User question:
{compressed["question"]}

{self._json_output_contract()}
        """.strip()

    def _render_final_decision(self, compressed: dict[str, Any]) -> str:
        """
        負責執行 BFCLPromptBuilder 中的 _render_final_decision 流程，產生 BFCL final decision prompt。

        Args:
            compressed: 已整理好的 BFCL prompt 欄位。

        Returns:
            要求模型比較候選並輸出最終 BFCL tool call JSON list 的 prompt 文字。

        限制或副作用:
            final decision 可修正候選格式或參數，但不會執行任何 function。
        """
        memory_block = f"\nRelevant memory:\n{compressed['memory_context']}\n" if compressed["memory_context"] else ""
        stage1_block = f"\nStage1 tool-use reasoning:\n{compressed['stage1_result']}\n" if compressed["stage1_result"] else ""
        return f"""
You are the final decision step for a BFCL function calling task.
Compare the candidate function call lists and output the single best final call list.

Available functions:
{compressed["functions"]}
{memory_block}{stage1_block}
User question:
{compressed["question"]}

Candidate outputs:
{compressed["candidates"] if compressed["candidates"] else "No candidates were provided."}

{self._json_output_contract()}
        """.strip()
