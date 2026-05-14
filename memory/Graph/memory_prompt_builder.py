from __future__ import annotations

from typing import Any

from prompt.builder import PromptBuilder, PromptPacket


def _clean_text(value: Any) -> str:
    """
    負責執行 memory.graph.memory_prompt_builder 中的 _clean_text 流程，依照 memory.graph.memory_prompt_builder 的流程需求處理 _clean_text 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return " ".join(str(value or "").split()).strip()


class MemoryPromptBuilder(PromptBuilder):
    """
    負責在 memory.graph.memory_prompt_builder 中封裝 MemoryPromptBuilder，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def gather(self, **kwargs: Any) -> list[PromptPacket]:
        """
        負責執行 MemoryPromptBuilder 中的 gather 流程，依照 MemoryPromptBuilder 的流程需求處理 gather 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        retrieval = kwargs.get("retrieval") or {}
        insights = kwargs.get("insights") or []
        max_failures = int(kwargs.get("max_failures", 1) or 1)
        injection_target = _clean_text(kwargs.get("injection_target") or "generic")
        return [
            PromptPacket(
                content=self._render_guidance(
                    retrieval,
                    insights=insights,
                    max_failures=max_failures,
                    injection_target=injection_target,
                ),
                packet_type="graph_memory_guidance",
                metadata={"source": "graph_memory", "injection_target": injection_target},
                priority=1.0,
            )
        ]

    def select(self, packets: list[PromptPacket], **kwargs: Any) -> list[PromptPacket]:
        """
        負責執行 MemoryPromptBuilder 中的 select 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            packets: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[PromptPacket]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [packet for packet in packets if _clean_text(packet.content)]

    def structure(self, packets: list[PromptPacket], **kwargs: Any) -> dict[str, Any]:
        """
        負責執行 MemoryPromptBuilder 中的 structure 流程，依照 MemoryPromptBuilder 的流程需求處理 structure 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            packets: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {"packets": packets}

    def compress(self, structured: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        負責執行 MemoryPromptBuilder 中的 compress 流程，依照 MemoryPromptBuilder 的流程需求處理 compress 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            structured: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return structured

    def render(self, compressed: dict[str, Any], **kwargs: Any) -> str:
        """
        負責執行 MemoryPromptBuilder 中的 render 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            compressed: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return "\n\n".join(packet.content for packet in compressed.get("packets", [])).strip()

    def build_guidance_prompt(
        self,
        retrieval: dict[str, Any],
        *,
        insights: list[dict[str, Any]] | None = None,
        max_failures: int = 1,
        injection_target: str = "generic",
    ) -> str:
        """
        負責執行 MemoryPromptBuilder 中的 build_guidance_prompt 流程，組裝提示詞內容，將任務、記憶、證據或格式要求整理成模型可讀的輸入。
        
        Args:
            retrieval: 記憶系統提供的檢索結果、寫入資料或操作介面。
            insights: 記憶系統提供的檢索結果、寫入資料或操作介面。
            max_failures: 控制檢索、篩選或輸出數量的數值參數。
            injection_target: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.build(
            retrieval=retrieval,
            insights=insights or [],
            max_failures=max_failures,
            injection_target=injection_target,
        )

    def _render_guidance(
        self,
        retrieval: dict[str, Any],
        *,
        insights: list[dict[str, Any]] | None = None,
        max_failures: int = 1,
        injection_target: str = "generic",
    ) -> str:
        """
        負責執行 MemoryPromptBuilder 中的 _render_guidance 流程，依照 MemoryPromptBuilder 的流程需求處理 _render_guidance 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            retrieval: 記憶系統提供的檢索結果、寫入資料或操作介面。
            insights: 記憶系統提供的檢索結果、寫入資料或操作介面。
            max_failures: 控制檢索、篩選或輸出數量的數值參數。
            injection_target: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        target = _clean_text(injection_target).lower()
        if target == "stage1_round0":
            return self._render_stage1_round0_guidance(
                retrieval,
                insights=insights or [],
                max_failures=max_failures,
            )
        if target == "stage2_top_k":
            return self._render_stage2_top_k_guidance(
                retrieval,
                insights=insights or [],
                max_failures=max_failures,
            )

        lines = ["Relevant Memory Guidance:"]
        task_type = retrieval.get("task_type")
        if task_type:
            lines.append(f"- Task type: {task_type}")
        terms = retrieval.get("trigger_terms") or []
        if terms:
            lines.append(f"- Trigger terms: {', '.join(map(str, terms[:8]))}")

        for insight in (insights or [])[:3]:
            strategy = _clean_text(insight.get("rule") or insight.get("strategy", ""))
            if strategy:
                lines.append(f"- Strategy: {strategy}")
            checklist = insight.get("checklist") or []
            if checklist:
                compact = "; ".join(_clean_text(item) for item in checklist[:4] if _clean_text(item))
                if compact:
                    lines.append(f"- Checklist: {compact}")

        for failure in (retrieval.get("similar_failures") or [])[:max_failures]:
            summary = _clean_text(failure.get("summary", ""))
            if summary:
                lines.append(f"- Similar failure warning: {summary}")

        for record in (retrieval.get("similar_task_records") or [])[:max_failures]:
            summary = _clean_text(record.get("summary", ""))
            if summary:
                lines.append(f"- Similar task trajectory: {summary}")

        policy = retrieval.get("tool_policy") or {}
        prefer = policy.get("prefer") or []
        avoid = policy.get("avoid") or []
        policy_bits = []
        if prefer:
            policy_bits.append(f"prefer {', '.join(map(str, prefer[:4]))}")
        if avoid:
            policy_bits.append(f"avoid {', '.join(map(str, avoid[:4]))}")
        if policy_bits:
            lines.append(f"- Tool policy for later repair: {'; '.join(policy_bits)}")

        return "\n".join(lines).strip()

    def _render_stage1_round0_guidance(
        self,
        retrieval: dict[str, Any],
        *,
        insights: list[dict[str, Any]],
        max_failures: int,
    ) -> str:
        """
        負責執行 MemoryPromptBuilder 中的 _render_stage1_round0_guidance 流程，依照 MemoryPromptBuilder 的流程需求處理 _render_stage1_round0_guidance 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            retrieval: 記憶系統提供的檢索結果、寫入資料或操作介面。
            insights: 記憶系統提供的檢索結果、寫入資料或操作介面。
            max_failures: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lines = ["Stage-1 Memory Strategy:"]
        task_type = retrieval.get("task_type")
        if task_type:
            lines.append(f"- Likely task type: {task_type}")
        terms = retrieval.get("trigger_terms") or []
        if terms:
            lines.append(f"- Task signals: {', '.join(map(str, terms[:8]))}")

        strategy_count = 0
        for insight in insights[:3]:
            rule = _clean_text(insight.get("rule") or insight.get("strategy", ""))
            if rule:
                lines.append(f"- Strategy reminder: {rule}")
                strategy_count += 1
            checklist = insight.get("checklist") or []
            compact = "; ".join(_clean_text(item) for item in checklist[:3] if _clean_text(item))
            if compact:
                lines.append(f"- Before answering: {compact}")
        if strategy_count == 0:
            lines.append("- Strategy reminder: solve from the current question evidence; do not reuse a past final answer directly.")

        for failure in (retrieval.get("similar_failures") or [])[:max_failures]:
            summary = _clean_text(failure.get("summary", ""))
            if summary:
                lines.append(f"- Prior pitfall to avoid: {summary}")
        for success in (retrieval.get("similar_successes") or [])[:1]:
            summary = _clean_text(success.get("summary", ""))
            if summary:
                lines.append(f"- Successful pattern to imitate: {summary}")

        lines.append("- Output focus: produce an independent first-pass answer with concise reasoning.")
        return "\n".join(lines).strip()

    def _render_stage2_top_k_guidance(
        self,
        retrieval: dict[str, Any],
        *,
        insights: list[dict[str, Any]],
        max_failures: int,
    ) -> str:
        """
        負責執行 MemoryPromptBuilder 中的 _render_stage2_top_k_guidance 流程，依照 MemoryPromptBuilder 的流程需求處理 _render_stage2_top_k_guidance 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            retrieval: 控制檢索、篩選或輸出數量的數值參數。
            insights: 控制檢索、篩選或輸出數量的數值參數。
            max_failures: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lines = ["Stage-2 Repair Memory:"]
        task_type = retrieval.get("task_type")
        if task_type:
            lines.append(f"- Repair task type: {task_type}")

        policy = retrieval.get("tool_policy") or {}
        prefer = policy.get("prefer") or []
        optional = policy.get("optional") or []
        avoid = policy.get("avoid") or []
        if prefer:
            lines.append(f"- Prefer tools/evidence: {', '.join(map(str, prefer[:4]))}")
        if optional:
            lines.append(f"- Optional tools/evidence: {', '.join(map(str, optional[:4]))}")
        if avoid:
            lines.append(f"- Avoid: {', '.join(map(str, avoid[:4]))}")

        for insight in insights[:3]:
            rule = _clean_text(insight.get("rule") or insight.get("strategy", ""))
            if rule:
                lines.append(f"- Repair strategy: {rule}")
            checklist = insight.get("checklist") or []
            compact = "; ".join(_clean_text(item) for item in checklist[:4] if _clean_text(item))
            if compact:
                lines.append(f"- Verification checklist: {compact}")

        for failure in (retrieval.get("similar_failures") or [])[:max_failures]:
            summary = _clean_text(failure.get("summary", ""))
            if summary:
                lines.append(f"- Similar failed trajectory: {summary}")
        for success in (retrieval.get("similar_successes") or [])[:max_failures]:
            summary = _clean_text(success.get("summary", ""))
            if summary:
                lines.append(f"- Successful trace to reuse carefully: {summary}")
        for record in (retrieval.get("similar_task_records") or [])[:max_failures]:
            summary = _clean_text(record.get("summary", ""))
            if summary:
                lines.append(f"- Related trace to compare against: {summary}")

        lines.append("- Repair focus: verify the stage-1/top-k answers against evidence before finalizing.")
        return "\n".join(lines).strip()
