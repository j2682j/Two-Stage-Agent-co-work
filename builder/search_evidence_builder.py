from __future__ import annotations

import re
from typing import Any

from utils.network_utils import normalize_text

from .evidence_builder import EvidenceBuilder


class SearchEvidenceBuilder(EvidenceBuilder):
    """
    負責在 builder.search_evidence_builder 中封裝 SearchEvidenceBuilder，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        tool_manager: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        memory_tool: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        runtime: 目前流程所需的上下文、狀態或附加資訊。
        search_query_planner: 已整理好的搜尋結果、共享資料包或可重用證據內容。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, tool_manager=None, memory_tool=None, runtime=None, *, search_query_planner=None):
        """
        負責執行 SearchEvidenceBuilder 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            tool_manager: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            memory_tool: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            runtime: 目前流程所需的上下文、狀態或附加資訊。
            search_query_planner: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(
            tool_manager=tool_manager,
            memory_tool=memory_tool,
            runtime=runtime,
            search_query_planner=search_query_planner,
            initialize_search_helpers=False,
        )

    def summarize_search_output(
        self,
        search_output_text: Any,
        question: str,
        max_sections: int = 3,
        max_chars_per_section: int = 300,
        max_total_chars: int = 900,
    ) -> str:
        """
        負責執行 SearchEvidenceBuilder 中的 summarize_search_output 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            search_output_text: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            question: 目前要處理的任務、問題或查詢文字。
            max_sections: 控制檢索、篩選或輸出數量的數值參數。
            max_chars_per_section: 控制檢索、篩選或輸出數量的數值參數。
            max_total_chars: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        cleaned = self._clean_search_text(search_output_text)
        if not cleaned:
            return ""

        sections = self._split_search_sections(cleaned)
        ranked_sections = self._rank_search_sections(sections, question)
        selected = ranked_sections[:max_sections]

        summarized_parts = []
        for idx, section in enumerate(selected, start=1):
            clipped = section[:max_chars_per_section].strip()
            if len(section) > max_chars_per_section:
                clipped += " ..."
            summarized_parts.append(f"[{idx}] {clipped}")

        summary = "\n".join(summarized_parts).strip()
        if len(summary) > max_total_chars:
            summary = summary[:max_total_chars].rstrip() + " ..."

        return summary

    def summarize_structured_search_result(
        self,
        search_result: dict[str, Any],
        question: str,
        max_results: int = 2,
        max_chars_per_result: int = 240,
        max_total_chars: int = 900,
    ) -> str:
        """
        負責執行 SearchEvidenceBuilder 中的 summarize_structured_search_result 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            search_result: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            question: 目前要處理的任務、問題或查詢文字。
            max_results: 控制檢索、篩選或輸出數量的數值參數。
            max_chars_per_result: 控制檢索、篩選或輸出數量的數值參數。
            max_total_chars: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        payload = search_result.get("raw_result")
        if not isinstance(payload, dict):
            return ""

        results = payload.get("results") or []
        if not results:
            return ""

        query_keywords = self._extract_query_keywords(question)
        parts: list[str] = []
        for idx, item in enumerate(results[:max_results], start=1):
            title = self._clean_search_text(item.get("title", "")) or item.get("url", "")
            url = self._clean_search_text(item.get("url", ""))
            body = self._clean_search_text(item.get("raw_content") or item.get("content") or "")
            if not body:
                continue

            clipped = body[:max_chars_per_result].strip()
            if len(body) > max_chars_per_result:
                clipped += " ..."

            bonus = ""
            rerank_score = item.get("rerank_score")
            if rerank_score is not None:
                bonus = f" (score={rerank_score})"

            keyword_hits = sum(1 for kw in query_keywords if kw in body.lower() or kw in title.lower())
            if keyword_hits == 0 and idx > 1:
                continue

            section = f"[{idx}] {title}{bonus}\n{clipped}"
            if url:
                section += f"\nSource: {url}"
            parts.append(section)

        summary = "\n".join(parts).strip()
        if len(summary) > max_total_chars:
            summary = summary[:max_total_chars].rstrip() + " ..."
        return summary

    def build_search_evidence_block(
        self,
        search_result: dict[str, Any],
        question: str,
        max_sections: int = 3,
        max_chars_per_section: int = 300,
        max_total_chars: int = 900,
    ) -> str:
        """
        負責執行 SearchEvidenceBuilder 中的 build_search_evidence_block 流程，建立任務需要的證據區塊，整理搜尋、附件或工具輸出的可引用內容。
        
        Args:
            search_result: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            question: 目前要處理的任務、問題或查詢文字。
            max_sections: 控制檢索、篩選或輸出數量的數值參數。
            max_chars_per_section: 控制檢索、篩選或輸出數量的數值參數。
            max_total_chars: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not search_result or not search_result.get("ok"):
            return ""

        summary = self.summarize_structured_search_result(
            search_result=search_result,
            question=question,
            max_results=max_sections,
            max_chars_per_result=max_chars_per_section,
            max_total_chars=max_total_chars,
        ) or self.summarize_search_output(
            search_output_text=search_result.get("output_text", ""),
            question=question,
            max_sections=max_sections,
            max_chars_per_section=max_chars_per_section,
            max_total_chars=max_total_chars,
        )

        if not summary:
            return ""

        return (
            "Search evidence:\n"
            f"Question focus: {question}\n"
            f"Tool used: {search_result.get('tool_name', 'search')}\n"
            f"Key findings:\n{summary}\n"
        )

    def build_planned_search_evidence_block(
        self,
        *,
        search_runs: list[dict[str, Any]],
        question: str,
        max_queries: int = 3,
        max_chars_per_query: int = 320,
        max_total_chars: int = 1000,
    ) -> str:
        """
        負責執行 SearchEvidenceBuilder 中的 build_planned_search_evidence_block 流程，建立任務需要的證據區塊，整理搜尋、附件或工具輸出的可引用內容。
        
        Args:
            search_runs: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            question: 目前要處理的任務、問題或查詢文字。
            max_queries: 控制檢索、篩選或輸出數量的數值參數。
            max_chars_per_query: 控制檢索、篩選或輸出數量的數值參數。
            max_total_chars: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not search_runs:
            return ""

        lines = [
            "Search evidence:",
            f"Question focus: {question}",
            "Search plan:",
        ]

        query_sections: list[str] = []
        for idx, run in enumerate(search_runs[:max_queries], start=1):
            query = str(run.get("query", "") or "").strip()
            search_result = run.get("result") or {}
            if not search_result.get("ok"):
                continue

            summary = self.summarize_structured_search_result(
                search_result=search_result,
                question=question,
                max_results=2,
                max_chars_per_result=180,
                max_total_chars=max_chars_per_query,
            ) or self.summarize_search_output(
                search_output_text=search_result.get("output_text", ""),
                question=question,
                max_sections=2,
                max_chars_per_section=180,
                max_total_chars=max_chars_per_query,
            )
            if not summary:
                continue

            lines.append(f"- Q{idx}: {query}")
            query_sections.append(f"[Q{idx}] {summary}")

        if not query_sections:
            return ""

        body = "\n".join(lines + ["Key findings:", "\n".join(query_sections)]).strip()
        if len(body) > max_total_chars:
            body = body[:max_total_chars].rstrip() + " ..."
        return body

    def _clean_search_text(self, text: Any) -> str:
        """
        負責執行 SearchEvidenceBuilder 中的 _clean_search_text 流程，依照 SearchEvidenceBuilder 的流程需求處理 _clean_search_text 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if text is None:
            return ""

        text = str(text)
        text = text.replace("\\n", "\n")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _split_search_sections(self, text: str) -> list[str]:
        """
        負責執行 SearchEvidenceBuilder 中的 _split_search_sections 流程，依照 SearchEvidenceBuilder 的流程需求處理 _split_search_sections 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not text:
            return []

        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        if blocks:
            return blocks

        return [line.strip() for line in text.splitlines() if line.strip()]

    def _extract_query_keywords(self, question: str) -> set[str]:
        """
        負責執行 SearchEvidenceBuilder 中的 _extract_query_keywords 流程，依照 SearchEvidenceBuilder 的流程需求處理 _extract_query_keywords 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 set[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        text = normalize_text(question).lower()
        text = re.sub(r"[^\w\s]", " ", text)

        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "of",
            "in",
            "on",
            "at",
            "for",
            "to",
            "and",
            "or",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "would",
        }

        tokens = [token for token in text.split() if len(token) > 1 and token not in stopwords]
        return set(tokens)

    def _looks_like_metadata_section(self, section: str) -> bool:
        """
        負責執行 SearchEvidenceBuilder 中的 _looks_like_metadata_section 流程，依照 SearchEvidenceBuilder 的流程需求處理 _looks_like_metadata_section 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            section: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lower = section.lower().strip()

        metadata_markers = [
            "source:",
            "url:",
            "http://",
            "https://",
        ]

        if any(marker in lower for marker in metadata_markers):
            return True

        if len(lower) < 20:
            return True

        return False

    def _score_search_section(self, section: str, question_keywords: set[str]) -> int:
        """
        負責執行 SearchEvidenceBuilder 中的 _score_search_section 流程，依照 SearchEvidenceBuilder 的流程需求處理 _score_search_section 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            section: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            question_keywords: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lower = normalize_text(section).lower()
        score = 0

        matched_keywords = [kw for kw in question_keywords if kw in lower]
        score += len(set(matched_keywords)) * 2

        if len(set(matched_keywords)) >= 2:
            score += 3

        if self._looks_like_metadata_section(section):
            score -= 2

        answer_markers = ["is", "means", "refers to", "capital", "founded", "released"]
        if any(marker in lower for marker in answer_markers):
            score += 1

        return score

    def _rank_search_sections(self, sections: list[str], question: str) -> list[str]:
        """
        負責執行 SearchEvidenceBuilder 中的 _rank_search_sections 流程，依照 SearchEvidenceBuilder 的流程需求處理 _rank_search_sections 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            sections: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        question_keywords = self._extract_query_keywords(question)

        scored_sections = []
        for idx, section in enumerate(sections):
            score = self._score_search_section(section, question_keywords)
            scored_sections.append((score, idx, section))

        scored_sections.sort(key=lambda entry: (-entry[0], entry[1]))
        return [section for _, _, section in scored_sections]
