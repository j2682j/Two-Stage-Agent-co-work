from __future__ import annotations

import re
from typing import Any

from utils.network_utils import normalize_text


class SearchQueryPlanner:
    """
    負責在 builder.search_query_planner 中封裝 SearchQueryPlanner，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    SEARCH_STOPWORDS = {
        "a",
        "an",
        "and",
        "answer",
        "as",
        "at",
        "between",
        "by",
        "can",
        "could",
        "distance",
        "do",
        "find",
        "for",
        "from",
        "give",
        "how",
        "i",
        "if",
        "in",
        "included",
        "is",
        "it",
        "latest",
        "many",
        "me",
        "must",
        "my",
        "nearest",
        "not",
        "of",
        "on",
        "or",
        "page",
        "please",
        "provide",
        "result",
        "round",
        "should",
        "take",
        "that",
        "the",
        "their",
        "them",
        "this",
        "to",
        "use",
        "using",
        "version",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
        "would",
        "you",
        "your",
    }

    SOURCE_MARKERS = {
        "wikipedia": "wikipedia",
        "official": "official",
        "latest": "latest",
        "english wikipedia": "english_wikipedia",
        "2022 version": "versioned_source",
    }

    INSTRUCTION_MARKERS = [
        "please use",
        "please provide your answer",
        "provide your answer",
        "round your",
        "do not use",
        "if necessary",
        "answer as",
        "you can use",
        "use the latest",
    ]

    LEADING_ENTITY_STOPWORDS = {
        "A",
        "An",
        "How",
        "If",
        "Please",
        "The",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Why",
        "You",
    }

    def plan(self, question: str, max_queries: int = 3) -> dict[str, Any]:
        """
        負責執行 SearchQueryPlanner 中的 plan 流程，依照 SearchQueryPlanner 的流程需求處理 plan 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            max_queries: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        text = normalize_text(question)
        if not text:
            return {
                "queries": [],
                "core_query": "",
                "keyword_query": "",
                "source_query": "",
                "source_hints": [],
                "year_tokens": [],
                "precision_needed": False,
            }

        source_hints = self._detect_source_hints(text)
        year_tokens = self._extract_year_tokens(text)
        core_query = self._build_core_query(text)
        keyword_query = self._build_keyword_query(
            text,
            core_query=core_query,
            year_tokens=year_tokens,
        )
        source_query = self._build_source_query(
            text,
            core_query=core_query,
            keyword_query=keyword_query,
            source_hints=source_hints,
            year_tokens=year_tokens,
        )

        ordered_candidates = [core_query, keyword_query, source_query]
        queries: list[str] = []
        seen: set[str] = set()
        for candidate in ordered_candidates:
            normalized = self._normalize_query_key(candidate)
            if not normalized or normalized in seen:
                continue
            queries.append(candidate.strip())
            seen.add(normalized)
            if len(queries) >= max(1, max_queries):
                break

        precision_needed = bool(source_hints or year_tokens or re.search(r"\d", text))

        return {
            "queries": queries,
            "core_query": core_query,
            "keyword_query": keyword_query,
            "source_query": source_query,
            "source_hints": source_hints,
            "year_tokens": year_tokens,
            "precision_needed": precision_needed,
        }

    def _build_core_query(self, question: str) -> str:
        """
        負責執行 SearchQueryPlanner 中的 _build_core_query 流程，依照 SearchQueryPlanner 的流程需求處理 _build_core_query 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        clauses = self._split_into_clauses(question)
        kept_clauses = []
        for clause in clauses:
            lowered = clause.lower()
            trimmed = clause.strip()
            for marker in self.INSTRUCTION_MARKERS:
                marker_index = lowered.find(marker)
                if marker_index != -1:
                    trimmed = trimmed[:marker_index].rstrip(" ,;:-")
                    break
            if trimmed:
                kept_clauses.append(trimmed)

        if not kept_clauses:
            return question.strip()

        core = " ".join(kept_clauses).strip()
        core = re.sub(r"\s+", " ", core)
        return core

    def _build_keyword_query(self, question: str, *, core_query: str, year_tokens: list[str]) -> str:
        """
        負責執行 SearchQueryPlanner 中的 _build_keyword_query 流程，依照 SearchQueryPlanner 的流程需求處理 _build_keyword_query 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            core_query: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            year_tokens: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        entities = self._extract_entity_phrases(question)
        tokens = self._extract_focus_tokens(core_query)

        parts: list[str] = []
        part_tokens: set[str] = set()
        for entity in entities[:2]:
            parts.append(entity)
            part_tokens.update(entity.lower().split())

        for token in year_tokens:
            if token not in parts:
                parts.append(token)
                part_tokens.update(token.lower().split())

        for token in tokens:
            lowered = token.lower()
            if lowered not in part_tokens:
                parts.append(token)
                part_tokens.add(lowered)
            if len(parts) >= 8:
                break

        if parts:
            return " ".join(parts)

        return core_query

    def _build_source_query(
        self,
        question: str,
        *,
        core_query: str,
        keyword_query: str,
        source_hints: list[str],
        year_tokens: list[str],
    ) -> str:
        """
        負責執行 SearchQueryPlanner 中的 _build_source_query 流程，依照 SearchQueryPlanner 的流程需求處理 _build_source_query 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            core_query: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            keyword_query: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            source_hints: 已整理好的搜尋結果、共享資料包或可重用證據內容。
            year_tokens: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        base = keyword_query or core_query
        if not base:
            return ""

        lowered = question.lower()
        if "english wikipedia" in lowered or "wikipedia" in lowered:
            extra_years = [token for token in year_tokens if token not in base]
            query = f"{base} site:en.wikipedia.org"
            if extra_years:
                query = f"{query} {' '.join(extra_years)}"
            return query

        if "official" in lowered:
            return f"{base} official"

        if source_hints:
            return f"{base} {' '.join(source_hints)}".strip()

        return ""

    def _split_into_clauses(self, text: str) -> list[str]:
        """
        負責執行 SearchQueryPlanner 中的 _split_into_clauses 流程，依照 SearchQueryPlanner 的流程需求處理 _split_into_clauses 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raw = re.split(r"(?<=[.?!])\s+|\n+", text)
        clauses = [segment.strip() for segment in raw if segment.strip()]
        return clauses or [text.strip()]

    def _detect_source_hints(self, text: str) -> list[str]:
        """
        負責執行 SearchQueryPlanner 中的 _detect_source_hints 流程，依照 SearchQueryPlanner 的流程需求處理 _detect_source_hints 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lowered = text.lower()
        hints = []
        for marker, hint in self.SOURCE_MARKERS.items():
            if marker in lowered and hint not in hints:
                hints.append(hint)
        return hints

    def _extract_year_tokens(self, text: str) -> list[str]:
        """
        負責執行 SearchQueryPlanner 中的 _extract_year_tokens 流程，依照 SearchQueryPlanner 的流程需求處理 _extract_year_tokens 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        tokens: list[str] = []
        consumed_years: set[str] = set()

        between_match = re.search(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", text, flags=re.IGNORECASE)
        if between_match:
            tokens.append(f"{between_match.group(1)} {between_match.group(2)}")
            consumed_years.update([between_match.group(1), between_match.group(2)])

        from_match = re.search(r"\bfrom\s+(\d{4})\s+to\s+(\d{4})\b", text, flags=re.IGNORECASE)
        if from_match:
            range_token = f"{from_match.group(1)} {from_match.group(2)}"
            if range_token not in tokens:
                tokens.append(range_token)
            consumed_years.update([from_match.group(1), from_match.group(2)])

        for year in re.findall(r"\b(?:19|20)\d{2}\b", text):
            if year in consumed_years:
                continue
            if year not in tokens:
                tokens.append(year)

        return tokens[:3]

    def _extract_entity_phrases(self, text: str) -> list[str]:
        """
        負責執行 SearchQueryPlanner 中的 _extract_entity_phrases 流程，依照 SearchQueryPlanner 的流程需求處理 _extract_entity_phrases 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        phrases: list[str] = []

        for quote in re.findall(r'"([^"]+)"|\'([^\']+)\'', text):
            candidate = next((part for part in quote if part), "").strip()
            if candidate and candidate not in phrases:
                phrases.append(candidate)

        capitalized_pattern = re.compile(
            r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|\d{4}))"
            r"{0,4}\b"
        )
        for match in capitalized_pattern.findall(text):
            candidate = self._trim_leading_entity_stopwords(match.strip())
            if candidate.lower() in self.SEARCH_STOPWORDS:
                continue
            if candidate not in phrases:
                phrases.append(candidate)

        return phrases[:4]

    def _extract_focus_tokens(self, text: str) -> list[str]:
        """
        負責執行 SearchQueryPlanner 中的 _extract_focus_tokens 流程，依照 SearchQueryPlanner 的流程需求處理 _extract_focus_tokens 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 已整理好的搜尋結果、共享資料包或可重用證據內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lowered = normalize_text(text).lower()
        lowered = re.sub(r"[^\w\s]", " ", lowered)
        tokens = []
        for token in lowered.split():
            if token in self.SEARCH_STOPWORDS:
                continue
            if len(token) <= 2:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens[:8]

    def _normalize_query_key(self, query: str) -> str:
        """
        負責執行 SearchQueryPlanner 中的 _normalize_query_key 流程，依照 SearchQueryPlanner 的流程需求處理 _normalize_query_key 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            query: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return re.sub(r"\s+", " ", normalize_text(query).lower()).strip()

    def _trim_leading_entity_stopwords(self, candidate: str) -> str:
        """
        負責執行 SearchQueryPlanner 中的 _trim_leading_entity_stopwords 流程，依照 SearchQueryPlanner 的流程需求處理 _trim_leading_entity_stopwords 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            candidate: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        tokens = candidate.split()
        while tokens and tokens[0] in self.LEADING_ENTITY_STOPWORDS:
            tokens = tokens[1:]
        return " ".join(tokens).strip()
