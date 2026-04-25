"""搜尋工具 - HelloAgents 原生搜尋實現。"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests

from ..base import Tool, ToolParameter

try:  # 可選依賴，缺失時降級能力
    from markdownify import markdownify
except Exception:  # pragma: no cover - 可選依賴
    markdownify = None  # type: ignore

try:
    from ddgs import DDGS  # type: ignore
except Exception:  # pragma: no cover - 可選依賴
    DDGS = None  # type: ignore

try:
    from tavily import TavilyClient  # type: ignore
except Exception:  # pragma: no cover - 可選依賴
    TavilyClient = None  # type: ignore

try:
    from serpapi import GoogleSearch  # type: ignore
except Exception:  # pragma: no cover - 可選依賴
    GoogleSearch = None  # type: ignore

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
DEFAULT_MAX_RESULTS = 5
SUPPORTED_RETURN_MODES = {"text", "structured", "json", "dict"}
SUPPORTED_BACKENDS = {
    "hybrid",
    "advanced",
    "tavily",
    "serpapi",
    "duckduckgo",
    "searxng",
    "perplexity",
}

RERANK_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

HIGH_TRUST_DOMAINS = (
    "wikipedia.org",
    ".gov",
    ".edu",
    ".ac.uk",
    "nature.com",
    "science.org",
    "nih.gov",
)

LOW_TRUST_DOMAINS = (
    "quora.com",
    "youtube.com",
    "youtu.be",
    "pinterest.com",
    "reddit.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
)


def _limit_text(text: str, token_limit: int) -> str:
    char_limit = token_limit * CHARS_PER_TOKEN
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + "... [truncated]"


def _fetch_raw_content(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - 網路環境不穩定
        logger.debug("Failed to fetch raw content for %s: %s", url, exc)
        return None

    if markdownify is not None:
        try:
            return markdownify(response.text)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - 可選依賴失敗
            logger.debug("markdownify failed for %s: %s", url, exc)
    return response.text


def _normalized_result(
    *,
    title: str,
    url: str,
    content: str,
    raw_content: str | None,
) -> Dict[str, str]:
    payload: Dict[str, str] = {
        "title": title or url,
        "url": url,
        "content": content or "",
    }
    if raw_content is not None:
        payload["raw_content"] = raw_content
    return payload


def _structured_payload(
    results: Iterable[Dict[str, Any]],
    *,
    backend: str,
    answer: str | None = None,
    notices: Iterable[str] | None = None,
) -> Dict[str, Any]:
    return {
        "results": list(results),
        "backend": backend,
        "answer": answer,
        "notices": list(notices or []),
    }


class SearchTool(Tool):
    """支援多後端、可回傳結構化結果的搜尋工具。"""

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: str | None = None,
        serpapi_key: str | None = None,
        perplexity_key: str | None = None,
    ) -> None:
        super().__init__(
            name="search",
            description=(
                "智慧網頁搜尋引擎，支援 Tavily、SerpApi、DuckDuckGo、SearXNG、"
                "Perplexity 等後端，可回傳結構化或文字化的搜尋結果。"
            ),
        )
        self.backend = (backend or "hybrid").lower()
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.perplexity_key = perplexity_key or os.getenv("PERPLEXITY_API_KEY")

        self.available_backends: list[str] = []
        self.tavily_client = None
        self._setup_backends()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, parameters: Dict[str, Any]) -> str | Dict[str, Any]:  # type: ignore[override]
        query = (parameters.get("input") or parameters.get("query") or "").strip()
        if not query:
            return "錯誤：搜尋查詢不能為空"

        backend = str(parameters.get("backend", self.backend) or "hybrid").lower()
        backend = backend if backend in SUPPORTED_BACKENDS else "hybrid"

        mode = str(
            parameters.get("mode")
            or parameters.get("return_mode")
            or "text"
        ).lower()
        if mode not in SUPPORTED_RETURN_MODES:
            mode = "text"

        fetch_full_page = bool(parameters.get("fetch_full_page", False))
        conditional_fetch = bool(parameters.get("conditional_fetch", False))
        max_results = int(parameters.get("max_results", DEFAULT_MAX_RESULTS))
        max_tokens = int(parameters.get("max_tokens_per_source", 2000))
        max_full_page_results = int(parameters.get("max_full_page_results", 2))
        loop_count = int(parameters.get("loop_count", 0))

        payload = self._structured_search(
            query=query,
            backend=backend,
            fetch_full_page=fetch_full_page,
            conditional_fetch=conditional_fetch,
            max_results=max_results,
            max_tokens=max_tokens,
            max_full_page_results=max_full_page_results,
            loop_count=loop_count,
        )

        if mode in {"structured", "json", "dict"}:
            return payload

        return self._format_text_response(query=query, payload=payload)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="搜尋查詢關鍵詞",
                required=True,
            ),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup_backends(self) -> None:
        if self.tavily_key and TavilyClient is not None:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
                print("[OK] Tavily 搜尋引擎已初始化")
            except Exception as exc:  # pragma: no cover - 第三方庫初始化失敗
                print(f"[WARN] Tavily 初始化失敗: {exc}")
        elif self.tavily_key:
            print("[WARN] 未安裝 tavily-python，無法使用 Tavily 搜尋")
        else:
            print("[WARN] TAVILY_API_KEY 未設定")

        if self.serpapi_key:
            if GoogleSearch is not None:
                self.available_backends.append("serpapi")
                print("[OK] SerpApi 搜尋引擎已初始化")
            else:
                print("[WARN] 未安裝 google-search-results，無法使用 SerpApi 搜尋")
        else:
            print("[WARN] SERPAPI_API_KEY 未設定")

        if self.backend not in SUPPORTED_BACKENDS:
            print("[WARN] 不支援的搜尋後端，將使用 hybrid 模式")
            self.backend = "hybrid"
        elif self.backend == "tavily" and "tavily" not in self.available_backends:
            print("[WARN] Tavily 不可用，將使用 hybrid 模式")
            self.backend = "hybrid"
        elif self.backend == "serpapi" and "serpapi" not in self.available_backends:
            print("[WARN] SerpApi 不可用，將使用 hybrid 模式")
            self.backend = "hybrid"

        if self.backend == "hybrid":
            if self.available_backends:
                print(
                    "[INFO] 混合搜尋模式已啟用，可用後端: "
                    + ", ".join(self.available_backends)
                )
            else:
                print("[WARN] 沒有可用的 Tavily/SerpApi 搜尋源，將回退到通用模式")

    def _structured_search(
        self,
        *,
        query: str,
        backend: str,
        fetch_full_page: bool,
        conditional_fetch: bool,
        max_results: int,
        max_tokens: int,
        max_full_page_results: int,
        loop_count: int,
    ) -> Dict[str, Any]:
        # 統一將 hybrid 視作 advanced，以保持向後相容的優先順序邏輯
        target_backend = "advanced" if backend == "hybrid" else backend

        if target_backend == "tavily":
            payload = self._search_tavily(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
            return self._finalize_payload(
                query=query,
                payload=payload,
                max_results=max_results,
                conditional_fetch=conditional_fetch and not fetch_full_page,
                max_tokens=max_tokens,
                max_full_page_results=max_full_page_results,
            )
        if target_backend == "serpapi":
            payload = self._search_serpapi(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
            return self._finalize_payload(
                query=query,
                payload=payload,
                max_results=max_results,
                conditional_fetch=conditional_fetch and not fetch_full_page,
                max_tokens=max_tokens,
                max_full_page_results=max_full_page_results,
            )
        if target_backend == "duckduckgo":
            payload = self._search_duckduckgo(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
            return self._finalize_payload(
                query=query,
                payload=payload,
                max_results=max_results,
                conditional_fetch=conditional_fetch and not fetch_full_page,
                max_tokens=max_tokens,
                max_full_page_results=max_full_page_results,
            )
        if target_backend == "searxng":
            payload = self._search_searxng(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
            return self._finalize_payload(
                query=query,
                payload=payload,
                max_results=max_results,
                conditional_fetch=conditional_fetch and not fetch_full_page,
                max_tokens=max_tokens,
                max_full_page_results=max_full_page_results,
            )
        if target_backend == "perplexity":
            payload = self._search_perplexity(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
                loop_count=loop_count,
            )
            return self._finalize_payload(
                query=query,
                payload=payload,
                max_results=max_results,
                conditional_fetch=conditional_fetch and not fetch_full_page,
                max_tokens=max_tokens,
                max_full_page_results=max_full_page_results,
            )
        if target_backend == "advanced":
            payload = self._search_advanced(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
                loop_count=loop_count,
            )
            return self._finalize_payload(
                query=query,
                payload=payload,
                max_results=max_results,
                conditional_fetch=conditional_fetch and not fetch_full_page,
                max_tokens=max_tokens,
                max_full_page_results=max_full_page_results,
            )

        raise ValueError(f"Unsupported search backend: {backend}")

    def _finalize_payload(
        self,
        *,
        query: str,
        payload: Dict[str, Any],
        max_results: int,
        conditional_fetch: bool,
        max_tokens: int,
        max_full_page_results: int,
    ) -> Dict[str, Any]:
        finalized = dict(payload)
        reranked_results = self._rerank_results(
            query=query,
            results=payload.get("results") or [],
            max_results=max_results,
        )
        notices = list(finalized.get("notices") or [])

        if conditional_fetch and self._should_conditional_fetch(query=query, results=reranked_results):
            reranked_results, fetch_count = self._conditionally_fetch_full_pages(
                query=query,
                results=reranked_results,
                max_tokens=max_tokens,
                max_full_page_results=max_full_page_results,
            )
            if fetch_count:
                notices.append(f"[INFO] Conditional full-page fetch applied to top {fetch_count} results")

        finalized["results"] = reranked_results
        finalized["notices"] = notices
        return finalized

    def _should_conditional_fetch(self, *, query: str, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return False

        query_lower = query.lower()
        precision_markers = [
            "wikipedia",
            "official",
            "latest",
            "minimum",
            "maximum",
            "nearest",
            "closest",
            "between",
            "included",
            "excluded",
            "according to",
            "site:",
        ]
        if any(marker in query_lower for marker in precision_markers):
            return True

        if re.search(r"\b(?:19|20)\d{2}\b", query_lower):
            return True

        top_results = results[:2]
        return any(len(str(item.get("content", "") or "").strip()) < 140 for item in top_results)

    def _conditionally_fetch_full_pages(
        self,
        *,
        query: str,
        results: List[Dict[str, Any]],
        max_tokens: int,
        max_full_page_results: int,
    ) -> tuple[List[Dict[str, Any]], int]:
        enriched_results: list[Dict[str, Any]] = []
        fetched_count = 0
        for index, item in enumerate(results):
            enriched = dict(item)
            should_fetch = index < max_full_page_results
            domain = self._extract_domain(str(enriched.get("url", "") or ""))
            existing_raw = str(enriched.get("raw_content", "") or "").strip()

            if (
                should_fetch
                and not existing_raw
                and domain
                and not any(marker in domain for marker in LOW_TRUST_DOMAINS)
            ):
                fetched = _fetch_raw_content(str(enriched.get("url", "") or ""))
                if fetched:
                    enriched["raw_content"] = _limit_text(fetched, max_tokens)
                    enriched["conditional_fetch_used"] = True
                    fetched_count += 1

            enriched_results.append(enriched)

        return enriched_results, fetched_count

    def _rerank_results(
        self,
        *,
        query: str,
        results: List[Dict[str, Any]],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        query_terms = self._extract_query_terms(query)
        year_terms = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        query_lower = query.lower()
        seen_urls: set[str] = set()
        domain_counts: dict[str, int] = {}
        scored: list[tuple[float, int, Dict[str, Any]]] = []

        for original_index, item in enumerate(results):
            url = str(item.get("url", "") or "").strip()
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)

            title = str(item.get("title", "") or "")
            content = str(item.get("content", "") or "")
            raw_content = str(item.get("raw_content", "") or "")
            combined = " ".join(part for part in [title, content, raw_content[:2000]] if part).lower()
            title_lower = title.lower()
            domain = self._extract_domain(url)

            title_overlap = sum(1 for term in query_terms if term in title_lower)
            content_overlap = sum(1 for term in query_terms if term in combined)

            score = 0.0
            score += title_overlap * 2.5
            score += content_overlap * 1.2

            if title_overlap >= 2:
                score += 1.5
            if content_overlap >= 3:
                score += 1.0

            score += self._score_domain(domain, query_lower)

            if year_terms:
                matched_years = sum(1 for year in year_terms if year in combined)
                score += matched_years * 1.2

            if len(content.strip()) < 40 and not raw_content.strip():
                score -= 0.5

            domain_penalty = domain_counts.get(domain, 0) * 0.35 if domain else 0.0
            score -= domain_penalty
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            score -= original_index * 0.05

            reranked = dict(item)
            reranked["rerank_score"] = round(score, 3)
            scored.append((score, original_index, reranked))

        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return [item for _, _, item in scored[:max_results]]

    def _score_domain(self, domain: str, query_lower: str) -> float:
        if not domain:
            return 0.0

        score = 0.0
        if any(domain.endswith(marker) or marker in domain for marker in HIGH_TRUST_DOMAINS):
            score += 1.5
        if any(marker in domain for marker in LOW_TRUST_DOMAINS):
            score -= 1.0

        if "site:en.wikipedia.org" in query_lower and "wikipedia.org" in domain:
            score += 3.0
        elif "wikipedia" in query_lower and "wikipedia.org" in domain:
            score += 2.0

        if "official" in query_lower and any(marker in domain for marker in [".gov", ".edu", ".org"]):
            score += 1.0

        return score

    def _extract_domain(self, url: str) -> str:
        if not url:
            return ""
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    def _extract_query_terms(self, query: str) -> List[str]:
        normalized = re.sub(r"site:[^\s]+", " ", query.lower())
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        terms: list[str] = []
        for token in normalized.split():
            if token in RERANK_STOPWORDS or len(token) <= 2:
                continue
            if token not in terms:
                terms.append(token)
        return terms[:12]

    def _search_tavily(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        if not self.tavily_client:
            message = "TAVILY_API_KEY 未設定或 tavily 未安裝"
            raise RuntimeError(message)

        response = self.tavily_client.search(  # type: ignore[call-arg]
            query=query,
            max_results=max_results,
            include_raw_content=fetch_full_page,
        )

        results = []
        for item in response.get("results", [])[:max_results]:
            raw = item.get("raw_content") if fetch_full_page else item.get("content")
            if raw and fetch_full_page:
                raw = _limit_text(raw, max_tokens)
            results.append(
                _normalized_result(
                    title=item.get("title") or item.get("url", ""),
                    url=item.get("url", ""),
                    content=item.get("content") or "",
                    raw_content=raw,
                )
            )

        return _structured_payload(
            results,
            backend="tavily",
            answer=response.get("answer"),
        )

    def _search_serpapi(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        if not self.serpapi_key:
            raise RuntimeError("SERPAPI_API_KEY 未設定，無法使用 SerpApi 搜尋")
        if GoogleSearch is None:
            raise RuntimeError("未安裝 google-search-results，無法使用 SerpApi")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "gl": "us",
            "hl": "en",
            "num": max_results,
        }

        response = GoogleSearch(params).get_dict()

        answer_box = response.get("answer_box") or {}
        answer = answer_box.get("answer") or answer_box.get("snippet")

        results = []
        for item in response.get("organic_results", [])[:max_results]:
            raw_content = item.get("snippet")
            if raw_content and fetch_full_page:
                raw_content = _limit_text(raw_content, max_tokens)
            results.append(
                _normalized_result(
                    title=item.get("title") or item.get("link", ""),
                    url=item.get("link", ""),
                    content=item.get("snippet") or "",
                    raw_content=raw_content,
                )
            )

        return _structured_payload(results, backend="serpapi", answer=answer)

    def _search_duckduckgo(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        if DDGS is None:
            raise RuntimeError("未安裝 ddgs，無法使用 DuckDuckGo 搜尋")

        results: List[Dict[str, Any]] = []
        notices: List[str] = []

        try:
            with DDGS(timeout=10) as client:  # type: ignore[call-arg]
                search_results = client.text(query, max_results=max_results, backend="duckduckgo")
        except Exception as exc:  # pragma: no cover - 網路異常
            raise RuntimeError(f"DuckDuckGo 搜尋失敗: {exc}")

        for entry in search_results:
            url = entry.get("href") or entry.get("url")
            title = entry.get("title") or url or ""
            content = entry.get("body") or entry.get("content") or ""

            if not url or not title:
                notices.append(f"忽略不完整的 DuckDuckGo 結果: {entry}")
                continue

            raw_content = content
            if fetch_full_page and url:
                fetched = _fetch_raw_content(url)
                if fetched:
                    raw_content = _limit_text(fetched, max_tokens)

            results.append(
                _normalized_result(
                    title=title,
                    url=url,
                    content=content,
                    raw_content=raw_content,
                )
            )

        return _structured_payload(results, backend="duckduckgo", notices=notices)

    def _search_searxng(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
    ) -> Dict[str, Any]:
        host = os.getenv("SEARXNG_URL", "http://localhost:8888").rstrip("/")
        endpoint = f"{host}/search"

        try:
            response = requests.get(
                endpoint,
                params={
                    "q": query,
                    "format": "json",
                    "language": "en",
                    "safesearch": 1,
                    "categories": "general",
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - 網路異常
            raise RuntimeError(f"SearXNG 搜尋失敗: {exc}")

        results = []
        for entry in payload.get("results", [])[:max_results]:
            url = entry.get("url") or entry.get("link")
            title = entry.get("title") or url or ""
            if not url or not title:
                continue
            content = entry.get("content") or entry.get("snippet") or ""
            raw_content = content
            if fetch_full_page and url:
                fetched = _fetch_raw_content(url)
                if fetched:
                    raw_content = _limit_text(fetched, max_tokens)
            results.append(
                _normalized_result(
                    title=title,
                    url=url,
                    content=content,
                    raw_content=raw_content,
                )
            )

        return _structured_payload(results, backend="searxng")

    def _search_perplexity(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
        loop_count: int,
    ) -> Dict[str, Any]:
        if not self.perplexity_key:
            raise RuntimeError("PERPLEXITY_API_KEY 未設定，無法使用 Perplexity 搜尋")

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.perplexity_key}",
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources.",
                },
                {"role": "user", "content": query},
            ],
        }

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", []) or ["https://perplexity.ai"]

        results = []
        for idx, url in enumerate(citations[:max_results], start=1):
            snippet = content if idx == 1 else "See main Perplexity response above."
            raw = _limit_text(content, max_tokens) if fetch_full_page and idx == 1 else None
            results.append(
                _normalized_result(
                    title=f"Perplexity Source {loop_count + 1}-{idx}",
                    url=url,
                    content=snippet,
                    raw_content=raw,
                )
            )

        return _structured_payload(results, backend="perplexity", answer=content)

    def _search_advanced(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
        loop_count: int,
    ) -> Dict[str, Any]:
        notices: List[str] = []
        aggregated: List[Dict[str, Any]] = []
        answer: str | None = None
        backend_used = "advanced"

        if self.tavily_client:
            try:
                tavily_payload = self._search_tavily(
                    query=query,
                    fetch_full_page=fetch_full_page,
                    max_results=max_results,
                    max_tokens=max_tokens,
                )
                if tavily_payload["results"]:
                    return tavily_payload
                notices.append("[WARN] Tavily 未回傳有效結果，嘗試其他搜尋源")
            except Exception as exc:  # pragma: no cover - 第三方庫異常
                notices.append(f"[WARN] Tavily 搜尋失敗：{exc}")

        if self.serpapi_key and GoogleSearch is not None:
            try:
                serp_payload = self._search_serpapi(
                    query=query,
                    fetch_full_page=fetch_full_page,
                    max_results=max_results,
                    max_tokens=max_tokens,
                )
                if serp_payload["results"]:
                    serp_payload["notices"] = notices + serp_payload.get("notices", [])
                    return serp_payload
                notices.append("[WARN] SerpApi 未回傳有效結果，回退到通用搜尋")
            except Exception as exc:  # pragma: no cover - 第三方庫異常
                notices.append(f"[WARN] SerpApi 搜尋失敗：{exc}")

        try:
            ddg_payload = self._search_duckduckgo(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
            aggregated.extend(ddg_payload["results"])
            notices.extend(ddg_payload.get("notices", []))
            backend_used = ddg_payload.get("backend", backend_used)
        except Exception as exc:  # pragma: no cover - 通用兜底
            notices.append(f"[WARN] DuckDuckGo 搜尋失敗：{exc}")

        return _structured_payload(
            aggregated,
            backend=backend_used,
            answer=answer,
            notices=notices,
        )

    def _format_text_response(self, *, query: str, payload: Dict[str, Any]) -> str:
        answer = payload.get("answer")
        notices = payload.get("notices") or []
        results = payload.get("results") or []
        backend = payload.get("backend", self.backend)

        lines = [f"[INFO] 搜尋關鍵詞：{query}", f"[INFO] 使用搜尋源：{backend}"]
        if answer:
            lines.append(f"[INFO] 直接答案：{answer}")

        if results:
            lines.append("")
            lines.append("[INFO] 參考來源：")
            for idx, item in enumerate(results, start=1):
                title = item.get("title") or item.get("url", "")
                lines.append(f"[{idx}] {title}")
                if item.get("content"):
                    lines.append(f"    {item['content']}")
                if item.get("url"):
                    lines.append(f"    來源: {item['url']}")
                lines.append("")
        else:
            lines.append("[ERROR] 找不到相關搜尋結果。")

        if notices:
            lines.append("[WARN] 注意事項：")
            for notice in notices:
                if notice:
                    lines.append(f"- {notice}")

        return "\n".join(line for line in lines if line is not None)


# 便捷函式

def search(query: str, backend: str = "hybrid") -> str:
    tool = SearchTool(backend=backend)
    return tool.run({"input": query, "backend": backend})  # type: ignore[return-value]


def search_tavily(query: str) -> str:
    tool = SearchTool(backend="tavily")
    return tool.run({"input": query, "backend": "tavily"})  # type: ignore[return-value]


def search_serpapi(query: str) -> str:
    tool = SearchTool(backend="serpapi")
    return tool.run({"input": query, "backend": "serpapi"})  # type: ignore[return-value]


def search_hybrid(query: str) -> str:
    tool = SearchTool(backend="hybrid")
    return tool.run({"input": query, "backend": "hybrid"})  # type: ignore[return-value]
