from __future__ import annotations

import re
from typing import Any

from .json_parse import try_parse_json


class BaseParser:
    """
    負責在 parser.base_parser 中封裝 BaseParser，封裝模型輸出解析流程，將文字結果轉成結構化資料。
    
    Args:
        parse_json: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, parse_json=try_parse_json):
        """
        負責執行 BaseParser 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            parse_json: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.parse_json = parse_json

    def _extract_first_match(
        self,
        text: str,
        patterns: list[str],
        flags: int = re.IGNORECASE | re.DOTALL,
    ) -> str | None:
        """
        負責執行 BaseParser 中的 _extract_first_match 流程，依照 BaseParser 的流程需求處理 _extract_first_match 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
            patterns: 此流程需要使用的輸入資料。
            flags: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for pattern in patterns:
            match = re.search(pattern, text or "", flags)
            if match:
                return match.group(1).strip()
        return None

    def _split_nonempty_lines(self, text: str) -> list[str]:
        """
        負責執行 BaseParser 中的 _split_nonempty_lines 流程，依照 BaseParser 的流程需求處理 _split_nonempty_lines 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [line.strip() for line in (text or "").splitlines() if line.strip()]

    def _looks_like_short_answer(
        self,
        text: str,
        max_chars: int = 40,
    ) -> bool:
        """
        負責執行 BaseParser 中的 _looks_like_short_answer 流程，依照 BaseParser 的流程需求處理 _looks_like_short_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
            max_chars: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        candidate = (text or "").strip()
        if not candidate:
            return False
        if len(candidate) > max_chars:
            return False
        if re.search(r"[.!?]", candidate):
            return False
        return True

    def _is_pure_symbol_fragment(self, text: str) -> bool:
        """
        負責執行 BaseParser 中的 _is_pure_symbol_fragment 流程，依照 BaseParser 的流程需求處理 _is_pure_symbol_fragment 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        candidate = (text or "").strip()
        if not candidate:
            return True
        return re.fullmatch(r"[\W_]+", candidate) is not None

    def _stringify(self, value: Any) -> str:
        """
        負責執行 BaseParser 中的 _stringify 流程，依照 BaseParser 的流程需求處理 _stringify 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            value: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)


class AgentReplyParser(BaseParser):
    """
    負責在 parser.base_parser 中封裝 AgentReplyParser，封裝代理節點的推理、工具使用、訊息傳遞或協作控制邏輯。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def parse(
        self,
        reply: str,
        expected_weight_count: int | None,
        require_final_answer: bool = True,
    ) -> dict[str, Any]:
        """
        負責執行 AgentReplyParser 中的 parse 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
            expected_weight_count: 此流程需要使用的輸入資料。
            require_final_answer: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        parsed_json = self.parse_json(reply)

        if parsed_json is None:
            parsed_json = self._fallback_parse_from_raw(reply)
            if parsed_json is None:
                raise ValueError("Failed to parse structured JSON from SLM reply.")

        reasoning = self._normalize_reasoning(parsed_json.get("reasoning", ""))
        final_answer = self._normalize_final_answer(
            self._extract_final_answer_candidate(parsed_json),
            parsed_json,
            require_final_answer=require_final_answer,
        )
        weights = self._normalize_weights(
            parsed_json.get("weights"),
            expected_weight_count,
            parsed_json,
        )

        return {
            "reasoning": reasoning.strip(),
            "final_answer": final_answer.strip(),
            "weights": weights,
        }

    def _fallback_parse_from_raw(self, reply: str) -> dict[str, Any] | None:
        """
        負責執行 AgentReplyParser 中的 _fallback_parse_from_raw 流程，依照 AgentReplyParser 的流程需求處理 _fallback_parse_from_raw 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any] | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not reply:
            return None

        final_answer = self._extract_field_from_raw(
            reply,
            ["final_answer", "correct_answer", "answer", "final", "result", "output"],
        )
        if final_answer is None:
            return None

        reasoning = self._extract_field_from_raw(reply, ["reasoning"]) or ""
        weights = self._extract_weights_from_raw(reply)
        return {
            "reasoning": reasoning,
            "final_answer": final_answer,
            "weights": weights,
        }

    def _extract_field_from_raw(self, reply: str, field_names: list[str]) -> str | None:
        """
        負責執行 AgentReplyParser 中的 _extract_field_from_raw 流程，依照 AgentReplyParser 的流程需求處理 _extract_field_from_raw 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
            field_names: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for field_name in field_names:
            patterns = [
                rf'"{field_name}"\s*:\s*"([^"]*)"',
                rf'"{field_name}"\s*:\s*([^\n,}}]+)',
            ]
            candidate = self._extract_first_match(reply, patterns, flags=re.IGNORECASE)
            if candidate is not None:
                return candidate.strip().strip('"').strip()
        return None

    def _extract_weights_from_raw(self, reply: str) -> list[Any] | None:
        """
        負責執行 AgentReplyParser 中的 _extract_weights_from_raw 流程，依照 AgentReplyParser 的流程需求處理 _extract_weights_from_raw 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[Any] | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        match = re.search(r'"weights"\s*:\s*\[([^\]]*)\]', reply or "", re.IGNORECASE | re.DOTALL)
        if not match:
            return None

        body = match.group(1).strip()
        if not body:
            return []

        items = []
        for part in body.split(","):
            value = part.strip().strip('"').strip()
            if value:
                items.append(value)
        return items

    def _extract_final_answer_candidate(self, parsed_json: dict[str, Any]) -> Any:
        """
        負責執行 AgentReplyParser 中的 _extract_final_answer_candidate 流程，依照 AgentReplyParser 的流程需求處理 _extract_final_answer_candidate 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            parsed_json: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Any。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if "final_answer" in parsed_json:
            return parsed_json.get("final_answer")

        for key in ["correct_answer", "answer", "final", "result", "output"]:
            if key in parsed_json:
                return parsed_json.get(key)

        return None

    def _normalize_reasoning(self, reasoning: Any) -> str:
        """
        負責執行 AgentReplyParser 中的 _normalize_reasoning 流程，依照 AgentReplyParser 的流程需求處理 _normalize_reasoning 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            reasoning: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if reasoning is None:
            return ""
        if not isinstance(reasoning, str):
            return str(reasoning)
        return reasoning

    def _normalize_final_answer(
        self,
        final_answer: Any,
        parsed_json: dict[str, Any],
        require_final_answer: bool = True,
    ) -> str:
        """
        負責執行 AgentReplyParser 中的 _normalize_final_answer 流程，依照 AgentReplyParser 的流程需求處理 _normalize_final_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            final_answer: 此流程需要使用的輸入資料。
            parsed_json: 此流程需要使用的輸入資料。
            require_final_answer: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if final_answer is None:
            if not require_final_answer:
                return ""
            raise ValueError(
                "Missing required field 'final_answer' in SLM reply JSON.\n"
                f"Parsed JSON: {parsed_json}"
            )

        if isinstance(final_answer, (int, float)):
            return str(final_answer)

        if not isinstance(final_answer, str):
            return str(final_answer)

        return final_answer

    def _normalize_weights(
        self,
        weights: Any,
        expected_weight_count: int | None,
        parsed_json: dict[str, Any],
    ) -> list[int]:
        """
        負責執行 AgentReplyParser 中的 _normalize_weights 流程，依照 AgentReplyParser 的流程需求處理 _normalize_weights 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            weights: 此流程需要使用的輸入資料。
            expected_weight_count: 此流程需要使用的輸入資料。
            parsed_json: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[int]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if expected_weight_count is None:
            return []

        if weights is None:
            weights = [] if expected_weight_count == 0 else [3] * expected_weight_count

        if not isinstance(weights, list):
            raise TypeError(
                f"'weights' must be list, got {type(weights).__name__}.\n"
                f"Parsed JSON: {parsed_json}"
            )

        normalized_weights = []
        for w in weights:
            if isinstance(w, bool):
                raise TypeError(
                    f"'weights' must contain integers, got bool.\n"
                    f"Parsed JSON: {parsed_json}"
                )

            if isinstance(w, (int, float, str)):
                try:
                    normalized_weights.append(int(w))
                except ValueError:
                    raise TypeError(
                        f"'weights' must contain integers, got {w!r}.\n"
                        f"Parsed JSON: {parsed_json}"
                    )
            else:
                raise TypeError(
                    f"'weights' must contain integers, got {type(w).__name__}.\n"
                    f"Parsed JSON: {parsed_json}"
                )

        if len(normalized_weights) != expected_weight_count:
            raise ValueError(
                f"Weight count mismatch: expected {expected_weight_count}, got {len(normalized_weights)}.\n"
                f"Weights: {normalized_weights}\n"
                f"Parsed JSON: {parsed_json}"
            )

        return normalized_weights
