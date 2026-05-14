from __future__ import annotations

import re
from typing import Any

from .base_parser import AgentReplyParser, BaseParser
from .json_parse import try_parse_json


class DecisionParser(BaseParser):
    """
    負責在 parser.decision_parser 中封裝 DecisionParser，封裝模型輸出解析流程，將文字結果轉成結構化資料。
    
    Args:
        parse_json: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, parse_json=try_parse_json):
        """
        負責執行 DecisionParser 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            parse_json: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(parse_json=parse_json)
        self.reply_parser = AgentReplyParser(parse_json=parse_json)

    def parse_critique(
        self,
        raw_reply: str,
        critic_agent_idx: int | None,
        fallback_answer: str = "",
    ) -> dict[str, Any]:
        """
        負責執行 DecisionParser 中的 parse_critique 流程，解析模型輸出並取出答案、決策、排序或 JSON 結構。
        
        Args:
            raw_reply: 此流程需要使用的輸入資料。
            critic_agent_idx: 此流程需要使用的輸入資料。
            fallback_answer: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        parsed = self.parse_json(raw_reply)
        if isinstance(parsed, dict):
            return {
                "critic_agent_idx": critic_agent_idx,
                "agree": bool(parsed.get("agree", False)),
                "critique": str(parsed.get("critique", "")).strip(),
                "revised_answer": str(
                    parsed.get("revised_answer", fallback_answer)
                ).strip(),
            }

        agree = self._extract_bool(raw_reply, "AGREE")
        critique = self._extract_value(raw_reply, "CRITIQUE")
        revised_answer = (
            self._extract_value(raw_reply, "REVISED_ANSWER") or fallback_answer
        )

        if agree is None and not critique and not revised_answer:
            raise ValueError("Failed to parse critique reply.")

        return {
            "critic_agent_idx": critic_agent_idx,
            "agree": bool(agree) if agree is not None else False,
            "critique": critique.strip(),
            "revised_answer": str(revised_answer).strip(),
        }

    def parse_solver_revision(self, raw_reply: str) -> dict[str, Any]:
        """
        負責執行 DecisionParser 中的 parse_solver_revision 流程，解析模型輸出並取出答案、決策、排序或 JSON 結構。
        
        Args:
            raw_reply: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            parsed = self.reply_parser.parse(raw_reply, expected_weight_count=None)
            return {
                "reasoning": parsed["reasoning"],
                "final_answer": parsed["final_answer"],
            }
        except Exception:
            reasoning = self._extract_reasoning(raw_reply)
            final_answer = self._extract_final_answer(raw_reply)
            if not final_answer:
                raise ValueError("Failed to parse solver revision reply.")
            return {
                "reasoning": reasoning,
                "final_answer": final_answer,
            }

    def _extract_value(self, text: str, key: str) -> str:
        """
        負責執行 DecisionParser 中的 _extract_value 流程，依照 DecisionParser 的流程需求處理 _extract_value 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
            key: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        match = re.search(rf"{re.escape(key)}\s*=\s*(.+)", text or "", re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_bool(self, text: str, key: str) -> bool | None:
        """
        負責執行 DecisionParser 中的 _extract_bool 流程，依照 DecisionParser 的流程需求處理 _extract_bool 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
            key: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        value = self._extract_value(text, key)
        if not value:
            return None
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        return None

    def _extract_reasoning(self, text: str) -> str:
        """
        負責執行 DecisionParser 中的 _extract_reasoning 流程，依照 DecisionParser 的流程需求處理 _extract_reasoning 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        value = self._extract_value(text, "REASONING")
        if value:
            return value
        lines = self._split_nonempty_lines(text)
        filtered = [
            line
            for line in lines
            if not re.match(
                r"(FINAL_ANSWER|FINAL ANSWER|final_answer)\s*[:=]",
                line,
                re.IGNORECASE,
            )
        ]
        return " ".join(filtered).strip()

    def _extract_final_answer(self, text: str) -> str:
        """
        負責執行 DecisionParser 中的 _extract_final_answer 流程，依照 DecisionParser 的流程需求處理 _extract_final_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        text = text or ""
        patterns = [
            r"FINAL_ANSWER\s*=\s*(.+)",
            r"FINAL ANSWER\s*:\s*(.+)",
            r"final_answer\s*[:=]\s*(.+)",
            r"\\boxed\{([^}]+)\}",
            r"FI+N?AL[_ ]?ANSW?E?R?\s*[:=]\s*(.+)",
        ]
        candidate = self._extract_first_match(text, patterns, flags=re.IGNORECASE)
        if candidate and self._is_valid_final_answer(candidate):
            return candidate

        lines = self._split_nonempty_lines(text)
        if lines:
            last_line = lines[-1]
            if self._looks_like_short_answer(last_line) and self._is_valid_final_answer(
                last_line
            ):
                return last_line
        return ""

    def _is_valid_final_answer(self, text: str) -> bool:
        """
        負責執行 DecisionParser 中的 _is_valid_final_answer 流程，依照 DecisionParser 的流程需求處理 _is_valid_final_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        candidate = (text or "").strip()
        if not candidate:
            return False
        if len(candidate) > 80:
            return False

        lowered = candidate.lower()
        invalid_literals = {
            "`",
            "```",
            "$",
            "$$",
            "{",
            "}",
            "[",
            "]",
            "(",
            ")",
            "final_answer",
            "final answer",
            "answer",
            "reasoning",
            "weights",
        }
        if lowered in invalid_literals:
            return False

        if self._is_pure_symbol_fragment(candidate):
            return False

        if re.search(r"(REASONING|WEIGHTS)\s*=", candidate, re.IGNORECASE):
            return False

        if re.search(r"^```[\w-]*$", candidate):
            return False

        return True
