from __future__ import annotations

import re
from typing import Any

from .base_parser import AgentReplyParser, BaseParser
from .json_parse import try_parse_json


class DecisionParser(BaseParser):
    def __init__(self, parse_json=try_parse_json):
        super().__init__(parse_json=parse_json)
        self.reply_parser = AgentReplyParser(parse_json=parse_json)

    def parse_critique(
        self,
        raw_reply: str,
        critic_agent_idx: int | None,
        fallback_answer: str = "",
    ) -> dict[str, Any]:
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
        match = re.search(rf"{re.escape(key)}\s*=\s*(.+)", text or "", re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_bool(self, text: str, key: str) -> bool | None:
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
