from __future__ import annotations

import re
from typing import Any

from .json_parse import try_parse_json


class BaseParser:
    def __init__(self, parse_json=try_parse_json):
        self.parse_json = parse_json

    def _extract_first_match(
        self,
        text: str,
        patterns: list[str],
        flags: int = re.IGNORECASE | re.DOTALL,
    ) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text or "", flags)
            if match:
                return match.group(1).strip()
        return None

    def _split_nonempty_lines(self, text: str) -> list[str]:
        return [line.strip() for line in (text or "").splitlines() if line.strip()]

    def _looks_like_short_answer(
        self,
        text: str,
        max_chars: int = 40,
    ) -> bool:
        candidate = (text or "").strip()
        if not candidate:
            return False
        if len(candidate) > max_chars:
            return False
        if re.search(r"[.!?]", candidate):
            return False
        return True

    def _is_pure_symbol_fragment(self, text: str) -> bool:
        candidate = (text or "").strip()
        if not candidate:
            return True
        return re.fullmatch(r"[\W_]+", candidate) is not None

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)


class AgentReplyParser(BaseParser):
    def parse(
        self,
        reply: str,
        expected_weight_count: int | None,
        require_final_answer: bool = True,
    ) -> dict[str, Any]:
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
        if "final_answer" in parsed_json:
            return parsed_json.get("final_answer")

        for key in ["correct_answer", "answer", "final", "result", "output"]:
            if key in parsed_json:
                return parsed_json.get(key)

        return None

    def _normalize_reasoning(self, reasoning: Any) -> str:
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
