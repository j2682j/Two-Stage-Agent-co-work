from __future__ import annotations

import re
from typing import Any

from .base_parser import BaseParser


class StageParser(BaseParser):
    """
    負責在 parser.stage_parser 中封裝 StageParser，封裝模型輸出解析流程，將文字結果轉成結構化資料。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def extract_reasoning(self, reply: str) -> str:
        """
        負責執行 StageParser 中的 extract_reasoning 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        patterns = [
            r"REASONING\s*=\s*(.*?)\s*FINAL_ANSWER\s*=",
            r'"reasoning"\s*:\s*"([^"]*)"',
            r"REASONING\s*[:=]\s*(.*?)(?:FINAL[_ ]ANSWER\s*[:=]|$)",
        ]
        candidate = self._extract_first_match(reply, patterns)
        if candidate:
            return candidate.strip()

        lines = self._split_nonempty_lines(reply)
        filtered = [
            line
            for line in lines
            if not re.match(r"FINAL_ANSWER\s*=", line, re.IGNORECASE)
            and not re.match(r"WEIGHTS\s*=", line, re.IGNORECASE)
            and not re.match(r"(FINAL_ANSWER|FINAL ANSWER|final_answer)\s*[:=]", line, re.IGNORECASE)
        ]
        if filtered and re.match(r"REASONING\s*=", filtered[0], re.IGNORECASE):
            filtered[0] = re.sub(
                r"^REASONING\s*=\s*",
                "",
                filtered[0],
                flags=re.IGNORECASE,
            ).strip()
        return " ".join(filtered).strip()

    def extract_final_answer(self, reply: str) -> str | None:
        """
        負責執行 StageParser 中的 extract_final_answer 流程，解析模型輸出並取出答案、決策、排序或 JSON 結構。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        patterns = [
            r"FINAL_ANSWER\s*=\s*(.+)",
            r'"final_answer"\s*:\s*"([^"]*)"',
            r'"final_answer"\s*:\s*([^\n,}]+)',
            r"FINAL ANSWER\s*:\s*(.+)",
            r"ANSWER\s*:\s*(.+)",
            r"final_answer\s*[:=]\s*(.+)",
            r"FINAL[_ ]ANSWER\s*[:=]\s*(.+)",
            r"FI+N?AL[_ ]?ANSW?E?R?\s*[:=]\s*(.+)",
            r"\\boxed\{([^{}]+)\}",
        ]
        candidate = self._extract_first_match(reply, patterns, flags=re.IGNORECASE)
        if candidate:
            normalized = candidate.strip().strip('"').strip()
            if self.is_valid_answer(normalized):
                return normalized

        return self._fallback_final_answer(reply)

    def is_valid_answer(self, text: Any) -> bool:
        """
        負責執行 StageParser 中的 is_valid_answer 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        candidate = self._stringify(text).strip()
        if not candidate:
            return False
        if len(candidate) > 80:
            return False
        if candidate in {"}", "{", "]", "[", ")", "(", "$", "$$", "```"}:
            return False
        if self._is_pure_symbol_fragment(candidate):
            return False
        if re.search(r"(REASONING|WEIGHTS)\s*=", candidate, re.IGNORECASE):
            return False
        if candidate.count("=") > 1:
            return False
        return True

    def _fallback_final_answer(self, reply: str) -> str | None:
        """
        負責執行 StageParser 中的 _fallback_final_answer 流程，依照 StageParser 的流程需求處理 _fallback_final_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lines = self._split_nonempty_lines(reply)
        if not lines:
            return None

        last_line = lines[-1]
        if re.match(r"WEIGHTS\s*=", last_line, re.IGNORECASE) and len(lines) >= 2:
            last_line = lines[-2]

        if "=" in last_line:
            key, value = last_line.split("=", 1)
            key_text = key.strip().upper().replace(" ", "_")
            if key_text in {
                "FINAL_ANSWER",
                "ANSWER",
                "FIINAL_ANSWE",
                "FINALANSWER",
            }:
                candidate = value.strip()
                if self.is_valid_answer(candidate):
                    return candidate

        if self._looks_like_short_answer(last_line) and self.is_valid_answer(last_line):
            return last_line
        return None


class Stage1ReplyParser(StageParser):
    """
    負責在 parser.stage_parser 中封裝 Stage1ReplyParser，封裝模型輸出解析流程，將文字結果轉成結構化資料。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def parse(self, reply: str, expected_weight_count: int) -> dict[str, Any]:
        """
        負責執行 Stage1ReplyParser 中的 parse 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
            expected_weight_count: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not reply:
            raise ValueError("Empty stage1 reply.")

        final_answer = self.extract_final_answer(reply)
        reasoning = self.extract_reasoning(reply)
        weights = self.extract_weights(reply, expected_weight_count)

        if final_answer is None or final_answer == "":
            raise ValueError("Missing FINAL ANSWER in stage1 reply.")

        return {
            "reasoning": reasoning.strip(),
            "final_answer": str(final_answer).strip(),
            "weights": weights,
        }

    def extract_weights(self, reply: str, expected_weight_count: int) -> list[int]:
        """
        負責執行 Stage1ReplyParser 中的 extract_weights 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
            expected_weight_count: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[int]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        lines = self._split_nonempty_lines(reply)
        if not lines:
            raise ValueError("Empty stage1 reply.")

        last_line = lines[-1]
        match = re.search(r"WEIGHTS\s*=\s*\[(.*)\]\s*$", last_line, re.IGNORECASE)
        if not match:
            return self.fallback_weights(expected_weight_count)

        body = match.group(1).strip()
        if not body:
            weights = []
        else:
            parts = [part.strip() for part in body.split(",")]
            weights = []
            for part in parts:
                value = float(part)
                if 0.0 <= value <= 1.0:
                    mapped = int(round(1 + value * 4))
                else:
                    mapped = int(round(value))
                mapped = max(1, min(5, mapped))
                weights.append(mapped)

        if expected_weight_count == 0:
            return []
        if len(weights) != expected_weight_count:
            return self.fallback_weights(expected_weight_count)
        return weights

    def fallback_weights(self, expected_weight_count: int) -> list[int]:
        """
        負責執行 Stage1ReplyParser 中的 fallback_weights 流程，依照 Stage1ReplyParser 的流程需求處理 fallback_weights 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            expected_weight_count: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[int]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if expected_weight_count <= 0:
            return []
        return [3] * expected_weight_count


class Stage2ReplyParser(StageParser):
    """
    負責在 parser.stage_parser 中封裝 Stage2ReplyParser，封裝模型輸出解析流程，將文字結果轉成結構化資料。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def parse_fallback(self, reply: str) -> dict[str, Any] | None:
        """
        負責執行 Stage2ReplyParser 中的 parse_fallback 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any] | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not reply:
            return None

        reasoning = self.extract_reasoning(reply)
        final_answer = self.extract_final_answer(reply)
        if final_answer is None or str(final_answer).strip() == "":
            return None

        return {
            "reasoning": reasoning.strip(),
            "final_answer": str(final_answer).strip(),
        }
