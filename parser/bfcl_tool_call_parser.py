from __future__ import annotations

import ast
import json
import re
from typing import Any

from .base_parser import BaseParser


class BFCLToolCallParser(BaseParser):
    """
    負責在 parser.bfcl_tool_call_parser 中封裝 BFCLToolCallParser，解析 BFCL function calling 輸出並正規化為 tool call list。

    Args:
        parse_json: 可選的 JSON 解析函式，預設沿用 BaseParser 的解析流程。

    Returns:
        類別本身不直接回傳值；建立實例後可透過 parse 取得 BFCL tool call 清單。

    限制或副作用:
        只負責解析與格式正規化，不負責判斷 function call 是否符合 ground truth，也不會執行任何 function。
    """

    EMPTY_CALL_MARKERS = {
        "[]",
        "none",
        "null",
        "no_call",
        "no call",
        "no function call",
        "no tool call",
    }

    def parse(self, response: str) -> list[dict[str, Any]]:
        """
        負責執行 BFCLToolCallParser 中的 parse 流程，將模型回覆解析成 BFCL tool call list。

        Args:
            response: 模型產生的原始文字、JSON 或 function call 表示。

        Returns:
            正規化後的 tool call 清單，每筆包含 name 與 arguments。

        限制或副作用:
            解析失敗時回傳空清單；若需要錯誤原因，應改用 parse_with_metadata。
        """
        return self.parse_with_metadata(response)["calls"]

    def parse_with_metadata(self, response: str) -> dict[str, Any]:
        """
        負責執行 BFCLToolCallParser 中的 parse_with_metadata 流程，解析 tool call 並回傳解析來源與錯誤資訊。

        Args:
            response: 模型產生的原始文字、JSON 或 function call 表示。

        Returns:
            包含 calls、source、parse_error 與 raw_response 的解析結果字典。

        限制或副作用:
            不會丟出 JSONDecodeError；解析失敗會將錯誤訊息寫入 parse_error。
        """
        raw_response = response or ""
        text = raw_response.strip()
        if not text or self._is_empty_call(text):
            return self._result([], "empty", None, raw_response)

        candidates = self._json_candidates(text)
        parse_errors: list[str] = []
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                parse_errors.append(str(exc))
                continue
            calls = self._normalize_parsed_value(parsed)
            if calls is not None:
                return self._result(calls, "json", None, raw_response)

        python_calls = self._parse_python_call_syntax(text)
        if python_calls:
            return self._result(python_calls, "python_call_syntax", None, raw_response)

        error = "; ".join(parse_errors[-3:]) if parse_errors else "No BFCL tool call pattern found."
        return self._result([], "unparsed", error, raw_response)

    def _result(
        self,
        calls: list[dict[str, Any]],
        source: str,
        parse_error: str | None,
        raw_response: str,
    ) -> dict[str, Any]:
        """
        負責執行 BFCLToolCallParser 中的 _result 流程，組裝一致的解析結果資料。

        Args:
            calls: 已正規化的 tool call 清單。
            source: 解析成功或失敗時使用的來源類型。
            parse_error: 解析失敗原因；成功時為 None。
            raw_response: 模型原始回覆文字。

        Returns:
            包含 calls、source、parse_error 與 raw_response 的字典。

        限制或副作用:
            會再次正規化 calls，確保輸出欄位穩定。
        """
        return {
            "calls": self._normalize_calls(calls),
            "source": source,
            "parse_error": parse_error,
            "raw_response": raw_response,
        }

    def _is_empty_call(self, text: str) -> bool:
        """
        負責執行 BFCLToolCallParser 中的 _is_empty_call 流程，判斷模型輸出是否代表不呼叫任何 function。

        Args:
            text: 已去除前後空白的模型回覆文字。

        Returns:
            若文字代表空 tool call，回傳 True；否則回傳 False。

        限制或副作用:
            只比對常見空呼叫標記，不會理解長句語意。
        """
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return normalized in self.EMPTY_CALL_MARKERS

    def _json_candidates(self, text: str) -> list[str]:
        """
        負責執行 BFCLToolCallParser 中的 _json_candidates 流程，從模型回覆中擷取可能的 JSON 片段。

        Args:
            text: 模型原始回覆文字。

        Returns:
            可能可被 json.loads 解析的候選字串清單。

        限制或副作用:
            使用括號平衡掃描，不保證每個候選都是合法 JSON。
        """
        candidates: list[str] = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
            candidate = match.group(1).strip()
            if candidate:
                candidates.append(candidate)

        candidates.extend(self._balanced_json_fragments(text, "[", "]"))
        candidates.extend(self._balanced_json_fragments(text, "{", "}"))

        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            repaired = self._repair_json_candidate(candidate)
            if repaired and repaired not in seen:
                unique.append(repaired)
                seen.add(repaired)
        return unique

    def _balanced_json_fragments(self, text: str, open_char: str, close_char: str) -> list[str]:
        """
        負責執行 BFCLToolCallParser 中的 _balanced_json_fragments 流程，擷取平衡括號包住的 JSON 片段。

        Args:
            text: 模型原始回覆文字。
            open_char: 起始括號字元。
            close_char: 結束括號字元。

        Returns:
            可能的 JSON 片段清單。

        限制或副作用:
            掃描時會略過字串內括號，但不處理所有 JSON escape 邊界案例。
        """
        fragments: list[str] = []
        start: int | None = None
        depth = 0
        in_string = False
        escape = False

        for index, char in enumerate(text):
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == open_char:
                if depth == 0:
                    start = index
                depth += 1
            elif char == close_char and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    fragments.append(text[start : index + 1])
                    start = None

        return fragments

    def _repair_json_candidate(self, text: str) -> str:
        """
        負責執行 BFCLToolCallParser 中的 _repair_json_candidate 流程，清理常見 markdown 與 JSON 尾逗號問題。

        Args:
            text: 可能包含 JSON 的候選文字。

        Returns:
            清理後的候選 JSON 字串。

        限制或副作用:
            只做保守修補，不會嘗試把任意自然語言轉成 JSON。
        """
        candidate = text.strip()
        candidate = re.sub(r"^```json\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^```\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        return candidate.strip()

    def _normalize_parsed_value(self, parsed: Any) -> list[dict[str, Any]] | None:
        """
        負責執行 BFCLToolCallParser 中的 _normalize_parsed_value 流程，將 JSON 解析結果轉成 tool call list。

        Args:
            parsed: json.loads 產生的 Python 物件。

        Returns:
            正規化後的 tool call list；若格式不是 tool call，回傳 None。

        限制或副作用:
            支援常見 wrapper 欄位，但不會對未知業務欄位做推論。
        """
        if parsed is None:
            return []
        if isinstance(parsed, list):
            return self._normalize_calls(parsed)
        if not isinstance(parsed, dict):
            return None

        for key in ("tool_calls", "function_calls", "calls", "call", "final_calls", "final_answer", "answer"):
            if key in parsed:
                value = parsed.get(key)
                if isinstance(value, str) and self._is_empty_call(value):
                    return []
                if isinstance(value, str):
                    nested = self.parse(value)
                    if nested or value.strip() == "[]":
                        return nested
                if isinstance(value, list):
                    return self._normalize_calls(value)
                if isinstance(value, dict):
                    return self._normalize_calls([value])

        if self._looks_like_call_dict(parsed):
            return self._normalize_calls([parsed])

        return None

    def _looks_like_call_dict(self, value: dict[str, Any]) -> bool:
        """
        負責執行 BFCLToolCallParser 中的 _looks_like_call_dict 流程，判斷 dict 是否像單一 function call。

        Args:
            value: 要檢查的字典。

        Returns:
            若字典包含可推導 function name 的欄位，回傳 True；否則回傳 False。

        限制或副作用:
            僅檢查欄位名稱，不驗證 function 是否存在於 BFCL schema。
        """
        return any(key in value for key in ("name", "function_name", "tool_name", "function"))

    def _normalize_calls(self, calls: list[Any]) -> list[dict[str, Any]]:
        """
        負責執行 BFCLToolCallParser 中的 _normalize_calls 流程，將不同 tool call 表示統一為 name/arguments 格式。

        Args:
            calls: 待正規化的 call 清單。

        Returns:
            每筆包含 name 與 arguments 的 tool call 清單。

        限制或副作用:
            會丟棄無法推導 function name 的項目。
        """
        normalized: list[dict[str, Any]] = []
        for call in calls or []:
            normalized_call = self._normalize_single_call(call)
            if normalized_call is not None:
                normalized.append(normalized_call)
        return normalized

    def _normalize_single_call(self, call: Any) -> dict[str, Any] | None:
        """
        負責執行 BFCLToolCallParser 中的 _normalize_single_call 流程，正規化單一 tool call 物件。

        Args:
            call: 可能是 dict、OpenAI tool_call dict 或 function call 字串的輸入資料。

        Returns:
            正規化後的單一 tool call；若無法解析則回傳 None。

        限制或副作用:
            arguments 若是 JSON 字串會被嘗試解析；解析失敗則保留為空字典。
        """
        if isinstance(call, str):
            parsed_calls = self._parse_python_call_syntax(call)
            return parsed_calls[0] if parsed_calls else None
        if not isinstance(call, dict):
            return None

        source = call
        if isinstance(call.get("function"), dict):
            source = call["function"]

        name = (
            source.get("name")
            or source.get("function_name")
            or source.get("tool_name")
        )
        if not name:
            return None

        arguments = (
            source.get("arguments")
            if "arguments" in source
            else source.get("parameters", source.get("args", {}))
        )
        return {
            "name": str(name).strip(),
            "arguments": self._normalize_arguments(arguments),
        }

    def _normalize_arguments(self, arguments: Any) -> dict[str, Any]:
        """
        負責執行 BFCLToolCallParser 中的 _normalize_arguments 流程，將 arguments 正規化為 dict。

        Args:
            arguments: 可能是 dict、JSON 字串、None 或其他型別的參數資料。

        Returns:
            正規化後的 arguments 字典。

        限制或副作用:
            非 dict 且無法解析的 arguments 會被轉成空字典。
        """
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                    return parsed if isinstance(parsed, dict) else {}
                except (ValueError, SyntaxError):
                    return {}
        return {}

    def _parse_python_call_syntax(self, text: str) -> list[dict[str, Any]]:
        """
        負責執行 BFCLToolCallParser 中的 _parse_python_call_syntax 流程，解析 function_name(arg=value) 類型輸出。

        Args:
            text: 模型原始回覆或其中一段文字。

        Returns:
            從 Python call syntax 萃取出的 tool call 清單。

        限制或副作用:
            只支援可被 ast.parse 解析的單行或多行 expression，不執行任何程式碼。
        """
        calls: list[dict[str, Any]] = []
        for line in self._split_nonempty_lines(text):
            candidate = line.strip().rstrip(",")
            try:
                tree = ast.parse(candidate, mode="eval")
            except SyntaxError:
                continue
            call_node = tree.body
            if not isinstance(call_node, ast.Call):
                continue
            name = self._call_name(call_node.func)
            if not name:
                continue
            arguments: dict[str, Any] = {}
            for keyword in call_node.keywords:
                if keyword.arg is None:
                    continue
                arguments[keyword.arg] = self._literal_node_value(keyword.value)
            calls.append({"name": name, "arguments": arguments})
        return calls

    def _call_name(self, node: ast.AST) -> str:
        """
        負責執行 BFCLToolCallParser 中的 _call_name 流程，從 AST call 節點取出 function 名稱。

        Args:
            node: ast.Call.func 對應的 AST 節點。

        Returns:
            function 名稱文字；若無法推導則回傳空字串。

        限制或副作用:
            對屬性呼叫會回傳完整 dotted name，例如 client.search。
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def _literal_node_value(self, node: ast.AST) -> Any:
        """
        負責執行 BFCLToolCallParser 中的 _literal_node_value 流程，將 AST literal 節點轉成 Python 值。

        Args:
            node: keyword argument 的 AST value 節點。

        Returns:
            literal_eval 後的 Python 值；失敗時回傳原始碼文字。

        限制或副作用:
            不執行程式碼，只使用 ast.literal_eval 與 ast.unparse。
        """
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError):
            try:
                return ast.unparse(node)
            except Exception:
                return ""
