import json
import re

def _repair_json_candidate(text: str) -> str:
    """
    負責執行 parser.json_parse 中的 _repair_json_candidate 流程，依照 parser.json_parse 的流程需求處理 _repair_json_candidate 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        text: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    candidate = text.strip()
    if not candidate:
        return candidate

    candidate = re.sub(r"^```json\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^```\s*", "", candidate)
    candidate = re.sub(r"\s*```$", "", candidate)

    first_brace = candidate.find("{")
    last_brace = candidate.rfind("}")
    if first_brace != -1:
        if last_brace != -1 and last_brace >= first_brace:
            candidate = candidate[first_brace:last_brace + 1]
        else:
            candidate = candidate[first_brace:]

    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    if open_braces > close_braces:
        candidate += "}" * (open_braces - close_braces)

    open_brackets = candidate.count("[")
    close_brackets = candidate.count("]")
    if open_brackets > close_brackets:
        candidate += "]" * (open_brackets - close_brackets)

    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)

    return candidate

def try_parse_json(reply: str) -> dict | None:
    """
    負責執行 parser.json_parse 中的 try_parse_json 流程，依照 parser.json_parse 的流程需求處理 try_parse_json 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        reply: 模型、節點或工具產生的候選回覆內容。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 dict | None。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if not reply:
        return None

    text = _repair_json_candidate(reply)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as e:
        print(f"[try_parse_json] direct parse failed: {e}")

    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", reply, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = _repair_json_candidate(fenced_match.group(1))
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                print("[try_parse_json] repaired parse success")
                print("[try_parse_json] candidate:", repr(candidate))
                return parsed
        except json.JSONDecodeError as e:
            print(f"[try_parse_json] fenced parse failed: {e}")

    brace_match = re.search(r"(\{.*)", reply, re.DOTALL)
    if brace_match:
        candidate = _repair_json_candidate(brace_match.group(1))
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                print("[try_parse_json] repaired parse success")
                print("[try_parse_json] candidate:", repr(candidate))
                return parsed
        except json.JSONDecodeError as e:
            print(f"[try_parse_json] brace parse failed: {e}")

    return None
