import json
import re

def _repair_json_candidate(text: str) -> str:
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
