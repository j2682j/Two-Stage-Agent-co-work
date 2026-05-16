from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Optional


def answer_equivalence(answer_a: str, answer_b: str) -> bool:
    """
    負責執行 utils.network_utils 中的 answer_equivalence 流程，依照 utils.network_utils 的流程需求處理 answer_equivalence 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        answer_a: 此流程需要使用的輸入資料。
        answer_b: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    math_a = extract_math_answer(answer_a)
    math_b = extract_math_answer(answer_b)

    if math_a is not None and math_b is not None:
        return math_a == math_b

    type_a = detect_answer_type(answer_a)
    type_b = detect_answer_type(answer_b)

    if type_a != type_b:
        return normalize_for_exact(answer_a) == normalize_for_exact(answer_b)

    if type_a == "choice":
        return extract_choice_answer(answer_a) == extract_choice_answer(answer_b)

    if type_a == "math":
        return extract_math_answer(answer_a) == extract_math_answer(answer_b)

    info_a = extract_key_info(answer_a)
    info_b = extract_key_info(answer_b)

    cheap_result = cheap_key_match(info_a, info_b)
    if cheap_result is not None:
        return cheap_result

    from network.slm_agent import SLM_Agent

    slm_answer_judge = SLM_Agent(model_name="gpt-oss:20b")
    if slm_answer_judge is None:
        return False

    judge_prompt = f"""
        You are a strict answer equivalence judge.

        Determine whether the two answers express the same final conclusion.
        Ignore wording differences. Focus only on final meaning.

        Return JSON only:
        {{"equivalent": true}} or {{"equivalent": false}}

        Answer A key info:
        {json.dumps(info_a, ensure_ascii=False)}

        Answer B key info:
        {json.dumps(info_b, ensure_ascii=False)}
        """.strip()

    try:
        raw = slm_answer_judge.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a strict JSON-only answer equivalence judge.",
                },
                {"role": "user", "content": judge_prompt},
            ]
        )
        parsed = json.loads(raw)
        return bool(parsed.get("equivalent", False))
    except Exception:
        return False


def normalize_text(text: Any) -> str:
    """
    負責執行 utils.network_utils 中的 normalize_text 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
    
    Args:
        text: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("嚗?, ", ":")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_exact(text: Any) -> str:
    """
    負責執行 utils.network_utils 中的 normalize_for_exact 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
    
    Args:
        text: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = normalize_text(text).lower()
    text = text.strip(" \n\r\t.,;:!?\"'")
    return text


def extract_choice_answer(text: Any) -> Optional[str]:
    """
    負責執行 utils.network_utils 中的 extract_choice_answer 流程，解析模型輸出並取出答案、決策、排序或 JSON 結構。
    
    Args:
        text: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Optional[str]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = normalize_text(text)

    direct = re.fullmatch(r"\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if direct:
        return direct.group(1).upper()

    labeled_patterns = [
        r"final answer\s*[^A-Da-d0-9]{0,3}\s*\(?([A-Da-d])\)?",
        r"the answer is\s*[^A-Da-d0-9]{0,3}\s*\(?([A-Da-d])\)?",
        r"answer\s*[^A-Da-d0-9]{0,3}\s*\(?([A-Da-d])\)?",
    ]
    for pattern in labeled_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def extract_math_answer(text: Any) -> Optional[str]:
    """
    負責執行 utils.network_utils 中的 extract_math_answer 流程，解析模型輸出並取出答案、決策、排序或 JSON 結構。
    
    Args:
        text: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Optional[str]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = normalize_text(text)

    patterns = [
        r"\bfinal answer\s*[^0-9-]{0,3}\s*(-?\d+(?:\.\d+)?)\b",
        r"\banswer\s*[^0-9-]{0,3}\s*(-?\d+(?:\.\d+)?)\b",
        r"\bthe answer is\s*[^0-9-]{0,3}\s*(-?\d+(?:\.\d+)?)\b",
        r"^\s*(-?\d+(?:\.\d+)?)\s*$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return normalize_number(m.group(1))

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) == 1:
        return normalize_number(nums[0])

    return None


def normalize_number(value: str) -> str:
    """
    負責執行 utils.network_utils 中的 normalize_number 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
    
    Args:
        value: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    try:
        dec = Decimal(value)
        return format(dec.normalize(), "f").rstrip("0").rstrip(".") or "0"
    except (InvalidOperation, ValueError):
        return value.strip()


def detect_answer_type(text: Any) -> str:
    """
    負責執行 utils.network_utils 中的 detect_answer_type 流程，依照 utils.network_utils 的流程需求處理 detect_answer_type 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        text: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if extract_choice_answer(text) is not None:
        return "choice"
    if extract_math_answer(text) is not None:
        return "math"
    return "free_form"


def extract_key_info(text: Any) -> dict[str, Any]:
    """
    負責執行 utils.network_utils 中的 extract_key_info 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
    
    Args:
        text: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = normalize_text(text)

    lower = text.lower()
    lower = re.sub(r"[^\w\s:/.-]", " ", lower)
    tokens = [t for t in lower.split() if len(t) > 1]

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "that",
        "this",
        "it",
        "as",
        "with",
        "by",
        "from",
        "answer",
        "final",
        "therefore",
        "so",
        "result",
    }
    keywords = [t for t in tokens if t not in stopwords]

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    dates = re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text)

    return {
        "normalized_text": normalize_for_exact(text),
        "keywords": sorted(set(keywords))[:20],
        "numbers": [normalize_number(n) for n in numbers],
        "dates": dates,
    }


def cheap_key_match(info_a: dict[str, Any], info_b: dict[str, Any]) -> Optional[bool]:
    """
    負責執行 utils.network_utils 中的 cheap_key_match 流程，依照 utils.network_utils 的流程需求處理 cheap_key_match 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        info_a: 此流程需要使用的輸入資料。
        info_b: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Optional[bool]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if info_a["normalized_text"] == info_b["normalized_text"]:
        return True

    if (
        info_a["numbers"]
        and info_b["numbers"]
        and info_a["numbers"] == info_b["numbers"]
    ):
        kw_a = set(info_a["keywords"])
        kw_b = set(info_b["keywords"])
        if not kw_a or not kw_b or len(kw_a & kw_b) >= 1:
            return True

    kw_a = set(info_a["keywords"])
    kw_b = set(info_b["keywords"])
    if kw_a and kw_b:
        overlap = len(kw_a & kw_b)
        union = len(kw_a | kw_b)
        if union > 0 and overlap / union >= 0.8:
            return True
        if overlap == 0 and info_a["numbers"] != info_b["numbers"]:
            return False

    return None


def should_use_calculator(question: str) -> bool:
    """
    負責執行 utils.network_utils 中的 should_use_calculator 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
    
    Args:
        question: 目前要處理的任務、問題或查詢文字。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = normalize_text(question).lower()

    math_keywords = [
        "calculate",
        "compute",
        "solve",
        "math",
        "percentage",
        "percent",
        "sum",
        "difference",
        "product",
        "quotient",
        "sqrt",
        "square root",
    ]

    has_math_keyword = any(keyword in text for keyword in math_keywords)
    has_operator = any(op in text for op in ["+", "-", "*", "/", "=", "(", ")"])
    number_count = len(re.findall(r"\d+(?:\.\d+)?", text))

    if has_operator and number_count >= 2:
        return True

    if has_math_keyword and number_count >= 1:
        return True

    return False


def should_use_search(question: str) -> bool:
    """
    負責執行 utils.network_utils 中的 should_use_search 流程，檢查目前輸入、狀態或條件是否符合流程繼續執行的要求。
    
    Args:
        question: 目前要處理的任務、問題或查詢文字。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = normalize_text(question).lower()

    search_keywords = [
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "difference",
        "compare",
        "explain",
        "history",
        "capital",
        "country",
        "language",
        "learn",
    ]

    if should_use_calculator(question):
        word_problem_markers = [
            "how much",
            "how many",
            "total",
            "remainder",
            "left",
            "each",
            "every",
            "per day",
            "per hour",
            "per item",
            "cost",
            "price",
            "earn",
            "make",
            "dollars",
            "$",
            "sold",
            "sell",
            "buys",
            "spent",
            "remaining",
        ]
        if any(marker in text for marker in word_problem_markers):
            return False

    if any(keyword in text for keyword in search_keywords):
        return True

    if not should_use_calculator(question):
        return True

    return False
