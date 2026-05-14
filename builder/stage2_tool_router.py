from __future__ import annotations

from dataclasses import dataclass, field
import ast
import re
from typing import Any


def _clean(value: Any) -> str:
    """
    負責執行 builder.stage2_tool_router 中的 _clean 流程，依照 builder.stage2_tool_router 的流程需求處理 _clean 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: Any, default: str = "general_reasoning") -> str:
    """
    負責執行 builder.stage2_tool_router 中的 _slug 流程，依照 builder.stage2_tool_router 的流程需求處理 _slug 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 此流程需要使用的輸入資料。
        default: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = _clean(value).lower()
    text = re.sub(r"[^a-z0-9_./:-]+", "_", text).strip("_")
    return text or default


@dataclass(slots=True)
class Stage2ToolRoutingInput:
    """
    負責在 builder.stage2_tool_router 中封裝 Stage2ToolRoutingInput，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    question: str
    task_type: str = "general_reasoning"
    trigger_terms: list[str] = field(default_factory=list)
    tool_policy: dict[str, list[str]] = field(default_factory=dict)
    stage1_result: str | None = None
    top_k_answers: list[str] = field(default_factory=list)
    judge_scores: list[float] = field(default_factory=list)
    has_attachment: bool = False


@dataclass(slots=True)
class Stage2ToolRoutingDecision:
    """
    負責在 builder.stage2_tool_router 中封裝 Stage2ToolRoutingDecision，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    use_search: bool = False
    use_calculator: bool = False
    use_python_solver: bool = False
    use_attachment: bool = False
    use_memory: bool = False
    use_rag: bool = False
    no_tool: bool = False
    calculator_expression: str | None = None
    task_type: str = "general_reasoning"
    trigger_terms: list[str] = field(default_factory=list)
    tool_policy: dict[str, list[str]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        負責執行 Stage2ToolRoutingDecision 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "use_search": self.use_search,
            "use_calculator": self.use_calculator,
            "use_python_solver": self.use_python_solver,
            "use_attachment": self.use_attachment,
            "use_memory": self.use_memory,
            "use_rag": self.use_rag,
            "no_tool": self.no_tool,
            "calculator_expression": self.calculator_expression,
            "task_type": self.task_type,
            "trigger_terms": list(self.trigger_terms),
            "tool_policy": {
                "prefer": list(self.tool_policy.get("prefer", [])),
                "optional": list(self.tool_policy.get("optional", [])),
                "avoid": list(self.tool_policy.get("avoid", [])),
            },
            "reasons": list(self.reasons),
        }


class Stage2ToolRouter:
    """
    負責在 builder.stage2_tool_router 中封裝 Stage2ToolRouter，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    FACTUAL_TASKS = {"factual_search", "source_lookup", "counting_scope"}
    MODELING_TASKS = {"stochastic_process", "combinatorics", "graph_reasoning"}
    TABLE_TASKS = {"spreadsheet_reasoning", "table_processing"}
    ATTACHMENT_TASKS = {"spreadsheet_reasoning", "image_understanding", "audio_understanding"}
    CALC_TASKS = {"unit_conversion", "simple_arithmetic"}

    def route(self, routing_input: Stage2ToolRoutingInput) -> Stage2ToolRoutingDecision:
        """
        負責執行 Stage2ToolRouter 中的 route 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            routing_input: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Stage2ToolRoutingDecision。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        question = _clean(routing_input.question)
        task_type = _slug(routing_input.task_type)
        policy = self._normalize_policy(routing_input.tool_policy)
        triggers = [_slug(term, default="") for term in routing_input.trigger_terms if _slug(term, default="")]
        decision = Stage2ToolRoutingDecision(
            task_type=task_type,
            trigger_terms=triggers,
            tool_policy=policy,
        )

        if routing_input.has_attachment or task_type in self.ATTACHMENT_TASKS:
            decision.use_attachment = True
            decision.reasons.append("attachment evidence is required or task_type uses attachments")

        prefer = set(policy.get("prefer", []))
        optional = set(policy.get("optional", []))
        avoid = set(policy.get("avoid", []))

        if task_type in self.FACTUAL_TASKS or "search" in prefer:
            decision.use_search = True
            decision.reasons.append(f"task_type={task_type} requires factual/search evidence")
        elif "search" in optional and self._question_requires_external_evidence(question):
            decision.use_search = True
            decision.reasons.append("question appears to require external evidence")

        if task_type in self.MODELING_TASKS or "python_solver" in prefer:
            decision.use_python_solver = True
            decision.reasons.append(f"task_type={task_type} benefits from modeling or simulation")

        if task_type in self.TABLE_TASKS or {"pandas_excel", "python_solver"} & prefer:
            decision.use_python_solver = True
            decision.reasons.append(f"task_type={task_type} benefits from structured Python/data handling")

        if self._question_needs_numeric_model(question) and not decision.use_calculator:
            decision.use_python_solver = True
            decision.reasons.append("question needs numeric modeling but no safe calculator expression was extracted")

        expression = self.extract_calculator_expression(question)
        if task_type in self.CALC_TASKS or "calculator" in prefer:
            if expression:
                decision.use_calculator = True
                decision.calculator_expression = expression
                decision.reasons.append("safe calculator expression extracted")
            else:
                decision.use_python_solver = True
                decision.reasons.append("calculator requested but no safe expression was extractable")
        elif expression and self._looks_like_direct_calculation(question):
            decision.use_calculator = True
            decision.calculator_expression = expression
            decision.reasons.append("question is a direct arithmetic expression")

        if "calculator_on_raw_question" in avoid and not expression:
            decision.use_calculator = False
            decision.calculator_expression = None
            decision.reasons.append("calculator_on_raw_question is disallowed by insight policy")

        if "search" in avoid:
            decision.use_search = False
            decision.reasons.append("search is disallowed by insight policy")

        if self._answers_disagree(routing_input.top_k_answers):
            if task_type in self.FACTUAL_TASKS:
                decision.use_search = True
                decision.reasons.append("top_k answers disagree; factual task escalates to search")
            elif task_type in self.MODELING_TASKS or task_type in self.CALC_TASKS:
                decision.use_python_solver = True
                decision.reasons.append("top_k answers disagree; modeling task escalates to python_solver")

        if not any(
            [
                decision.use_search,
                decision.use_calculator,
                decision.use_python_solver,
                decision.use_attachment,
                decision.use_memory,
                decision.use_rag,
            ]
        ):
            decision.no_tool = True
            decision.reasons.append("no tool needed for high-level general reasoning path")

        return decision

    def extract_calculator_expression(self, question: str) -> str | None:
        """
        負責執行 Stage2ToolRouter 中的 extract_calculator_expression 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        text = _clean(question)
        if not text:
            return None

        candidates: list[str] = []
        if self._is_safe_expression(text):
            candidates.append(text)

        quoted = re.findall(r"`([^`]+)`", text)
        candidates.extend(quoted)

        labeled = re.findall(
            r"(?:calculate|compute|evaluate|expression)\s*[:=]\s*([0-9eEpiPI().,+\-*/%^\s]+)",
            text,
            flags=re.IGNORECASE,
        )
        candidates.extend(labeled)

        # Last resort: extract compact arithmetic spans, but only if the span
        # includes an arithmetic operator and at least two numbers.
        spans = re.findall(r"(?<![A-Za-z])[-+*/().\d\s]{5,}(?![A-Za-z])", text)
        candidates.extend(spans)

        for candidate in candidates:
            expression = self._normalize_expression(candidate)
            if self._is_safe_expression(expression):
                return expression
        return None

    def _normalize_policy(self, policy: dict[str, Any] | None) -> dict[str, list[str]]:
        """
        負責執行 Stage2ToolRouter 中的 _normalize_policy 流程，依照 Stage2ToolRouter 的流程需求處理 _normalize_policy 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            policy: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, list[str]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        data = policy or {}
        return {
            "prefer": [_slug(item, default="") for item in data.get("prefer", []) if _slug(item, default="")],
            "optional": [_slug(item, default="") for item in data.get("optional", []) if _slug(item, default="")],
            "avoid": [_slug(item, default="") for item in data.get("avoid", []) if _slug(item, default="")],
        }

    def _normalize_expression(self, value: str) -> str:
        """
        負責執行 Stage2ToolRouter 中的 _normalize_expression 流程，依照 Stage2ToolRouter 的流程需求處理 _normalize_expression 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            value: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        expression = _clean(value)
        expression = expression.replace("^", "**")
        expression = expression.replace("×", "*").replace("÷", "/")
        expression = expression.strip(" .,:;")
        return expression

    def _is_safe_expression(self, expression: str) -> bool:
        """
        負責執行 Stage2ToolRouter 中的 _is_safe_expression 流程，依照 Stage2ToolRouter 的流程需求處理 _is_safe_expression 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            expression: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        expression = self._normalize_expression(expression)
        if not expression:
            return False
        if not re.fullmatch(r"[0-9eEpiPI().,+\-*/%\s*]+", expression):
            return False
        if len(re.findall(r"\d+(?:\.\d+)?", expression)) < 2:
            return False
        if not any(op in expression for op in ["+", "-", "*", "/", "%"]):
            return False
        try:
            ast.parse(expression, mode="eval")
        except SyntaxError:
            return False
        return True

    def _looks_like_direct_calculation(self, question: str) -> bool:
        """
        負責執行 Stage2ToolRouter 中的 _looks_like_direct_calculation 流程，依照 Stage2ToolRouter 的流程需求處理 _looks_like_direct_calculation 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        text = _clean(question).lower()
        if len(text.split()) <= 8 and any(op in text for op in ["+", "-", "*", "/", "%"]):
            return True
        return any(marker in text for marker in ["calculate:", "compute:", "evaluate:"])

    def _question_requires_external_evidence(self, question: str) -> bool:
        """
        負責執行 Stage2ToolRouter 中的 _question_requires_external_evidence 流程，依照 Stage2ToolRouter 的流程需求處理 _question_requires_external_evidence 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        text = _clean(question).lower()
        markers = [
            "who",
            "when",
            "where",
            "website",
            "webpage",
            "wikipedia",
            "latest",
            "current",
            "published",
            "released",
            "source",
            "record",
            "page",
        ]
        return any(marker in text for marker in markers)

    def _question_needs_numeric_model(self, question: str) -> bool:
        """
        負責執行 Stage2ToolRouter 中的 _question_needs_numeric_model 流程，依照 Stage2ToolRouter 的流程需求處理 _question_needs_numeric_model 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        text = _clean(question).lower()
        markers = [
            "probability",
            "odds",
            "random",
            "randomly",
            "maximize",
            "permutation",
            "combination",
            "simulate",
            "dynamic programming",
            "state transition",
            "how many thousand",
            "round",
            "nearest",
            "pace",
            "distance",
        ]
        return any(marker in text for marker in markers)

    def _answers_disagree(self, answers: list[str]) -> bool:
        """
        負責執行 Stage2ToolRouter 中的 _answers_disagree 流程，依照 Stage2ToolRouter 的流程需求處理 _answers_disagree 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            answers: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        normalized = {re.sub(r"\W+", "", str(answer or "").lower()) for answer in answers if str(answer or "").strip()}
        return len(normalized) > 1
