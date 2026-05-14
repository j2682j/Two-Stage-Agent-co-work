from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Iterator

import networkx as nx
from networkx.readwrite import json_graph


def _clean(value: Any) -> str:
    """
    負責執行 memory.graph.interaction_graph 中的 _clean 流程，依照 memory.graph.interaction_graph 的流程需求處理 _clean 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: Any, default: str = "node") -> str:
    """
    負責執行 memory.graph.interaction_graph 中的 _slug 流程，依照 memory.graph.interaction_graph 的流程需求處理 _slug 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
        default: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    text = _clean(value).lower()
    text = re.sub(r"[^a-z0-9_.:-]+", "_", text).strip("_")
    return text or default


def _json_dumps(value: Any) -> str:
    """
    負責執行 memory.graph.interaction_graph 中的 _json_dumps 流程，依照 memory.graph.interaction_graph 的流程需求處理 _json_dumps 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 str。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return json.dumps(value, ensure_ascii=False, default=str)


def _tokenize(value: Any) -> set[str]:
    """
    負責執行 memory.graph.interaction_graph 中的 _tokenize 流程，依照 memory.graph.interaction_graph 的流程需求處理 _tokenize 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        value: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 set[str]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return set(re.findall(r"[a-z0-9_./:-]{3,}", _clean(value).lower()))


@dataclass(slots=True)
class TaskMetadata:
    """
    負責在 memory.graph.interaction_graph 中封裝 TaskMetadata，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    task_type: str = "general_reasoning"
    trigger_terms: list[str] = field(default_factory=list)
    attachment_type: str | None = None
    failure_modes: list[str] = field(default_factory=list)
    tool_policy: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.45

    def to_dict(self) -> dict[str, Any]:
        """
        負責執行 TaskMetadata 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "task_type": self.task_type,
            "trigger_terms": list(self.trigger_terms),
            "attachment_type": self.attachment_type,
            "failure_modes": list(self.failure_modes),
            "tool_policy": {key: list(value) for key, value in self.tool_policy.items()},
            "confidence": float(self.confidence),
        }


def classify_task_metadata(question: str, attachment_type: str | None = None) -> TaskMetadata:
    """
    負責執行 memory.graph.interaction_graph 中的 classify_task_metadata 流程，依照 memory.graph.interaction_graph 的流程需求處理 classify_task_metadata 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        question: 目前要處理的任務、問題或查詢文字。
        attachment_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 TaskMetadata。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """

    text = _clean(question).lower()
    ext = _slug(attachment_type or "", default="")

    if ext in {"xlsx", "xls", "csv"}:
        return TaskMetadata(
            task_type="spreadsheet_reasoning",
            trigger_terms=[ext, "spreadsheet"],
            attachment_type=ext,
            failure_modes=["table_scope_mismatch", "missed_attachment_evidence"],
            tool_policy={
                "prefer": ["attachment_reader", "pandas_excel"],
                "optional": ["python_solver"],
                "avoid": ["raw_text_guess"],
            },
            confidence=0.9,
        )
    if ext in {"png", "jpg", "jpeg", "webp", "gif"}:
        return TaskMetadata(
            task_type="image_understanding",
            trigger_terms=[ext, "image"],
            attachment_type=ext,
            failure_modes=["missed_visual_evidence", "weak_ocr_or_caption"],
            tool_policy={
                "prefer": ["attachment_reader", "vision_model"],
                "optional": ["search"],
                "avoid": ["text_only_guess"],
            },
            confidence=0.9,
        )
    if ext in {"mp3", "wav", "m4a", "flac", "ogg"}:
        return TaskMetadata(
            task_type="audio_understanding",
            trigger_terms=[ext, "audio"],
            attachment_type=ext,
            failure_modes=["missed_audio_evidence", "transcription_error"],
            tool_policy={
                "prefer": ["attachment_reader", "audio_transcription"],
                "optional": ["search"],
                "avoid": ["text_only_guess"],
            },
            confidence=0.9,
        )

    rules: list[tuple[str, list[str], list[str], dict[str, list[str]], float]] = [
        (
            "stochastic_process",
            ["random", "randomly", "probability", "odds", "maximize", "position", "advance"],
            ["missing_state_transition_model", "surface_numeric_guess"],
            {"prefer": ["python_solver"], "optional": ["search"], "avoid": ["calculator_on_raw_question"]},
            0.82,
        ),
        (
            "spreadsheet_reasoning",
            ["spreadsheet", "excel", "xlsx", "xls", "sheet", "cell", "row", "column", "color"],
            ["table_scope_mismatch", "missed_attachment_evidence"],
            {"prefer": ["attachment_reader", "pandas_excel"], "optional": ["python_solver"], "avoid": ["raw_text_guess"]},
            0.86,
        ),
        (
            "image_understanding",
            ["image", "png", "jpg", "jpeg", "screenshot", "photo", "visual", "picture"],
            ["missed_visual_evidence", "weak_ocr_or_caption"],
            {"prefer": ["attachment_reader", "vision_model"], "optional": ["search"], "avoid": ["text_only_guess"]},
            0.82,
        ),
        (
            "audio_understanding",
            ["audio", "mp3", "listen", "transcribe", "recording", "sound"],
            ["missed_audio_evidence", "transcription_error"],
            {"prefer": ["attachment_reader", "audio_transcription"], "optional": ["search"], "avoid": ["text_only_guess"]},
            0.82,
        ),
        (
            "counting_scope",
            ["how many", "count", "number of", "total", "list all", "between", "during"],
            ["scope_filter_mismatch", "boundary_condition_slip"],
            {"prefer": ["search"], "optional": ["python_solver"], "avoid": ["candidate_collapse"]},
            0.72,
        ),
        (
            "unit_conversion",
            ["unit", "convert", "nearest", "round", "km", "mile", "meter", "kg", "percent"],
            ["unit_or_scale_mismatch", "format_or_rounding_slip"],
            {"prefer": ["python_solver"], "optional": ["calculator"], "avoid": ["unverified_mental_math"]},
            0.70,
        ),
        (
            "factual_search",
            ["who", "when", "where", "website", "source", "latest", "current", "published", "released"],
            ["insufficient_evidence", "outdated_fact"],
            {"prefer": ["search"], "optional": ["rag"], "avoid": ["memory_as_answer_lookup"]},
            0.68,
        ),
    ]

    best: TaskMetadata | None = None
    for task_type, terms, failures, policy, base_score in rules:
        matched: list[str] = []
        for term in terms:
            if " " in term:
                if term in text:
                    matched.append(term.replace(" ", "_"))
            elif term in text:
                matched.append(term)
        if not matched:
            continue
        score = base_score + min(0.12, 0.02 * len(matched))
        if best is None or score > best.confidence:
            best = TaskMetadata(
                task_type=task_type,
                trigger_terms=matched,
                attachment_type=ext or None,
                failure_modes=failures,
                tool_policy=policy,
                confidence=min(score, 0.98),
            )
    if best is not None:
        return best

    return TaskMetadata(
        task_type="general_reasoning",
        trigger_terms=sorted(_tokenize(question))[:8],
        attachment_type=ext or None,
        failure_modes=["insufficient_verification"],
        tool_policy={"prefer": [], "optional": ["search"], "avoid": ["memory_as_answer_lookup"]},
        confidence=0.45,
    )


@dataclass(slots=True)
class AgentMessage:
    """
    負責在 memory.graph.interaction_graph 中封裝 AgentMessage，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    agent_name: str
    system_instruction: str | None = None
    user_instruction: str | None = None
    message: str | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def add_extra_field(self, key: str, value: Any) -> None:
        """
        負責執行 AgentMessage 中的 add_extra_field 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            key: 記憶系統提供的檢索結果、寫入資料或操作介面。
            value: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Any | None:
        """
        負責執行 AgentMessage 中的 get_extra_field 流程，依照 AgentMessage 的流程需求處理 get_extra_field 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            key: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Any | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.extra_fields.get(key)


class StateChain:
    """
    負責在 memory.graph.interaction_graph 中封裝 StateChain，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self) -> None:
        """
        負責執行 StateChain 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._states: list[nx.DiGraph] = []
        self._current_state: nx.DiGraph | None = None

    def __iter__(self) -> Iterator[nx.DiGraph]:
        """
        負責執行 StateChain 中的 __iter__ 流程，依照 StateChain 的流程需求處理 __iter__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Iterator[nx.DiGraph]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return iter(self.chain_of_states)

    def __len__(self) -> int:
        """
        負責執行 StateChain 中的 __len__ 流程，依照 StateChain 的流程需求處理 __len__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return len(self.chain_of_states)

    @property
    def chain_of_states(self) -> list[nx.DiGraph]:
        """
        負責執行 StateChain 中的 chain_of_states 流程，依照 StateChain 的流程需求處理 chain_of_states 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[nx.DiGraph]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        states = list(self._states)
        if self._current_state is not None and self._current_state.number_of_nodes() > 0:
            states.append(self._current_state)
        return states

    def start_state(
        self,
        *,
        state_id: str,
        state_type: str,
        stage: str,
        round_id: int | None = None,
        action: str = "",
        observation: str = "",
        reward: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> nx.DiGraph:
        """
        負責執行 StateChain 中的 start_state 流程，依照 StateChain 的流程需求處理 start_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            state_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            state_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            stage: 目前執行的階段、輪次或流程位置。
            round_id: 目前執行的階段、輪次或流程位置。
            action: 記憶系統提供的檢索結果、寫入資料或操作介面。
            observation: 記憶系統提供的檢索結果、寫入資料或操作介面。
            reward: 記憶系統提供的檢索結果、寫入資料或操作介面。
            extra_fields: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 nx.DiGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self._current_state is not None:
            self.end_state()

        graph = nx.DiGraph()
        graph.graph.update(
            {
                "state_id": state_id,
                "state_type": state_type,
                "stage": stage,
                "round": round_id,
                "action": action,
                "observation": observation,
                "reward": reward,
                "name_counter": {},
                "extra_fields": dict(extra_fields or {}),
            }
        )
        self._current_state = graph
        return graph

    def end_state(
        self,
        *,
        action: str | None = None,
        observation: str | None = None,
        reward: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """
        負責執行 StateChain 中的 end_state 流程，依照 StateChain 的流程需求處理 end_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            action: 記憶系統提供的檢索結果、寫入資料或操作介面。
            observation: 記憶系統提供的檢索結果、寫入資料或操作介面。
            reward: 記憶系統提供的檢索結果、寫入資料或操作介面。
            extra_fields: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self._current_state is None:
            return
        if action is not None:
            self._current_state.graph["action"] = action
        if observation is not None:
            self._current_state.graph["observation"] = observation
        if reward is not None:
            self._current_state.graph["reward"] = reward
        if extra_fields:
            self._current_state.graph.setdefault("extra_fields", {}).update(extra_fields)
        self._states.append(self._current_state)
        self._current_state = None

    def add_message(
        self,
        agent_message: AgentMessage,
        upstream_agent_ids: list[str] | None = None,
        *,
        node_id: str | None = None,
        edge_type: str = "spatial",
        edge_attrs: dict[str, Any] | None = None,
        upstream_refs: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        負責執行 StateChain 中的 add_message 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            agent_message: 記憶系統提供的檢索結果、寫入資料或操作介面。
            upstream_agent_ids: 記憶系統提供的檢索結果、寫入資料或操作介面。
            node_id: 目前執行或需要記錄的代理節點識別資訊。
            edge_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            edge_attrs: 記憶系統提供的檢索結果、寫入資料或操作介面。
            upstream_refs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        current_state = self._get_current_state()
        resolved_node_id = node_id or self._generate_node_id(agent_message.agent_name)

        message_dict = asdict(agent_message)
        message_dict.setdefault("extra_fields", {})
        if upstream_refs:
            message_dict["extra_fields"].setdefault("upstream_refs", []).extend(upstream_refs)

        current_state.add_node(resolved_node_id, **message_dict)

        for up_node_id in upstream_agent_ids or []:
            if current_state.has_node(up_node_id):
                current_state.add_edge(
                    up_node_id,
                    resolved_node_id,
                    edge_type=edge_type,
                    **dict(edge_attrs or {}),
                )
            else:
                current_state.nodes[resolved_node_id]["extra_fields"].setdefault("upstream_refs", []).append(
                    {
                        "node_id": up_node_id,
                        "edge_type": edge_type,
                        **dict(edge_attrs or {}),
                    }
                )
        return resolved_node_id

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        edge_type: str = "spatial",
        **edge_attrs: Any,
    ) -> None:
        """
        負責執行 StateChain 中的 add_edge 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            source_node_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            target_node_id: 記憶系統提供的檢索結果、寫入資料或操作介面。
            edge_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **edge_attrs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        current_state = self._get_current_state()
        if not current_state.has_node(source_node_id) or not current_state.has_node(target_node_id):
            raise ValueError("Both edge endpoints must exist in the current state.")
        current_state.add_edge(source_node_id, target_node_id, edge_type=edge_type, **edge_attrs)

    def get_state(self, idx: int) -> nx.DiGraph:
        """
        負責執行 StateChain 中的 get_state 流程，依照 StateChain 的流程需求處理 get_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            idx: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 nx.DiGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        states = self.chain_of_states
        if idx >= len(states) or idx < -len(states):
            raise ValueError("Index out of range.")
        return states[idx]

    def pop_state(self, idx: int) -> nx.DiGraph:
        """
        負責執行 StateChain 中的 pop_state 流程，依照 StateChain 的流程需求處理 pop_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            idx: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 nx.DiGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self._current_state is not None:
            self.end_state()
        if idx >= len(self._states) or idx < -len(self._states):
            raise ValueError("Index out of range.")
        return self._states.pop(idx)

    def _generate_node_id(self, agent_name: str) -> str:
        """
        負責執行 StateChain 中的 _generate_node_id 流程，依照 StateChain 的流程需求處理 _generate_node_id 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            agent_name: 目前執行或需要記錄的代理節點識別資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        current_state = self._get_current_state()
        counter = current_state.graph.setdefault("name_counter", {})
        name = _slug(agent_name, default="agent")
        value = int(counter.get(name, 0))
        counter[name] = value + 1
        state_id = _slug(current_state.graph.get("state_id"), default="state")
        return f"{state_id}:{name}-{value}"

    def _get_current_state(self) -> nx.DiGraph:
        """
        負責執行 StateChain 中的 _get_current_state 流程，依照 StateChain 的流程需求處理 _get_current_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 nx.DiGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self._current_state is None:
            self.start_state(
                state_id="state0",
                state_type="unspecified",
                stage="unspecified",
            )
        return self._current_state

    def to_list(self) -> list[dict[str, Any]]:
        """
        負責執行 StateChain 中的 to_list 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 list[dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [json_graph.node_link_data(state) for state in self.chain_of_states]

    @classmethod
    def from_list(cls, states_data: list[dict[str, Any]]) -> StateChain:
        """
        負責執行 StateChain 中的 from_list 流程，依照 StateChain 的流程需求處理 from_list 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            states_data: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 StateChain。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        state_chain = cls()
        state_chain._states = [json_graph.node_link_graph(state_data) for state_data in states_data]
        state_chain._current_state = None
        return state_chain

    @staticmethod
    def to_str(state_chain: StateChain) -> str:
        """
        負責執行 StateChain 中的 to_str 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            state_chain: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return _json_dumps(state_chain.to_list())

    @staticmethod
    def from_str(state_chain_str: str | None) -> StateChain:
        """
        負責執行 StateChain 中的 from_str 流程，依照 StateChain 的流程需求處理 from_str 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            state_chain_str: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 StateChain。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not state_chain_str:
            return StateChain()
        return StateChain.from_list(json.loads(state_chain_str))


@dataclass
class InteractionGraph:
    """
    負責在 memory.graph.interaction_graph 中封裝 InteractionGraph，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    task_id: str
    task_main: str
    task_description: str | None = None
    task_trajectory: str = ""
    label: str | None = None
    chain_of_states: StateChain = field(default_factory=StateChain, repr=False)
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def add_message_to_current_state(
        self,
        agent_message: AgentMessage,
        upstream_agent_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        負責執行 InteractionGraph 中的 add_message_to_current_state 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            agent_message: 記憶系統提供的檢索結果、寫入資料或操作介面。
            upstream_agent_ids: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.chain_of_states.add_message(agent_message, upstream_agent_ids, **kwargs)

    def start_state(self, **kwargs: Any) -> nx.DiGraph:
        """
        負責執行 InteractionGraph 中的 start_state 流程，依照 InteractionGraph 的流程需求處理 start_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 nx.DiGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.chain_of_states.start_state(**kwargs)

    def end_state(
        self,
        *,
        action: str | None = None,
        observation: str | None = None,
        reward: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """
        負責執行 InteractionGraph 中的 end_state 流程，依照 InteractionGraph 的流程需求處理 end_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            action: 記憶系統提供的檢索結果、寫入資料或操作介面。
            observation: 記憶系統提供的檢索結果、寫入資料或操作介面。
            reward: 記憶系統提供的檢索結果、寫入資料或操作介面。
            extra_fields: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if action or observation:
            self.task_trajectory += f"{action or ''}\n{observation or ''}\n>"
        self.chain_of_states.end_state(
            action=action,
            observation=observation,
            reward=reward,
            extra_fields=extra_fields,
        )

    def move_state(self, action: str, observation: str, **kwargs: Any) -> None:
        """
        負責執行 InteractionGraph 中的 move_state 流程，依照 InteractionGraph 的流程需求處理 move_state 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            action: 記憶系統提供的檢索結果、寫入資料或操作介面。
            observation: 記憶系統提供的檢索結果、寫入資料或操作介面。
            **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.end_state(action=action, observation=observation, **kwargs)
        state_idx = len(self.chain_of_states.chain_of_states)
        self.start_state(
            state_id=f"{self.task_id}:state{state_idx}",
            state_type="unspecified",
            stage="unspecified",
        )

    def add_extra_field(self, key: str, value: Any) -> None:
        """
        負責執行 InteractionGraph 中的 add_extra_field 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            key: 記憶系統提供的檢索結果、寫入資料或操作介面。
            value: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Any | None:
        """
        負責執行 InteractionGraph 中的 get_extra_field 流程，依照 InteractionGraph 的流程需求處理 get_extra_field 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            key: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Any | None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.extra_fields.get(key)

    def set_task_metadata(self, metadata: TaskMetadata | dict[str, Any]) -> None:
        """
        負責執行 InteractionGraph 中的 set_task_metadata 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        data = metadata.to_dict() if isinstance(metadata, TaskMetadata) else dict(metadata)
        self.extra_fields["task_metadata"] = data
        self.extra_fields.setdefault("task_type", data.get("task_type"))
        self.extra_fields.setdefault("trigger_terms", data.get("trigger_terms", []))
        self.extra_fields.setdefault("failure_modes", data.get("failure_modes", []))
        self.extra_fields.setdefault("tool_policy", data.get("tool_policy", {}))

    def to_dict(self) -> dict[str, Any]:
        """
        負責執行 InteractionGraph 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "task_id": self.task_id,
            "task_main": self.task_main,
            "task_description": self.task_description,
            "task_trajectory": self.task_trajectory,
            "label": self.label,
            "state_chain": self.chain_of_states.to_list(),
            "extra_fields": self.extra_fields,
        }

    def to_mas_message(self) -> dict[str, Any]:
        """
        負責執行 InteractionGraph 中的 to_mas_message 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "task_id": self.task_id,
            "task_main": self.task_main,
            "task_description": self.task_description,
            "task_trajectory": self.task_trajectory,
            "label": self.label,
            "extra_fields": self.extra_fields,
            "state_chain": self.chain_of_states.to_list(),
        }

    @classmethod
    def from_dict(cls, graph_dict: dict[str, Any]) -> InteractionGraph:
        """
        負責執行 InteractionGraph 中的 from_dict 流程，依照 InteractionGraph 的流程需求處理 from_dict 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            graph_dict: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InteractionGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        state_chain_value = graph_dict.get("state_chain") or graph_dict.get("chain_of_state") or []
        if isinstance(state_chain_value, str):
            state_chain = StateChain.from_str(state_chain_value)
        else:
            state_chain = StateChain.from_list(list(state_chain_value or []))
        extra_fields = graph_dict.get("extra_fields") or {}
        if isinstance(extra_fields, str):
            extra_fields = json.loads(extra_fields or "{}")
        return cls(
            task_id=str(graph_dict.get("task_id", "") or ""),
            task_main=str(graph_dict.get("task_main", "") or ""),
            task_description=graph_dict.get("task_description"),
            task_trajectory=str(graph_dict.get("task_trajectory", "") or ""),
            label=graph_dict.get("label"),
            chain_of_states=state_chain,
            extra_fields=dict(extra_fields or {}),
        )

    def to_json(self) -> str:
        """
        負責執行 InteractionGraph 中的 to_json 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> InteractionGraph:
        """
        負責執行 InteractionGraph 中的 from_json 流程，依照 InteractionGraph 的流程需求處理 from_json 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            value: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InteractionGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return cls.from_dict(json.loads(value))

    @classmethod
    def from_mas_message(cls, value: dict[str, Any]) -> InteractionGraph:
        """
        負責執行 InteractionGraph 中的 from_mas_message 流程，依照 InteractionGraph 的流程需求處理 from_mas_message 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            value: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 InteractionGraph。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return cls.from_dict(value)
