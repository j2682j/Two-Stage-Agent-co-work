from __future__ import annotations

import copy
import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MemoryRetrievalCacheKey:
    benchmark: str
    task_id: str
    stage: str
    injection_target: str
    source: str
    question_hash: str
    attachment_type: str
    limit: int

    def short_id(self) -> str:
        raw = "|".join(
            [
                self.benchmark,
                self.task_id,
                self.stage,
                self.injection_target,
                self.source,
                self.question_hash,
                self.attachment_type,
                str(self.limit),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]


class MemoryRetrievalCache:
    """MemoryRetrievalCache 類別。
    
    封裝此類別負責的狀態、設定與操作流程。
    """

    def __init__(self) -> None:
        self._values: dict[MemoryRetrievalCacheKey, dict[str, Any]] = {}
        self._inflight: dict[MemoryRetrievalCacheKey, threading.Event] = {}
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._inflight.clear()

    def get_or_compute(
        self,
        key: MemoryRetrievalCacheKey,
        compute: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        while True:
            with self._lock:
                cached = self._values.get(key)
                if cached is not None:
                    return copy.deepcopy(cached), True

                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    owner = True
                else:
                    owner = False

            if owner:
                try:
                    value = compute()
                    with self._lock:
                        self._values[key] = copy.deepcopy(value)
                        return copy.deepcopy(value), False
                finally:
                    with self._lock:
                        done = self._inflight.pop(key, None)
                        if done is not None:
                            done.set()

            event.wait()


def build_question_hash(question: str) -> str:
    normalized = " ".join(str(question or "").strip().split())
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:16]
