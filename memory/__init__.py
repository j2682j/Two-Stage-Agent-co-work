"""memory.__init__ 模組。

提供此模組相關的資料結構、流程輔助或整合邏輯。
"""

try:
    from .base import BaseMemory, MemoryConfig, MemoryItem
    from .policy import (
        build_memory_records,
        should_write_final_memory,
        should_write_stage1_memory,
        should_write_stage2_memory,
    )
    from .storage.document_store import DocumentStore, SQLiteDocumentStore
except ModuleNotFoundError:
    DocumentStore = None
    SQLiteDocumentStore = None
    BaseMemory = None
    MemoryConfig = None
    MemoryItem = None
    build_memory_records = None
    should_write_final_memory = None
    should_write_stage1_memory = None
    should_write_stage2_memory = None

__all__ = [
    "DocumentStore",
    "SQLiteDocumentStore",
    "MemoryItem",
    "MemoryConfig",
    "BaseMemory",
    "should_write_stage1_memory",
    "should_write_stage2_memory",
    "should_write_final_memory",
    "build_memory_records",
]
