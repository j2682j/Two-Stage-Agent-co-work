"""記憶子系統對外匯出的公開介面。"""

try:
    from .base import BaseMemory, MemoryConfig, MemoryItem
    from .manager import MemoryManager
    from .policy import (
        build_memory_records,
        should_write_final_memory,
        should_write_stage1_memory,
        should_write_stage2_memory,
    )
    from .storage.document_store import DocumentStore, SQLiteDocumentStore
    from .types.episodic import EpisodicMemory
    from .types.perceptual import PerceptualMemory
    from .types.semantic import SemanticMemory
    from .types.working import WorkingMemory
except ModuleNotFoundError:
    MemoryManager = None
    WorkingMemory = None
    EpisodicMemory = None
    SemanticMemory = None
    PerceptualMemory = None
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
    "MemoryManager",
    "Memory",
    "GlobalMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "PerceptualMemory",
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
