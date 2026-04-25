"""記憶子系統對外匯出的公開介面。"""

try:
    from .base import BaseMemory, MemoryConfig, MemoryItem
    from .lesson_rule import (
        LessonMatchResult,
        LessonRetrievalProfile,
        SemanticLesson,
        build_applicability,
        build_correction_checklist,
        build_retrieval_profile,
        build_semantic_lesson,
        build_tags,
        classify_error_type,
        classify_failure_mode,
        normalize_text,
        parse_semantic_lesson_memory,
        parse_semantic_lesson_text,
        score_semantic_lesson_match,
        select_relevant_semantic_lessons,
    )
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
    SemanticLesson = None
    LessonRetrievalProfile = None
    LessonMatchResult = None
    classify_error_type = None
    classify_failure_mode = None
    build_tags = None
    build_applicability = None
    build_correction_checklist = None
    build_retrieval_profile = None
    build_semantic_lesson = None
    normalize_text = None
    parse_semantic_lesson_memory = None
    parse_semantic_lesson_text = None
    score_semantic_lesson_match = None
    select_relevant_semantic_lessons = None
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
    "SemanticLesson",
    "LessonRetrievalProfile",
    "LessonMatchResult",
    "classify_error_type",
    "classify_failure_mode",
    "build_tags",
    "build_applicability",
    "build_correction_checklist",
    "build_retrieval_profile",
    "build_semantic_lesson",
    "normalize_text",
    "parse_semantic_lesson_memory",
    "parse_semantic_lesson_text",
    "score_semantic_lesson_match",
    "select_relevant_semantic_lessons",
    "should_write_stage1_memory",
    "should_write_stage2_memory",
    "should_write_final_memory",
    "build_memory_records",
]
