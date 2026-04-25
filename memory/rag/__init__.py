"""RAG (搜尋增強生成) 模組

合併了 GraphRAG 能力：
- loader：檔案載入/分塊（含PDF、語言標注、去重）
- embedding/cache：嵌入與SQLite快取，預設哈希回退
- vector search：Qdrant召回
- rank/merge：融合排序與片段合併
"""

# 說明：原先的 .embeddings 已合併到上級目錄的 memory/embedding.py
# 這裡做相容匯出，避免歷史引用報錯。
from ..embedding import (
    EmbeddingModel,
    LocalTransformerEmbedding,
    TFIDFEmbedding,
    create_embedding_model,
    create_embedding_model_with_fallback,
)
from .document import Document, DocumentProcessor
from .pipeline import (
    load_and_chunk_texts,
    build_graph_from_chunks,
    index_chunks,
    embed_query,
    search_vectors,
    rank,
    merge_snippets,
    rerank_with_cross_encoder,
    expand_neighbors_from_pool,
    compute_graph_signals_from_pool,
    merge_snippets_grouped,
    search_vectors_expanded,
    compress_ranked_items,
    tldr_summarize,
)

# 相容舊類名（歷史代碼中可能從此處匯入）
SentenceTransformerEmbedding = LocalTransformerEmbedding
HuggingFaceEmbedding = LocalTransformerEmbedding

__all__ = [
    "EmbeddingModel",
    "LocalTransformerEmbedding",
    "SentenceTransformerEmbedding",  # 相容別名
    "HuggingFaceEmbedding",          # 相容別名
    "TFIDFEmbedding",
    "create_embedding_model",
    "create_embedding_model_with_fallback",
    "Document",
    "DocumentProcessor",
    "load_and_chunk_texts",
    "build_graph_from_chunks",
    "index_chunks",
    "embed_query",
    "search_vectors",
    "rank",
    "merge_snippets",
    "rerank_with_cross_encoder",
    "expand_neighbors_from_pool",
    "compute_graph_signals_from_pool",
    "merge_snippets_grouped",
    "search_vectors_expanded",
    "compress_ranked_items",
    "tldr_summarize",
]
