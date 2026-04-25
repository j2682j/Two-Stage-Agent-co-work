"""統一嵌入模組（實現 + 提供器）

說明（中文）：
- 提供統一的文字嵌入介面與多實現：本地Transformer、TF-IDF兜底。
- 暴露 get_text_embedder()/get_dimension()/refresh_embedder() 供各記憶類型統一使用。
- 通過環境變數優先順序：local > tfidf。

環境變數：
- EMBED_MODEL_TYPE: "local" | "tfidf"
- EMBED_MODEL_NAME: 模型名稱（local預設 sentence-transformers/all-MiniLM-L6-v2）
- EMBED_API_KEY: Embedding API Key（統一命名）
- EMBED_BASE_URL: Embedding Base URL（統一命名，可選）
"""

from typing import List, Union, Optional
import threading
import os
import numpy as np
from pathlib import Path

from dotenv import load_dotenv


# ==============
# 抽象與實現
# ==============

class EmbeddingModel:
    """嵌入模型基類（最小介面）"""

    def encode(self, texts: Union[str, List[str]]):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class LocalTransformerEmbedding(EmbeddingModel):
    """本地Transformer嵌入（優先 sentence-transformers，缺失回退 transformers+torch）"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._backend = None  # "st" 或 "hf"
        self._st_model = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._dimension = None
        self._load_backend()

    def _load_backend(self):
        # 優先 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            test_vec = self._st_model.encode("test_text")
            self._dimension = len(test_vec)
            self._backend = "st"
            return
        except Exception:
            self._st_model = None

        # 回退 transformers
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            with torch.no_grad():
                inputs = self._hf_tokenizer("test_text", return_tensors="pt", padding=True, truncation=True)
                outputs = self._hf_model(**inputs)
                test_embedding = outputs.last_hidden_state.mean(dim=1)
                self._dimension = int(test_embedding.shape[1])
            self._backend = "hf"
            return
        except Exception:
            self._hf_tokenizer = None
            self._hf_model = None

        raise ImportError("找不到可用的本地嵌入後端，請安裝 sentence-transformers 或 transformers+torch")

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        if self._backend == "st":
            vecs = self._st_model.encode(inputs)
            if hasattr(vecs, "tolist"):
                vecs = [v for v in vecs]
        else:
            import torch
            tokenized = self._hf_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._hf_model(**tokenized)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vecs = [v for v in embeddings]

        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)


class TFIDFEmbedding(EmbeddingModel):
    """TF-IDF 在無深度模型時保證可用）"""

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self._vectorizer = None
        self._is_fitted = False
        self._dimension = max_features
        self._init_vectorizer()

    def _init_vectorizer(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        except ImportError:
            raise ImportError("請安裝 scikit-learn: pip install scikit-learn")

    def fit(self, texts: List[str]):
        self._vectorizer.fit(texts)
        self._is_fitted = True
        self._dimension = len(self._vectorizer.get_feature_names_out())

    def encode(self, texts: Union[str, List[str]]):
        if not self._is_fitted:
            raise ValueError("TF-IDF模型未訓練，請先呼叫fit()方法")
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        tfidf_matrix = self._vectorizer.transform(texts)
        embeddings = tfidf_matrix.toarray()
        if single:
            return embeddings[0]
        return [e for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


# ==============
# 工廠與回退
# ==============

def create_embedding_model(model_type: str = "local", **kwargs) -> EmbeddingModel:
    """建立嵌入模型實例

    model_type: "dashscope" | "local" | "tfidf"
    kwargs: model_name, api_key
    """
    if model_type in ("local", "sentence_transformer", "huggingface"):
        return LocalTransformerEmbedding(**kwargs)
    elif model_type == "tfidf":
        return TFIDFEmbedding(**kwargs)
    else:
        raise ValueError(f"不支援的模型類型: {model_type}")


def create_embedding_model_with_fallback(preferred_type: str = "dashscope", **kwargs) -> EmbeddingModel:
    """帶回退的建立：local -> tfidf"""
    if preferred_type in ("sentence_transformer", "huggingface"):
        preferred_type = "local"
    fallback = ["local", "tfidf"]
    # 將首選放最前
    if preferred_type in fallback:
        fallback.remove(preferred_type)
        fallback.insert(0, preferred_type)
    for t in fallback:
        try:
            return create_embedding_model(t, **kwargs)
        except Exception:
            continue
    raise RuntimeError("所有嵌入模型都不可用，請安裝依賴或檢查設定")


# ==================
# Provider（單例）
# ==================

_lock = threading.RLock()
_embedder: Optional[EmbeddingModel] = None
_ENV_LOADED = False


def _ensure_env_loaded():
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    load_dotenv(env_path, override=False)
    _ENV_LOADED = True


def _build_embedder() -> EmbeddingModel:
    _ensure_env_loaded()
    preferred = (os.getenv("EMBED_MODEL_TYPE") or "local").strip()
    # 根據提供商選擇預設模型
    default_model = "sentence-transformers/all-MiniLM-L6-v2"
    model_name = (os.getenv("EMBED_MODEL_NAME") or default_model).strip()
    kwargs = {}
    if model_name:
        kwargs["model_name"] = model_name
    # 僅使用統一命名
    api_key = os.getenv("EMBED_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    base_url = os.getenv("EMBED_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return create_embedding_model_with_fallback(preferred_type=preferred, **kwargs)


def get_text_embedder() -> EmbeddingModel:
    """取得全局共享的文字嵌入實例（執行緒安全單例）"""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder


def get_dimension(default: int = 384) -> int:
    """取得統一向量維度（失敗回退預設值）"""
    try:
        return int(getattr(get_text_embedder(), "dimension", default))
    except Exception:
        return int(default)


def refresh_embedder() -> EmbeddingModel:
    """強制重建嵌入實例（可用於動態切換環境變數）"""
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder



