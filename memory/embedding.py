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
    """
    負責在 memory.embedding 中封裝 EmbeddingModel，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def encode(self, texts: Union[str, List[str]]):
        """
        負責執行 EmbeddingModel 中的 encode 流程，依照 EmbeddingModel 的流程需求處理 encode 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            texts: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """
        負責執行 EmbeddingModel 中的 dimension 流程，依照 EmbeddingModel 的流程需求處理 dimension 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        raise NotImplementedError


class LocalTransformerEmbedding(EmbeddingModel):
    """
    負責在 memory.embedding 中封裝 LocalTransformerEmbedding，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        model_name: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        負責執行 LocalTransformerEmbedding 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            model_name: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.model_name = model_name
        self._backend = None  # "st" 或 "hf"
        self._st_model = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._dimension = None
        self._load_backend()

    def _load_backend(self):
        # 優先 sentence-transformers
        """
        負責執行 LocalTransformerEmbedding 中的 _load_backend 流程，依照 LocalTransformerEmbedding 的流程需求處理 _load_backend 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 LocalTransformerEmbedding 中的 encode 流程，依照 LocalTransformerEmbedding 的流程需求處理 encode 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            texts: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 LocalTransformerEmbedding 中的 dimension 流程，依照 LocalTransformerEmbedding 的流程需求處理 dimension 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return int(self._dimension or 0)


class TFIDFEmbedding(EmbeddingModel):
    """
    負責在 memory.embedding 中封裝 TFIDFEmbedding，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        max_features: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, max_features: int = 1000):
        """
        負責執行 TFIDFEmbedding 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            max_features: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.max_features = max_features
        self._vectorizer = None
        self._is_fitted = False
        self._dimension = max_features
        self._init_vectorizer()

    def _init_vectorizer(self):
        """
        負責執行 TFIDFEmbedding 中的 _init_vectorizer 流程，依照 TFIDFEmbedding 的流程需求處理 _init_vectorizer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        except ImportError:
            raise ImportError("請安裝 scikit-learn: pip install scikit-learn")

    def fit(self, texts: List[str]):
        """
        負責執行 TFIDFEmbedding 中的 fit 流程，依照 TFIDFEmbedding 的流程需求處理 fit 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            texts: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._vectorizer.fit(texts)
        self._is_fitted = True
        self._dimension = len(self._vectorizer.get_feature_names_out())

    def encode(self, texts: Union[str, List[str]]):
        """
        負責執行 TFIDFEmbedding 中的 encode 流程，依照 TFIDFEmbedding 的流程需求處理 encode 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            texts: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        """
        負責執行 TFIDFEmbedding 中的 dimension 流程，依照 TFIDFEmbedding 的流程需求處理 dimension 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self._dimension


# ==============
# 工廠與回退
# ==============

def create_embedding_model(model_type: str = "local", **kwargs) -> EmbeddingModel:
    """
    負責執行 memory.embedding 中的 create_embedding_model 流程，建立記憶圖或任務記錄結構，供後續檢索、寫入與提示注入使用。
    
    Args:
        model_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
        **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 EmbeddingModel。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if model_type in ("local", "sentence_transformer", "huggingface"):
        return LocalTransformerEmbedding(**kwargs)
    elif model_type == "tfidf":
        return TFIDFEmbedding(**kwargs)
    else:
        raise ValueError(f"不支援的模型類型: {model_type}")


def create_embedding_model_with_fallback(preferred_type: str = "dashscope", **kwargs) -> EmbeddingModel:
    """
    負責執行 memory.embedding 中的 create_embedding_model_with_fallback 流程，建立記憶圖或任務記錄結構，供後續檢索、寫入與提示注入使用。
    
    Args:
        preferred_type: 記憶系統提供的檢索結果、寫入資料或操作介面。
        **kwargs: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 EmbeddingModel。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
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
    """
    負責執行 memory.embedding 中的 _ensure_env_loaded 流程，依照 memory.embedding 的流程需求處理 _ensure_env_loaded 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    load_dotenv(env_path, override=False)
    _ENV_LOADED = True


def _build_embedder() -> EmbeddingModel:
    """
    負責執行 memory.embedding 中的 _build_embedder 流程，依照 memory.embedding 的流程需求處理 _build_embedder 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 EmbeddingModel。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
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
    """
    負責執行 memory.embedding 中的 get_text_embedder 流程，依照 memory.embedding 的流程需求處理 get_text_embedder 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 EmbeddingModel。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder


def get_dimension(default: int = 384) -> int:
    """
    負責執行 memory.embedding 中的 get_dimension 流程，依照 memory.embedding 的流程需求處理 get_dimension 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        default: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 int。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    try:
        return int(getattr(get_text_embedder(), "dimension", default))
    except Exception:
        return int(default)


def refresh_embedder() -> EmbeddingModel:
    """
    負責執行 memory.embedding 中的 refresh_embedder 流程，依照 memory.embedding 的流程需求處理 refresh_embedder 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 EmbeddingModel。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder



