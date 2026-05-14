"""文檔處理模組"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class Document:
    """
    負責在 memory.rag.document 中封裝 Document，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        """
        負責執行 Document 中的 __post_init__ 流程，依照 Document 的流程需求處理 __post_init__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.doc_id is None:
            # 基於內容生成ID
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()

@dataclass 
class DocumentChunk:
    """
    負責在 memory.rag.document 中封裝 DocumentChunk，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: int = 0
    
    def __post_init__(self):
        """
        負責執行 DocumentChunk 中的 __post_init__ 流程，依照 DocumentChunk 的流程需求處理 __post_init__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if self.chunk_id is None:
            # 基於文檔ID和塊索引生成ID
            chunk_content = f"{self.doc_id}_{self.chunk_index}_{self.content[:50]}"
            self.chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()

class DocumentProcessor:
    """
    負責在 memory.rag.document 中封裝 DocumentProcessor，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        chunk_size: 記憶系統提供的檢索結果、寫入資料或操作介面。
        chunk_overlap: 記憶系統提供的檢索結果、寫入資料或操作介面。
        separators: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        負責執行 DocumentProcessor 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            chunk_size: 記憶系統提供的檢索結果、寫入資料或操作介面。
            chunk_overlap: 記憶系統提供的檢索結果、寫入資料或操作介面。
            separators: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " "]
    
    def process_document(self, document: Document) -> List[DocumentChunk]:
        """
        負責執行 DocumentProcessor 中的 process_document 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
        
        Args:
            document: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[DocumentChunk]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        chunks = self._split_text(document.content)
        
        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            # 建立塊的元資料
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "doc_id": document.doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "processed_at": datetime.now().isoformat()
            })
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                doc_id=document.doc_id,
                chunk_index=i
            )
            document_chunks.append(chunk)
        
        return document_chunks
    
    def process_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        負責執行 DocumentProcessor 中的 process_documents 流程，整理呼叫端傳入的資料，清理格式並轉換為後續流程可使用的內容。
        
        Args:
            documents: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[DocumentChunk]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        all_chunks = []
        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        負責執行 DocumentProcessor 中的 _split_text 流程，依照 DocumentProcessor 的流程需求處理 _split_text 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 確定塊的結束位置
            end = start + self.chunk_size
            
            if end >= len(text):
                # 最後一塊
                chunks.append(text[start:])
                break
            
            # 尋找合適的分割點
            split_point = self._find_split_point(text, start, end)
            
            if split_point == -1:
                # 沒找到合適的分割點，強制分割
                split_point = end
            
            chunks.append(text[start:split_point])
            
            # 計算下一塊的開始位置（考慮重疊）
            start = max(start + 1, split_point - self.chunk_overlap)
        
        return chunks
    
    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """
        負責執行 DocumentProcessor 中的 _find_split_point 流程，依照 DocumentProcessor 的流程需求處理 _find_split_point 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 記憶系統提供的檢索結果、寫入資料或操作介面。
            start: 記憶系統提供的檢索結果、寫入資料或操作介面。
            end: 記憶系統提供的檢索結果、寫入資料或操作介面。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 從后往前尋找分隔符
        for separator in self.separators:
            # 在end附近尋找分隔符
            search_start = max(start, end - 100)  # 在最後100個字符中尋找
            
            for i in range(end - len(separator), search_start - 1, -1):
                if text[i:i + len(separator)] == separator:
                    return i + len(separator)
        
        return -1
    
    def merge_chunks(self, chunks: List[DocumentChunk], max_length: int = 2000) -> List[DocumentChunk]:
        """
        負責執行 DocumentProcessor 中的 merge_chunks 流程，依照 DocumentProcessor 的流程需求處理 merge_chunks 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            chunks: 記憶系統提供的檢索結果、寫入資料或操作介面。
            max_length: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[DocumentChunk]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not chunks:
            return []
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # 檢查是否可以合併
            combined_length = len(current_chunk.content) + len(next_chunk.content)
            
            if (combined_length <= max_length and 
                current_chunk.doc_id == next_chunk.doc_id):
                # 合併塊
                current_chunk.content += "\n" + next_chunk.content
                current_chunk.metadata["total_chunks"] = current_chunk.metadata.get("total_chunks", 1) + 1
            else:
                # 不能合併，保存目前塊
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        # 添加最後一個塊
        merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def filter_chunks(self, chunks: List[DocumentChunk], min_length: int = 50) -> List[DocumentChunk]:
        """
        負責執行 DocumentProcessor 中的 filter_chunks 流程，依照 DocumentProcessor 的流程需求處理 filter_chunks 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            chunks: 記憶系統提供的檢索結果、寫入資料或操作介面。
            min_length: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[DocumentChunk]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [chunk for chunk in chunks if len(chunk.content.strip()) >= min_length]
    
    def add_chunk_metadata(self, chunks: List[DocumentChunk], metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        負責執行 DocumentProcessor 中的 add_chunk_metadata 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            chunks: 記憶系統提供的檢索結果、寫入資料或操作介面。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[DocumentChunk]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        for chunk in chunks:
            chunk.metadata.update(metadata)
        
        return chunks

def load_text_file(file_path: str, encoding: str = "utf-8") -> Document:
    """
    負責執行 memory.rag.document 中的 load_text_file 流程，讀取本地或外部資料來源並轉換成系統可處理的格式。
    
    Args:
        file_path: 要讀取或寫入的檔案或目錄路徑。
        encoding: 記憶系統提供的檢索結果、寫入資料或操作介面。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Document。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()
    
    metadata = {
        "source": file_path,
        "type": "text_file",
        "loaded_at": datetime.now().isoformat()
    }
    
    return Document(content=content, metadata=metadata)

def create_document(content: str, **metadata) -> Document:
    """
    負責執行 memory.rag.document 中的 create_document 流程，建立記憶圖或任務記錄結構，供後續檢索、寫入與提示注入使用。
    
    Args:
        content: 記憶系統提供的檢索結果、寫入資料或操作介面。
        **metadata: 目前流程所需的上下文、狀態或附加資訊。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 Document。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return Document(content=content, metadata=metadata)
