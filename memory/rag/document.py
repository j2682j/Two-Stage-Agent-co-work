"""文檔處理模組"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class Document:
    """文檔類"""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            # 基於內容生成ID
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()

@dataclass 
class DocumentChunk:
    """文檔塊類"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: int = 0
    
    def __post_init__(self):
        if self.chunk_id is None:
            # 基於文檔ID和塊索引生成ID
            chunk_content = f"{self.doc_id}_{self.chunk_index}_{self.content[:50]}"
            self.chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()

class DocumentProcessor:
    """文檔處理器"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " "]
    
    def process_document(self, document: Document) -> List[DocumentChunk]:
        """
        處理文檔，分割成塊
        
        Args:
            document: 輸入文檔
            
        Returns:
            文檔塊列表
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
        批量處理文檔
        
        Args:
            documents: 文檔列表
            
        Returns:
            所有文檔塊列表
        """
        all_chunks = []
        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        分割文字為塊
        
        Args:
            text: 輸入文字
            
        Returns:
            文字塊列表
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
        在指定范圍內尋找最佳分割點
        
        Args:
            text: 文字
            start: 開始位置
            end: 結束位置
            
        Returns:
            分割點位置，-1表示找不到
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
        合併小的文檔塊
        
        Args:
            chunks: 文檔塊列表
            max_length: 合併後的最大長度
            
        Returns:
            合併後的文檔塊列表
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
        過濾太短的文檔塊
        
        Args:
            chunks: 文檔塊列表
            min_length: 最小長度
            
        Returns:
            過濾後的文檔塊列表
        """
        return [chunk for chunk in chunks if len(chunk.content.strip()) >= min_length]
    
    def add_chunk_metadata(self, chunks: List[DocumentChunk], metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        為文檔塊添加額外元資料
        
        Args:
            chunks: 文檔塊列表
            metadata: 要添加的元資料
            
        Returns:
            更新後的文檔塊列表
        """
        for chunk in chunks:
            chunk.metadata.update(metadata)
        
        return chunks

def load_text_file(file_path: str, encoding: str = "utf-8") -> Document:
    """
    載入文字檔案為文檔
    
    Args:
        file_path: 檔案路徑
        encoding: 檔案編碼
        
    Returns:
        文檔對象
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
    建立文檔的便捷函式
    
    Args:
        content: 文檔內容
        **metadata: 元資料
        
    Returns:
        文檔對象
    """
    return Document(content=content, metadata=metadata)
