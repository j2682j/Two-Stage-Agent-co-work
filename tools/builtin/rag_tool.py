"""RAG工具 - 搜尋增強生成

為HelloAgents框架提供簡潔易用的RAG能力：
- 🔄 資料流程：使用者資料 → 文檔解析 → 向量化儲存 → 智慧搜尋 → LLM增強問答
- 多格式支援：PDF、Word、Excel、PPT、圖片、音頻、網頁等
- 🧠 智慧問答：自動搜尋相關內容，注入提示詞，生成準確答案
- 命名空間：支援多項目隔離，便於管理不同知識庫

使用範例：
```python
# 1. 初始化RAG工具
rag = RAGTool()

# 2. 添加文檔
rag.run({"action": "add_document", "file_path": "document.pdf"})

# 3. 智慧問答
answer = rag.run({"action": "ask", "question": "什么是機器學習？"})
```
"""

from typing import Dict, Any, List, Optional
import os
import time

from ..base import Tool, ToolParameter, tool_action
from memory.rag.pipeline import create_rag_pipeline
from core.llm import HelloAgentsLLM

class RAGTool(Tool):
    """RAG工具
    
    提供完整的 RAG 能力：
    - 添加多格式文檔（PDF、Office、圖片、音頻等）
    - 智慧搜尋與召回
    - LLM 增強問答
    - 知識庫管理
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        expandable: bool = False
    ):
        super().__init__(
            name="rag",
            description="RAG工具 - 支援多格式文檔搜尋增強生成，提供智慧問答能力",
            expandable=expandable
        )
        
        self.knowledge_base_path = knowledge_base_path
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        
        # 確保知識庫目錄存在
        os.makedirs(knowledge_base_path, exist_ok=True)
        
        # 初始化元件
        self._init_components()
    
    def _init_components(self):
        """初始化RAG元件"""
        try:
            # 初始化預設命名空間的 RAG 管道
            default_pipeline = create_rag_pipeline(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                rag_namespace=self.rag_namespace
            )
            self._pipelines[self.rag_namespace] = default_pipeline

            # 初始化 LLM 用於回答生成
            self.llm = HelloAgentsLLM()

            self.initialized = True
            print(f"[OK] RAG工具初始化成功: namespace={self.rag_namespace}, collection={self.collection_name}")
            
        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"[ERROR] RAG工具初始化失敗: {e}")

    def _get_pipeline(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """取得指定命名空間的 RAG 管道，若不存在則自動建立"""
        target_ns = namespace or self.rag_namespace
        if target_ns in self._pipelines:
            return self._pipelines[target_ns]

        pipeline = create_rag_pipeline(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            rag_namespace=target_ns
        )
        self._pipelines[target_ns] = pipeline
        return pipeline

    def run(self, parameters: Dict[str, Any]) -> str:
        """執行工具（非展開模式）

        Args:
            parameters: 工具參數字典，必須包含action參數

        Returns:
            執行結果字串
        """
        if not self.validate_parameters(parameters):
            return "[ERROR] 參數驗證失敗：缺少必需的參數"

        if not self.initialized:
            return f"[ERROR] RAG工具未正確初始化，請檢查設定: {getattr(self, 'init_error', '未知錯誤')}"

        action = parameters.get("action")

        # 根據action呼叫對應的方法，傳入提取的參數
        try:
            if action == "add_document":
                return self._add_document(
                    file_path=parameters.get("file_path"),
                    document_id=parameters.get("document_id"),
                    namespace=parameters.get("namespace", "default"),
                    chunk_size=parameters.get("chunk_size", 800),
                    chunk_overlap=parameters.get("chunk_overlap", 100)
                )
            elif action == "add_text":
                return self._add_text(
                    text=parameters.get("text"),
                    document_id=parameters.get("document_id"),
                    namespace=parameters.get("namespace", "default"),
                    chunk_size=parameters.get("chunk_size", 800),
                    chunk_overlap=parameters.get("chunk_overlap", 100)
                )
            elif action == "ask":
                question = parameters.get("question") or parameters.get("query")
                return self._ask(
                    question=question,
                    limit=parameters.get("limit", 5),
                    enable_advanced_search=parameters.get("enable_advanced_search", True),
                    include_citations=parameters.get("include_citations", True),
                    max_chars=parameters.get("max_chars", 1200),
                    namespace=parameters.get("namespace", "default")
                )
            elif action == "search":
                return self._search(
                    query=parameters.get("query") or parameters.get("question"),
                    limit=parameters.get("limit", 5),
                    min_score=parameters.get("min_score", 0.1),
                    enable_advanced_search=parameters.get("enable_advanced_search", True),
                    max_chars=parameters.get("max_chars", 1200),
                    include_citations=parameters.get("include_citations", True),
                    namespace=parameters.get("namespace", "default")
                )
            elif action == "stats":
                return self._get_stats(namespace=parameters.get("namespace", "default"))
            elif action == "clear":
                return self._clear_knowledge_base(
                    confirm=parameters.get("confirm", False),
                    namespace=parameters.get("namespace", "default")
                )
            else:
                return f"[ERROR] 不支援的操作: {action}"
        except Exception as e:
            return f"[ERROR] 執行操作 '{action}' 時發生錯誤: {str(e)}"

    def get_parameters(self) -> List[ToolParameter]:
        """取得工具參數定義 - Tool基類要求的介面"""
        return [
            # 核心操作參數
            ToolParameter(
                name="action",
                type="string",
                description="操作類型：add_document(添加文檔), add_text(添加文字), ask(智慧問答), search(搜尋), stats(統計), clear(清空)",
                required=True
            ),
            
            # 內容參數
            ToolParameter(
                name="file_path",
                type="string",
                description="文檔檔案路徑（支援PDF、Word、Excel、PPT、圖片、音頻等多種格式）",
                required=False
            ),
            ToolParameter(
                name="text",
                type="string",
                description="要添加的文字內容",
                required=False
            ),
            ToolParameter(
                name="question",
                type="string", 
                description="使用者問題（用於智慧問答）",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜尋查詢詞（用於基礎搜尋）",
                required=False
            ),
            
            # 可選設定參數
            ToolParameter(
                name="namespace",
                type="string",
                description="知識庫命名空間（用於隔離不同項目，預設：default）",
                required=False,
                default="default"
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="回傳結果數量（預設：5）",
                required=False,
                default=5
            ),
            ToolParameter(
                name="include_citations",
                type="boolean",
                description="是否包含引用來源（預設：true）",
                required=False,
                default=True
            )
        ]

    @tool_action("rag_add_document", "添加文檔到知識庫（支援PDF、Word、Excel、PPT、圖片、音頻等多種格式）")
    def _add_document(
        self,
        file_path: str,
        document_id: str = None,
        namespace: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> str:
        """添加文檔到知識庫

        Args:
            file_path: 文檔檔案路徑
            document_id: 文檔ID（可選）
            namespace: 知識庫命名空間（用於隔離不同項目）
            chunk_size: 分塊大小
            chunk_overlap: 分塊重疊大小

        Returns:
            執行結果
        """
        try:
            if not file_path or not os.path.exists(file_path):
                return f"[ERROR] 檔案不存在: {file_path}"
            
            pipeline = self._get_pipeline(namespace)
            t0 = time.time()

            chunks_added = pipeline["add_documents"](
                file_paths=[file_path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            t1 = time.time()
            process_ms = int((t1 - t0) * 1000)
            
            if chunks_added == 0:
                return f"[WARN] 未能從檔案解析內容: {os.path.basename(file_path)}"
            
            return (
                f"[OK] 文檔已添加到知識庫: {os.path.basename(file_path)}\n"
                f"[INFO] 分塊數量: {chunks_added}\n"
                f"[INFO] 處理時間: {process_ms}ms\n"
                f"[INFO] 命名空間: {pipeline.get('namespace', self.rag_namespace)}"
            )
            
        except Exception as e:
            return f"[ERROR] 添加文檔失敗: {str(e)}"
    
    @tool_action("rag_add_text", "添加文字到知識庫")
    def _add_text(
        self,
        text: str,
        document_id: str = None,
        namespace: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> str:
        """添加文字到知識庫

        Args:
            text: 要添加的文字內容
            document_id: 文檔ID（可選）
            namespace: 知識庫命名空間
            chunk_size: 分塊大小
            chunk_overlap: 分塊重疊大小

        Returns:
            執行結果
        """
        metadata = None
        try:
            if not text or not text.strip():
                return "[ERROR] 文字內容不能為空"
            
            # 建立臨時檔案
            document_id = document_id or f"text_{abs(hash(text)) % 100000}"
            tmp_path = os.path.join(self.knowledge_base_path, f"{document_id}.md")
            
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                pipeline = self._get_pipeline(namespace)
                t0 = time.time()

                chunks_added = pipeline["add_documents"](
                    file_paths=[tmp_path],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                t1 = time.time()
                process_ms = int((t1 - t0) * 1000)
                
                if chunks_added == 0:
                    return f"[WARN] 未能從文字生成有效分塊"
                
                return (
                    f"[OK] 文字已添加到知識庫: {document_id}\n"
                    f"[INFO] 分塊數量: {chunks_added}\n"
                    f"[INFO] 處理時間: {process_ms}ms\n"
                    f"[INFO] 命名空間: {pipeline.get('namespace', self.rag_namespace)}"
                )
                
            finally:
                # 清理臨時檔案
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            
        except Exception as e:
            return f"[ERROR] 添加文字失敗: {str(e)}"
    
    @tool_action("rag_search", "搜尋知識庫中的相關內容")
    def _search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
        enable_advanced_search: bool = True,
        max_chars: int = 1200,
        include_citations: bool = True,
        namespace: str = "default"
    ) -> str:
        """搜尋知識庫

        Args:
            query: 搜尋查詢詞
            limit: 回傳結果數量
            min_score: 最低相關度分數
            enable_advanced_search: 是否啟用高級搜尋（MQE、HyDE）
            max_chars: 每個結果最大字符數
            include_citations: 是否包含引用來源
            namespace: 知識庫命名空間

        Returns:
            搜尋結果
        """
        try:
            if not query or not query.strip():
                return "[ERROR] 搜尋查詢不能為空"
            
            # 使用統一 RAG 管道搜尋
            pipeline = self._get_pipeline(namespace)

            if enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=query,
                    top_k=limit,
                    enable_mqe=True,
                    enable_hyde=True,
                    score_threshold=min_score if min_score > 0 else None
                )
            else:
                results = pipeline["search"](
                    query=query,
                    top_k=limit,
                    score_threshold=min_score if min_score > 0 else None
                )
            
            if not results:
                return f"[INFO] 找不到與 '{query}' 相關的內容"
            
            # 格式化搜尋結果
            search_result = ["搜尋結果："]
            for i, result in enumerate(results, 1):
                meta = result.get("metadata", {})
                score = result.get("score", 0.0)
                content = meta.get("content", "")[:200] + "..."
                source = meta.get("source_path", "unknown")
                
                # 安全處理Unicode
                def clean_text(text):
                    try:
                        return str(text).encode('utf-8', errors='ignore').decode('utf-8')
                    except Exception:
                        return str(text)
                
                clean_content = clean_text(content)
                clean_source = clean_text(source)
                
                search_result.append(f"\n{i}. 文檔: **{clean_source}** (相似度: {score:.3f})")
                search_result.append(f"   {clean_content}")
                
                if include_citations and meta.get("heading_path"):
                    clean_heading = clean_text(str(meta['heading_path']))
                    search_result.append(f"   章節: {clean_heading}")
            
            return "\n".join(search_result)
            
        except Exception as e:
            return f"[ERROR] 搜尋失敗: {str(e)}"
    
    @tool_action("rag_ask", "基於知識庫進行智慧問答")
    def _ask(
        self,
        question: str,
        limit: int = 5,
        enable_advanced_search: bool = True,
        include_citations: bool = True,
        max_chars: int = 1200,
        namespace: str = "default"
    ) -> str:
        """智慧問答：搜尋 → 上下文注入 → LLM生成答案

        Args:
            question: 使用者問題
            limit: 搜尋結果數量
            enable_advanced_search: 是否啟用高級搜尋
            include_citations: 是否包含引用來源
            max_chars: 每個結果最大字符數
            namespace: 知識庫命名空間

        Returns:
            智慧問答結果

        核心流程:
        1. 解析使用者問題
        2. 智慧搜尋相關內容
        3. 建構上下文和提示詞
        4. LLM生成準確答案
        5. 添加引用來源
        """
        try:
            # 驗證問題
            if not question or not question.strip():
                return "[ERROR] 請提供要詢問的問題"

            user_question = question.strip()
            print(f"[INFO] 智慧問答: {user_question}")
            
            # 1. 搜尋相關內容
            pipeline = self._get_pipeline(namespace)
            search_start = time.time()
            
            if enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=user_question,
                    top_k=limit,
                    enable_mqe=True,
                    enable_hyde=True
                )
            else:
                results = pipeline["search"](
                    query=user_question,
                    top_k=limit
                )
            
            search_time = int((time.time() - search_start) * 1000)
            
            if not results:
                return (
                    f"🤔 抱歉，我在知識庫中沒有找到與「{user_question}」相關的資訊。\n\n"
                    f"[INFO] 建議：\n"
                    f"• 嘗試使用更簡潔的關鍵詞\n"
                    f"• 檢查是否已添加相關文檔\n"
                    f"• 使用 stats 操作查看知識庫狀態"
                )
            
            # 2. 智慧整理上下文
            context_parts = []
            citations = []
            total_score = 0
            
            for i, result in enumerate(results):
                meta = result.get("metadata", {})
                content = meta.get("content", "").strip()
                source = meta.get("source_path", "unknown")
                score = result.get("score", 0.0)
                total_score += score
                
                if content:
                    # 清理內容格式
                    cleaned_content = self._clean_content_for_context(content)
                    context_parts.append(f"片段 {i+1}：{cleaned_content}")
                    
                    if include_citations:
                        citations.append({
                            "index": i+1,
                            "source": os.path.basename(source),
                            "score": score
                        })
            
            # 3. 建構上下文（智慧截斷）
            context = "\n\n".join(context_parts)
            if len(context) > max_chars:
                # 智慧截斷，保持完整性
                context = self._smart_truncate_context(context, max_chars)
            
            # 4. 建構增強提示詞
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(user_question, context)
            
            enhanced_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 5. 呼叫 LLM 生成答案
            llm_start = time.time()
            answer = self.llm.invoke(enhanced_prompt)
            llm_time = int((time.time() - llm_start) * 1000)
            
            if not answer or not answer.strip():
                return "[ERROR] LLM未能生成有效答案，請稍后重試"
            
            # 6. 建構最終回答
            final_answer = self._format_final_answer(
                question=user_question,
                answer=answer.strip(),
                citations=citations if include_citations else None,
                search_time=search_time,
                llm_time=llm_time,
                avg_score=total_score / len(results) if results else 0
            )
            
            return final_answer
            
        except Exception as e:
            return f"[ERROR] 智慧問答失敗: {str(e)}\n[INFO] 請檢查知識庫狀態或稍后重試"
    
    def _clean_content_for_context(self, content: str) -> str:
        """清理內容用於上下文"""
        # 移除過多的換行和空格
        content = " ".join(content.split())
        # 截斷過長內容
        if len(content) > 300:
            content = content[:300] + "..."
        return content
    
    def _smart_truncate_context(self, context: str, max_chars: int) -> str:
        """智慧截斷上下文，保持段落完整性"""
        if len(context) <= max_chars:
            return context
        
        # 尋找最近的段落分隔符
        truncated = context[:max_chars]
        last_break = truncated.rfind("\n\n")
        
        if last_break > max_chars * 0.7:  # 如果斷點位置合理
            return truncated[:last_break] + "\n\n[...更多內容被截斷]"
        else:
            return truncated[:max_chars-20] + "...[內容被截斷]"
    
    def _build_system_prompt(self) -> str:
        """建構系統提示詞"""
        return (
            "你是一個專業的知識助手，具備以下能力：\n"
            "1. 📖 精準理解：仔細理解使用者問題的核心意圖\n"
            "2. 可信回答：嚴格基於提供的上下文資訊回答，不編造內容\n"
            "3. 資訊整合：從多個片段中提取關鍵資訊，形成完整答案\n"
            "4. 清晰表達：用簡潔明了的語言回答，適當使用結構化格式\n"
            "5. 🚫 誠實表達：如果上下文不足以回答問題，請坦誠說明\n\n"
            "回答格式要求：\n"
            "• 直接回答核心問題\n"
            "• 必要時使用要點或步驟\n"
            "• 引用關鍵原文時使用引號\n"
            "• 避免重復和冗余"
        )
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        """建構使用者提示詞"""
        return (
            f"請基於以下上下文資訊回答問題：\n\n"
            f"【問題】{question}\n\n"
            f"【相關上下文】\n{context}\n\n"
            f"【要求】請提供準確、有幫助的回答。如果上下文資訊不足，請說明需要什么額外資訊。"
        )
    
    def _format_final_answer(self, question: str, answer: str, citations: Optional[List[Dict]] = None, search_time: int = 0, llm_time: int = 0, avg_score: float = 0) -> str:
        """格式化最終答案"""
        result = [f"🤖 **智慧問答結果**\n"]
        result.append(answer)
        
        if citations:
            result.append("\n\n[INFO] **參考來源**")
            for citation in citations:
                score_emoji = "🟢" if citation["score"] > 0.8 else "🟡" if citation["score"] > 0.6 else "🔵"
                result.append(f"{score_emoji} [{citation['index']}] {citation['source']} (相似度: {citation['score']:.3f})")
        
        # 添加性能資訊（調試模式）
        result.append(f"\n⚡ 搜尋: {search_time}ms | 生成: {llm_time}ms | 平均相似度: {avg_score:.3f}")
        
        return "\n".join(result)

    @tool_action("rag_clear", "清空知識庫（危險操作，請謹慎使用）")
    def _clear_knowledge_base(self, confirm: bool = False, namespace: str = "default") -> str:
        """清空知識庫

        Args:
            confirm: 確認執行（必須設定為True）
            namespace: 知識庫命名空間

        Returns:
            執行結果
        """
        try:
            if not confirm:
                return (
                    "[WARN] 危險操作：清空知識庫將刪除所有資料！\n"
                    "請使用 confirm=true 參數確認執行。"
                )
            
            pipeline = self._get_pipeline(namespace)
            store = pipeline.get("store")
            namespace_id = pipeline.get("namespace", self.rag_namespace)
            success = store.clear_collection() if store else False
            
            if success:
                # 重新初始化該命名空間
                self._pipelines[namespace_id] = create_rag_pipeline(
                    qdrant_url=self.qdrant_url,
                    qdrant_api_key=self.qdrant_api_key,
                    collection_name=self.collection_name,
                    rag_namespace=namespace_id
                )
                return f"[OK] 知識庫已成功清空（命名空間：{namespace_id}）"
            else:
                return "[ERROR] 清空知識庫失敗"
            
        except Exception as e:
            return f"[ERROR] 清空知識庫失敗: {str(e)}"

    @tool_action("rag_stats", "取得知識庫統計資訊")
    def _get_stats(self, namespace: str = "default") -> str:
        """取得知識庫統計

        Args:
            namespace: 知識庫命名空間

        Returns:
            統計資訊
        """
        try:
            pipeline = self._get_pipeline(namespace)
            stats = pipeline["get_stats"]()
            
            stats_info = [
                "[INFO] **RAG 知識庫統計**",
                f"[INFO] 命名空間: {pipeline.get('namespace', self.rag_namespace)}",
                f"📋 集合名稱: {self.collection_name}",
                f"📂 儲存根路徑: {self.knowledge_base_path}"
            ]
            
            # 添加儲存統計
            if stats:
                store_type = stats.get("store_type", "unknown")
                total_vectors = (
                    stats.get("points_count") or 
                    stats.get("vectors_count") or 
                    stats.get("count") or 0
                )
                
                stats_info.extend([
                    f"📦 儲存類型: {store_type}",
                    f"[INFO] 文檔分塊數: {int(total_vectors)}",
                ])
                
                if "config" in stats:
                    config = stats["config"]
                    if isinstance(config, dict):
                        vector_size = config.get("vector_size", "unknown")
                        distance = config.get("distance", "unknown")
                        stats_info.extend([
                            f"🔢 向量維度: {vector_size}",
                            f"📎 距離度量: {distance}"
                        ])
            
            # 添加系統狀態
            stats_info.extend([
                "",
                "🟢 **系統狀態**",
                f"[OK] RAG 管道: {'正常' if self.initialized else '異常'}",
                f"[OK] LLM 連線: {'正常' if hasattr(self, 'llm') else '異常'}"
            ])
            
            return "\n".join(stats_info)
            
        except Exception as e:
            return f"[ERROR] 取得統計資訊失敗: {str(e)}"

    def get_relevant_context(self, query: str, limit: int = 3, max_chars: int = 1200, namespace: Optional[str] = None) -> str:
        """為查詢取得相關上下文
        
        這個方法可以被Agent呼叫來取得相關的知識庫上下文
        """
        try:
            if not query:
                return ""
            
            # 使用統一 RAG 管道搜尋
            pipeline = self._get_pipeline(namespace)
            results = pipeline["search"](
                query=query,
                top_k=limit
            )
            
            if not results:
                return ""
            
            # 合併上下文
            context_parts = []
            for result in results:
                content = result.get("metadata", {}).get("content", "")
                if content:
                    context_parts.append(content)
            
            merged_context = "\n\n".join(context_parts)
            
            # 限制長度
            if len(merged_context) > max_chars:
                merged_context = merged_context[:max_chars] + "..."
            
            return merged_context
            
        except Exception as e:
            return f"取得上下文失敗: {str(e)}"
    
    def batch_add_texts(self, texts: List[str], document_ids: Optional[List[str]] = None, chunk_size: int = 800, chunk_overlap: int = 100, namespace: Optional[str] = None) -> str:
        """批量添加文字"""
        try:
            if not texts:
                return "[ERROR] 文字列表不能為空"
            
            if document_ids and len(document_ids) != len(texts):
                return "[ERROR] 文字數量和文檔ID數量不匹配"
            
            pipeline = self._get_pipeline(namespace)
            t0 = time.time()
            
            total_chunks = 0
            successful_files = []
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                    
                doc_id = document_ids[i] if document_ids else f"batch_text_{i}"
                tmp_path = os.path.join(self.knowledge_base_path, f"{doc_id}.md")
                
                try:
                    with open(tmp_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    chunks_added = pipeline["add_documents"](
                        file_paths=[tmp_path],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    total_chunks += chunks_added
                    successful_files.append(doc_id)
                    
                finally:
                    # 清理臨時檔案
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
            
            t1 = time.time()
            process_ms = int((t1 - t0) * 1000)
            
            return (
                f"[OK] 批量添加完成\n"
                f"[INFO] 成功檔案: {len(successful_files)}/{len(texts)}\n"
                f"[INFO] 總分塊數: {total_chunks}\n"
                f"[INFO] 處理時間: {process_ms}ms"
            )
            
        except Exception as e:
            return f"[ERROR] 批量添加失敗: {str(e)}"
    
    def clear_all_namespaces(self) -> str:
        """清空目前工具管理的所有命名空間資料"""
        try:
            for ns, pipeline in self._pipelines.items():
                store = pipeline.get("store")
                if store:
                    store.clear_collection()
            self._pipelines.clear()
            # 重新初始化預設命名空間
            self._init_components()
            return "[OK] 所有命名空間資料已清空並重新初始化"
        except Exception as e:
            return f"[ERROR] 清空所有命名空間失敗: {str(e)}"
    
    # ========================================
    # 便捷介面方法（簡化使用者呼叫）
    # ========================================
    
    def add_document(self, file_path: str, namespace: str = "default") -> str:
        """便捷方法：添加單個文檔"""
        return self.run({
            "action": "add_document",
            "file_path": file_path,
            "namespace": namespace
        })
    
    def add_text(self, text: str, namespace: str = "default", document_id: str = None) -> str:
        """便捷方法：添加文字內容"""
        return self.run({
            "action": "add_text",
            "text": text,
            "namespace": namespace,
            "document_id": document_id
        })
    
    def ask(self, question: str, namespace: str = "default", **kwargs) -> str:
        """便捷方法：智慧問答"""
        params = {
            "action": "ask",
            "question": question,
            "namespace": namespace
        }
        params.update(kwargs)
        return self.run(params)
    
    def search(self, query: str, namespace: str = "default", **kwargs) -> str:
        """便捷方法：搜尋知識庫"""
        params = {
            "action": "search",
            "query": query,
            "namespace": namespace
        }
        params.update(kwargs)
        return self.run(params)
    
    def add_documents_batch(self, file_paths: List[str], namespace: str = "default") -> str:
        """批量添加多個文檔"""
        if not file_paths:
            return "[ERROR] 檔案路徑列表不能為空"
        
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        start_time = time.time()
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"📄 處理文檔 {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            try:
                result = self.add_document(file_path, namespace)
                if "[OK]" in result:
                    successful += 1
                    # 提取分塊數量
                    if "分塊數量:" in result:
                        chunks = int(result.split("分塊數量: ")[1].split("\n")[0])
                        total_chunks += chunks
                else:
                    failed += 1
                    results.append(f"[ERROR] {os.path.basename(file_path)}: 處理失敗")
            except Exception as e:
                failed += 1
                results.append(f"[ERROR] {os.path.basename(file_path)}: {str(e)}")
        
        process_time = int((time.time() - start_time) * 1000)
        
        summary = [
            "[INFO] **批量處理完成**",
            f"[OK] 成功: {successful}/{len(file_paths)} 個文檔",
            f"[INFO] 總分塊數: {total_chunks}",
            f"[INFO] 總耗時: {process_time}ms",
            f"[INFO] 命名空間: {namespace}"
        ]
        
        if failed > 0:
            summary.append(f"[ERROR] 失敗: {failed} 個文檔")
            summary.append("\n**失敗詳情:**")
            summary.extend(results)
        
        return "\n".join(summary)
    
    def add_texts_batch(self, texts: List[str], namespace: str = "default", document_ids: Optional[List[str]] = None) -> str:
        """批量添加多個文字"""
        if not texts:
            return "[ERROR] 文字列表不能為空"
        
        if document_ids and len(document_ids) != len(texts):
            return "[ERROR] 文字數量和文檔ID數量不匹配"
        
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        start_time = time.time()
        
        for i, text in enumerate(texts):
            doc_id = document_ids[i] if document_ids else f"batch_text_{i+1}"
            print(f"[INFO] 處理文字 {i+1}/{len(texts)}: {doc_id}")
            
            try:
                result = self.add_text(text, namespace, doc_id)
                if "[OK]" in result:
                    successful += 1
                    # 提取分塊數量
                    if "分塊數量:" in result:
                        chunks = int(result.split("分塊數量: ")[1].split("\n")[0])
                        total_chunks += chunks
                else:
                    failed += 1
                    results.append(f"[ERROR] {doc_id}: 處理失敗")
            except Exception as e:
                failed += 1
                results.append(f"[ERROR] {doc_id}: {str(e)}")
        
        process_time = int((time.time() - start_time) * 1000)
        
        summary = [
            "[INFO] **批量文字處理完成**",
            f"[OK] 成功: {successful}/{len(texts)} 個文字",
            f"[INFO] 總分塊數: {total_chunks}",
            f"[INFO] 總耗時: {process_time}ms",
            f"[INFO] 命名空間: {namespace}"
        ]
        
        if failed > 0:
            summary.append(f"[ERROR] 失敗: {failed} 個文字")
            summary.append("\n**失敗詳情:**")
            summary.extend(results)

        return "\n".join(summary)

