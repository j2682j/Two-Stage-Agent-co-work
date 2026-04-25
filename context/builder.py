"""????????? GSSC ???"""


from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import tiktoken
import math

from ..core.message import Message
from ..tools import MemoryTool, RAGTool


@dataclass
class ContextPacket:
    """上下文資訊包"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0  # 0.0-1.0
    
    def __post_init__(self):
        """自動計算token 數"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """上下文建構設定"""
    max_tokens: int = 8000  # 總預算
    reserve_ratio: float = 0.15  # 生成餘量（10-20%）
    min_relevance: float = 0.3  # 最小相關性閾值
    enable_mmr: bool = True  # 啟用最大邊際相關性（多樣性）
    mmr_lambda: float = 0.7  # MMR平衡參數（0=純多樣性, 1=純相關性）
    system_prompt_template: str = ""  # 系統提示範本
    enable_compression: bool = True  # 啟用壓縮
    
    def get_available_tokens(self) -> int:
        """取得可用 token 預算（扣除餘量）"""
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """上下文建構器 - GSSC流水線
    
    使用範例：
    ```python
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )
    
    context = builder.build(
        user_query="使用者問題",
        conversation_history=[...],
        system_instructions="系統指令"
    )
    ```
    """
    
    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        rag_tool: Optional[RAGTool] = None,
        config: Optional[ContextConfig] = None
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None
    ) -> str:
        """建構完整上下文
        
        Args:
            user_query: 使用者查詢
            conversation_history: 對話歷史
            system_instructions: 系統指令
            additional_packets: 額外的上下文包
            
        Returns:
            結構化上下文字串
        """
        # 1. Gather: 收集候選資訊
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or []
        )
        
        # 2. Select: 篩選與排序
        selected_packets = self._select(packets, user_query)
        
        # 3. Structure: 組織成結構化模板
        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
            system_instructions=system_instructions
        )
        
        # 4. Compress: 壓縮與規範化（如果超預算）
        final_context = self._compress(structured_context)
        
        return final_context
    
    def _gather(
        self,
        user_query: str,
        conversation_history: List[Message],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket]
    ) -> List[ContextPacket]:
        """Gather: 收集候選資訊"""
        packets = []
        
        # P0: 系統指令（強約束）
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"}
            ))
        
        # P1: 從記憶中取得任務狀態與關鍵結論
        if self.memory_tool:
            try:
                # 搜尋任務狀態相關記憶
                state_results = self.memory_tool.execute(
                    "search",
                    query="(任務狀態 OR 子目標 OR 結論 OR 阻塞)",
                    min_importance=0.7,
                    limit=5
                )
                if state_results and "找不到" not in state_results:
                    packets.append(ContextPacket(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"}
                    ))
                
                # 搜尋與目前查詢相關的記憶
                related_results = self.memory_tool.execute(
                    "search",
                    query=user_query,
                    limit=5
                )
                if related_results and "找不到" not in related_results:
                    packets.append(ContextPacket(
                        content=related_results,
                        metadata={"type": "related_memory"}
                    ))
            except Exception as e:
                print(f"⚠️ 記憶搜尋失敗: {e}")
        
        # P2: 從RAG中取得事實依據
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if rag_results and "找不到" not in rag_results and "錯誤" not in rag_results:
                    packets.append(ContextPacket(
                        content=rag_results,
                        metadata={"type": "knowledge_base"}
                    ))
            except Exception as e:
                print(f"⚠️ RAG搜尋失敗: {e}")
        
        # P3: 對話歷史（輔助資料）
        if conversation_history:
            # 只保留最近N條
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            packets.append(ContextPacket(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)}
            ))
        
        # 添加額外包
        packets.extend(additional_packets)
        
        return packets
    
    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str
    ) -> List[ContextPacket]:
        """Select: 基於分數與預算的篩選"""
        # 1) 計算相關性（關鍵詞重疊）
        query_tokens = set(user_query.lower().split())
        for packet in packets:
            content_tokens = set(packet.content.lower().split())
            if len(query_tokens) > 0:
                overlap = len(query_tokens & content_tokens)
                packet.relevance_score = overlap / len(query_tokens)
            else:
                packet.relevance_score = 0.0
        
        # 2) 計算新近性（指數衰減）
        def recency_score(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600  # 1小時時間尺度，可暴露到設定
            return math.exp(-delta / tau)
        
        # 3) 計算復合分：0.7*相關性 + 0.3*新近性
        scored_packets: List[Tuple[float, ContextPacket]] = []
        for p in packets:
            rec = recency_score(p.timestamp)
            score = 0.7 * p.relevance_score + 0.3 * rec
            scored_packets.append((score, p))
        
        # 4) 系統指令單獨拿出，固定納入
        system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "instructions"]
        remaining = [p for (s, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
                     if p.metadata.get("type") != "instructions"]
        
        # 5) 依據 min_relevance 過濾（對非系統包）
        filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]
        
        # 6) 按預算填充
        available_tokens = self.config.get_available_tokens()
        selected: List[ContextPacket] = []
        used_tokens = 0
        
        # 先放入系統指令（不排序）
        for p in system_packets:
            if used_tokens + p.token_count <= available_tokens:
                selected.append(p)
                used_tokens += p.token_count
        
        # 再按分數加入其余
        for p in filtered:
            if used_tokens + p.token_count > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token_count
        
        return selected
    
    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str,
        system_instructions: Optional[str]
    ) -> str:
        """Structure: 組織成結構化上下文模板"""
        sections = []
        
        # [Role & Policies] - 系統指令
        p0_packets = [p for p in selected_packets if p.metadata.get("type") == "instructions"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)
        
        # [Task] - 目前任務
        sections.append(f"[Task]\n使用者問題：{user_query}")
        
        # [State] - 任務狀態
        p1_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if p1_packets:
            state_section = "[State]\n關鍵進展與待解問題：\n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)
        
        # [Evidence] - 事實依據
        p2_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\n事實與引用：\n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)
        
        # [Context] - 輔助資料（歷史等）
        p3_packets = [p for p in selected_packets if p.metadata.get("type") == "history"]
        if p3_packets:
            context_section = "[Context]\n對話歷史與背景：\n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)
        
        # [Output] - 輸出約束
        output_section = """[Output]
                            請按以下格式回答：
                            1. 結論（簡潔明確）
                            2. 依據（列出支撐證據及來源）
                            3. 風險與假設（如有）
                            4. 下一步行動建議（如適用）"""
        sections.append(output_section)
        
        return "\n\n".join(sections)
    
    def _compress(self, context: str) -> str:
        """Compress: 壓縮與規範化"""
        if not self.config.enable_compression:
            return context
        
        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()
        
        if current_tokens <= available_tokens:
            return context
        
        # 簡單截斷策略（保留前N個token）
        # 實際應用中可用LLM做高保真摘要
        print(f"⚠️ 上下文超預算 ({current_tokens} > {available_tokens})，執行截斷")
        
        # 按段落截斷，保留結構
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens
        
        return "\n".join(compressed_lines)


def count_tokens(text: str) -> int:
    """計算文字token 數（使用tiktoken）"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 降級方案：粗略估算（1 token ≈ 4 字符）
        return len(text) // 4
