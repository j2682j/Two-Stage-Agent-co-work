"""NoteTool - 結構化筆記工具

為Agent提供結構化筆記能力，支援：
- 建立/讀取/更新/刪除筆記
- 按類型組織（任務狀態、結論、阻塞項、行動計劃等）
- 持久化儲存（Markdown格式，帶YAML前置元資料）
- 搜尋與過濾
- 與MemoryTool集成（可選）

使用場景：
- 長時程任務的狀態跟蹤
- 關鍵結論與依賴紀錄
- 待辦事項與行動計劃
- 項目知識沉淀

筆記格式範例：
```markdown
---
id: note_20250118_120000_0
title: 項目進展
type: task_state
tags: [milestone, phase1]
created_at: 2025-01-18T12:00:00
updated_at: 2025-01-18T12:00:00
---

# 項目進展

已完成需求分析，下一步：設計方案

## 關鍵里程碑
- [x] 需求收集
- [ ] 方案設計
```
"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
import re

from ..base import Tool, ToolParameter, tool_action


class NoteTool(Tool):
    """筆記工具
    
    為Agent提供結構化筆記管理能力，支援多種筆記類型：
    - task_state: 任務狀態
    - conclusion: 關鍵結論
    - blocker: 阻塞項
    - action: 行動計劃
    - reference: 參考資料
    - general: 通用筆記
    
    使用範例：
    ```python
    note_tool = NoteTool(workspace="./project_notes")
    
    # 建立筆記
    note_tool.run({
        "action": "create",
        "title": "項目進展",
        "content": "已完成需求分析，下一步：設計方案",
        "note_type": "task_state",
        "tags": ["milestone", "phase1"]
    })
    
    # 讀取筆記
    notes = note_tool.run({"action": "list", "note_type": "task_state"})
    ```
    """
    
    def __init__(
        self,
        workspace: str = "./notes",
        auto_backup: bool = True,
        max_notes: int = 1000,
        expandable: bool = False
    ):
        super().__init__(
            name="note",
            description="筆記工具 - 建立、讀取、更新、刪除結構化筆記，支援任務狀態、結論、阻塞項等類型",
            expandable=expandable
        )
        
        self.workspace = Path(workspace)
        self.auto_backup = auto_backup
        self.max_notes = max_notes
        
        # 確保工作目錄存在
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # 筆記索引檔案
        self.index_file = self.workspace / "notes_index.json"
        self._load_index()
    
    def _load_index(self):
        """載入筆記索引"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.notes_index = json.load(f)
        else:
            self.notes_index = {
                "notes": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_notes": 0
                }
            }
            self._save_index()
    
    def _save_index(self):
        """保存筆記索引"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.notes_index, f, ensure_ascii=False, indent=2)
    
    def _generate_note_id(self) -> str:
        """生成筆記ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        count = len(self.notes_index["notes"])
        return f"note_{timestamp}_{count}"
    
    def _get_note_path(self, note_id: str) -> Path:
        """取得筆記檔案路徑"""
        return self.workspace / f"{note_id}.md"
    
    def _note_to_markdown(self, note: Dict[str, Any]) -> str:
        """將筆記對象轉換為Markdown格式"""
        # YAML前置元資料
        frontmatter = "---\n"
        frontmatter += f"id: {note['id']}\n"
        frontmatter += f"title: {note['title']}\n"
        frontmatter += f"type: {note['type']}\n"
        if note.get('tags'):
            tags_str = json.dumps(note['tags'])
            frontmatter += f"tags: {tags_str}\n"
        frontmatter += f"created_at: {note['created_at']}\n"
        frontmatter += f"updated_at: {note['updated_at']}\n"
        frontmatter += "---\n\n"
        
        # Markdown內容
        content = f"# {note['title']}\n\n"
        content += note['content']
        
        return frontmatter + content
    
    def _markdown_to_note(self, markdown_text: str) -> Dict[str, Any]:
        """將Markdown文字解析為筆記對象"""
        # 提取YAML前置元資料
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', markdown_text, re.DOTALL)
        
        if not frontmatter_match:
            raise ValueError("無效的筆記格式：缺少YAML前置元資料")
        
        frontmatter_text = frontmatter_match.group(1)
        content_start = frontmatter_match.end()
        
        # 解析YAML（簡化版）
        note = {}
        for line in frontmatter_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 處理特殊字段
                if key == 'tags':
                    try:
                        note[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        note[key] = []
                else:
                    note[key] = value
        
        # 提取內容（去掉標題行）
        markdown_content = markdown_text[content_start:].strip()
        # 移除第一行的 # 標題
        lines = markdown_content.split('\n')
        if lines and lines[0].startswith('# '):
            markdown_content = '\n'.join(lines[1:]).strip()
        
        note['content'] = markdown_content
        
        # 添加元資料
        note['metadata'] = {
            'word_count': len(markdown_content),
            'status': 'active'
        }
        
        return note
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """執行工具（非展開模式）"""
        if not self.validate_parameters(parameters):
            return "[ERROR] 參數驗證失敗"

        action = parameters.get("action")

        # 根據action呼叫對應的方法，傳入提取的參數
        if action == "create":
            return self._create_note(
                title=parameters.get("title"),
                content=parameters.get("content"),
                note_type=parameters.get("note_type", "general"),
                tags=parameters.get("tags")
            )
        elif action == "read":
            return self._read_note(note_id=parameters.get("note_id"))
        elif action == "update":
            return self._update_note(
                note_id=parameters.get("note_id"),
                title=parameters.get("title"),
                content=parameters.get("content"),
                note_type=parameters.get("note_type"),
                tags=parameters.get("tags")
            )
        elif action == "delete":
            return self._delete_note(note_id=parameters.get("note_id"))
        elif action == "list":
            return self._list_notes(
                note_type=parameters.get("note_type"),
                limit=parameters.get("limit", 10)
            )
        elif action == "search":
            return self._search_notes(
                query=parameters.get("query"),
                limit=parameters.get("limit", 10)
            )
        elif action == "summary":
            return self._get_summary()
        else:
            return f"[ERROR] 不支援的操作: {action}"
    
    def get_parameters(self) -> List[ToolParameter]:
        """取得工具參數定義"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "操作類型: create(建立), read(讀取), update(更新), "
                    "delete(刪除), list(列表), search(搜尋), summary(摘要)"
                ),
                required=True
            ),
            ToolParameter(
                name="title",
                type="string",
                description="筆記標題（create/update時必需）",
                required=False
            ),
            ToolParameter(
                name="content",
                type="string",
                description="筆記內容（create/update時必需）",
                required=False
            ),
            ToolParameter(
                name="note_type",
                type="string",
                description=(
                    "筆記類型: task_state(任務狀態), conclusion(結論), "
                    "blocker(阻塞項), action(行動計劃), reference(參考), general(通用)"
                ),
                required=False,
                default="general"
            ),
            ToolParameter(
                name="tags",
                type="array",
                description="標簽列表（可選）",
                required=False
            ),
            ToolParameter(
                name="note_id",
                type="string",
                description="筆記ID（read/update/delete時必需）",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜尋關鍵詞（search時必需）",
                required=False
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="回傳結果數量限制（預設10）",
                required=False,
                default=10
            ),
        ]
    
    @tool_action("note_create", "建立一條新的結構化筆記")
    def _create_note(self, title: str, content: str, note_type: str = "general", tags: List[str] = None) -> str:
        """建立筆記

        Args:
            title: 筆記標題
            content: 筆記內容
            note_type: 筆記類型 (task_state, conclusion, blocker, action, reference, general)
            tags: 標簽列表

        Returns:
            建立結果
        """
        if not title or not content:
            return "[ERROR] 建立筆記需要提供 title 和 content"
        
        # 檢查筆記數量限制
        if len(self.notes_index["notes"]) >= self.max_notes:
            return f"[ERROR] 筆記數量已達上限 ({self.max_notes})"
        
        # 生成筆記ID
        note_id = self._generate_note_id()
        
        # 建立筆記對象
        note = {
            "id": note_id,
            "title": title,
            "content": content,
            "type": note_type,
            "tags": tags if isinstance(tags, list) else [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {
                "word_count": len(content),
                "status": "active"
            }
        }
        
        # 保存筆記檔案（Markdown格式）
        note_path = self._get_note_path(note_id)
        markdown_content = self._note_to_markdown(note)
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # 更新索引
        self.notes_index["notes"].append({
            "id": note_id,
            "title": title,
            "type": note_type,
            "tags": tags if isinstance(tags, list) else [],
            "created_at": note["created_at"]
        })
        self.notes_index["metadata"]["total_notes"] = len(self.notes_index["notes"])
        self._save_index()
        
        return f"[OK] 筆記建立成功\nID: {note_id}\n標題: {title}\n類型: {note_type}"
    
    @tool_action("note_read", "讀取指定ID的筆記")
    def _read_note(self, note_id: str) -> str:
        """讀取筆記

        Args:
            note_id: 筆記ID

        Returns:
            筆記內容
        """
        if not note_id:
            return "[ERROR] 讀取筆記需要提供 note_id"
        
        note_path = self._get_note_path(note_id)
        if not note_path.exists():
            return f"[ERROR] 筆記不存在: {note_id}"
        
        with open(note_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        note = self._markdown_to_note(markdown_text)
        
        return self._format_note(note)
    
    @tool_action("note_update", "更新已存在的筆記")
    def _update_note(self, note_id: str, title: str = None, content: str = None, note_type: str = None, tags: List[str] = None) -> str:
        """更新筆記

        Args:
            note_id: 筆記ID
            title: 新標題（可選）
            content: 新內容（可選）
            note_type: 新類型（可選）
            tags: 新標簽列表（可選）

        Returns:
            更新結果
        """
        if not note_id:
            return "[ERROR] 更新筆記需要提供 note_id"
        
        note_path = self._get_note_path(note_id)
        if not note_path.exists():
            return f"[ERROR] 筆記不存在: {note_id}"
        
        # 讀取現有筆記
        with open(note_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        note = self._markdown_to_note(markdown_text)

        # 更新字段
        if title:
            note["title"] = title
        if content:
            note["content"] = content
            note["metadata"]["word_count"] = len(content)
        if note_type:
            note["type"] = note_type
        if tags is not None:
            note["tags"] = tags if isinstance(tags, list) else []
        
        note["updated_at"] = datetime.now().isoformat()
        
        # 保存更新（Markdown格式）
        markdown_content = self._note_to_markdown(note)
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # 更新索引
        for idx_note in self.notes_index["notes"]:
            if idx_note["id"] == note_id:
                idx_note["title"] = note["title"]
                idx_note["type"] = note["type"]
                idx_note["tags"] = note["tags"]
                break
        self._save_index()
        
        return f"[OK] 筆記更新成功: {note_id}"
    
    @tool_action("note_delete", "刪除指定ID的筆記")
    def _delete_note(self, note_id: str) -> str:
        """刪除筆記

        Args:
            note_id: 筆記ID

        Returns:
            刪除結果
        """
        if not note_id:
            return "[ERROR] 刪除筆記需要提供 note_id"
        
        note_path = self._get_note_path(note_id)
        if not note_path.exists():
            return f"[ERROR] 筆記不存在: {note_id}"
        
        # 刪除檔案
        note_path.unlink()
        
        # 更新索引
        self.notes_index["notes"] = [
            n for n in self.notes_index["notes"] if n["id"] != note_id
        ]
        self.notes_index["metadata"]["total_notes"] = len(self.notes_index["notes"])
        self._save_index()
        
        return f"[OK] 筆記已刪除: {note_id}"
    
    @tool_action("note_list", "列出所有筆記或指定類型的筆記")
    def _list_notes(self, note_type: str = None, limit: int = 10) -> str:
        """列出筆記

        Args:
            note_type: 筆記類型過濾（可選）
            limit: 回傳結果數量限制

        Returns:
            筆記列表
        """
        # 過濾筆記
        filtered_notes = self.notes_index["notes"]
        if note_type:
            filtered_notes = [n for n in filtered_notes if n["type"] == note_type]
        
        # 限制數量
        filtered_notes = filtered_notes[:limit]
        
        if not filtered_notes:
            return "[INFO] 暫無筆記"
        
        result = f"[INFO] 筆記列表（共 {len(filtered_notes)} 條）\n\n"
        for note in filtered_notes:
            result += f"• [{note['type']}] {note['title']}\n"
            result += f"  ID: {note['id']}\n"
            if note.get('tags'):
                result += f"  標簽: {', '.join(note['tags'])}\n"
            result += f"  建立時間: {note['created_at']}\n\n"
        
        return result
    
    @tool_action("note_search", "搜尋包含關鍵詞的筆記")
    def _search_notes(self, query: str, limit: int = 10) -> str:
        """搜尋筆記

        Args:
            query: 搜尋關鍵詞
            limit: 回傳結果數量限制

        Returns:
            搜尋結果
        """
        if not query:
            return "[ERROR] 搜尋需要提供 query"

        query_lower = query.lower()
        
        # 搜尋匹配的筆記
        matched_notes = []
        for idx_note in self.notes_index["notes"]:
            note_path = self._get_note_path(idx_note["id"])
            if note_path.exists():
                with open(note_path, 'r', encoding='utf-8') as f:
                    markdown_text = f.read()
                
                try:
                    note = self._markdown_to_note(markdown_text)
                except Exception as e:
                    print(f"[WARN] 解析筆記失敗 {idx_note['id']}: {e}")
                    continue
                
                # 檢查標題、內容、標簽是否匹配
                if (query_lower in note["title"].lower() or
                    query_lower in note["content"].lower() or
                    any(query_lower in tag.lower() for tag in note.get("tags", []))):
                    matched_notes.append(note)
        
        # 限制數量
        matched_notes = matched_notes[:limit]
        
        if not matched_notes:
            return f"[INFO] 找不到匹配 '{query}' 的筆記"
        
        result = f"[INFO] 搜尋結果（共 {len(matched_notes)} 條）\n\n"
        for note in matched_notes:
            result += self._format_note(note, compact=True) + "\n"
        
        return result
    
    @tool_action("note_summary", "取得筆記系統的摘要統計資訊")
    def _get_summary(self) -> str:
        """取得筆記摘要

        Returns:
            摘要資訊
        """
        total = len(self.notes_index["notes"])
        
        # 按類型統計
        type_counts = {}
        for note in self.notes_index["notes"]:
            note_type = note["type"]
            type_counts[note_type] = type_counts.get(note_type, 0) + 1
        
        result = f"[INFO] 筆記摘要\n\n"
        result += f"總筆記數: {total}\n\n"
        result += "按類型統計:\n"
        for note_type, count in sorted(type_counts.items()):
            result += f"  • {note_type}: {count}\n"
        
        return result
    
    def _format_note(self, note: Dict[str, Any], compact: bool = False) -> str:
        """格式化筆記輸出"""
        if compact:
            return (
                f"[{note['type']}] {note['title']}\n"
                f"ID: {note['id']}\n"
                f"內容: {note['content'][:100]}{'...' if len(note['content']) > 100 else ''}"
            )
        else:
            result = f"[INFO] 筆記詳情\n\n"
            result += f"ID: {note['id']}\n"
            result += f"標題: {note['title']}\n"
            result += f"類型: {note['type']}\n"
            if note.get('tags'):
                result += f"標簽: {', '.join(note['tags'])}\n"
            result += f"建立時間: {note['created_at']}\n"
            result += f"更新時間: {note['updated_at']}\n"
            result += f"\n內容:\n{note['content']}\n"
            return result
 
