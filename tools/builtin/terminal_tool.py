"""TerminalTool - 命令行工具

為Agent提供安全的命令行執行能力，支援：
- 檔案系統操作（ls, cat, head, tail, find, grep）
- 文字處理（wc, sort, uniq）
- 目錄導航（pwd, cd）
- 安全限制（白名單命令、路徑限制、逾時控制）

使用場景：
- JIT（即時）檔案搜尋與分析
- 代碼倉庫探索
- 日誌檔案分析
- 資料檔案預覽

安全特性：
- 命令白名單（只允許安全的只讀命令）
- 工作目錄限制（沙箱）
- 逾時控制
- 輸出大小限制
- 禁止危險操作（rm, mv, chmod等）
"""

from typing import Dict, Any, List, Optional
import subprocess
import os
from pathlib import Path
import shlex
import platform

from ..base import Tool, ToolParameter


class TerminalTool(Tool):
    """
    負責在 tools.builtin.terminal_tool 中封裝 TerminalTool，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        workspace: 此流程需要使用的輸入資料。
        timeout: 此流程需要使用的輸入資料。
        max_output_size: 控制檢索、篩選或輸出數量的數值參數。
        allow_cd: 此流程需要使用的輸入資料。
        os_type: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    # 允許的命令白名單（跨平臺）
    ALLOWED_COMMANDS = {
        # 檔案列表與資訊
        'ls', 'dir', 'tree',
        # 檔案內容查看
        'cat', 'type', 'head', 'tail', 'less', 'more',
        # 檔案搜尋
        'find', 'where', 'grep', 'egrep', 'fgrep', 'findstr',
        # 文字處理
        'wc', 'sort', 'uniq', 'cut', 'awk', 'sed',
        # 目錄操作
        'pwd', 'cd',
        # 檔案資訊
        'file', 'stat', 'du', 'df',
        # 其他
        'echo', 'which', 'whereis',
        # 代碼執行
        'python', 'python3', 'node', 'bash', 'sh', 'powershell', 'cmd',
    }

    def __init__(
        self,
        workspace: str = ".",
        timeout: int = 30,
        max_output_size: int = 10 * 1024 * 1024,  # 10MB
        allow_cd: bool = True,
        os_type: str = "auto"  # "auto", "windows", "linux", "mac"
    ):
        """
        負責執行 TerminalTool 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            workspace: 此流程需要使用的輸入資料。
            timeout: 此流程需要使用的輸入資料。
            max_output_size: 控制檢索、篩選或輸出數量的數值參數。
            allow_cd: 此流程需要使用的輸入資料。
            os_type: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        super().__init__(
            name="terminal",
            description="跨平臺命令行工具 - 執行安全的檔案系統、文字處理和代碼執行命令（支援Windows/Linux/Mac）"
        )

        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allow_cd = allow_cd

        # 檢測或設定操作系統類型
        if os_type == "auto":
            self.os_type = self._detect_os()
        else:
            self.os_type = os_type.lower()

        # 目前工作目錄（相對於workspace）
        self.current_dir = self.workspace

        # 確保工作目錄存在
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _detect_os(self) -> str:
        """
        負責執行 TerminalTool 中的 _detect_os 流程，依照 TerminalTool 的流程需求處理 _detect_os 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "darwin":
            return "mac"
        else:
            return "linux"
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        負責執行 TerminalTool 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            parameters: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.validate_parameters(parameters):
            return "[ERROR] 參數驗證失敗"
        
        command = parameters.get("command", "").strip()
        
        if not command:
            return "[ERROR] 命令不能為空"
        
        # 解析命令
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return f"[ERROR] 命令解析失敗: {e}"
        
        if not parts:
            return "[ERROR] 命令不能為空"
        
        base_command = parts[0]
        
        # 檢查命令是否在白名單中
        if base_command not in self.ALLOWED_COMMANDS:
            return f"[ERROR] 不允許的命令: {base_command}\n允許的命令: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
        
        # 特殊處理 cd 命令
        if base_command == 'cd':
            return self._handle_cd(parts)
        
        # 執行命令
        return self._execute_command(command)
    
    def get_parameters(self) -> List[ToolParameter]:
        """
        負責執行 TerminalTool 中的 get_parameters 流程，依照 TerminalTool 的流程需求處理 get_parameters 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[ToolParameter]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [
            ToolParameter(
                name="command",
                type="string",
                description=(
                    f"要執行的命令（白名單: {', '.join(sorted(list(self.ALLOWED_COMMANDS)[:10]))}...）\n"
                    "範例: 'ls -la', 'cat file.txt', 'grep pattern *.py', 'head -n 20 data.csv'"
                ),
                required=True
            ),
        ]
    
    def _handle_cd(self, parts: List[str]) -> str:
        """
        負責執行 TerminalTool 中的 _handle_cd 流程，依照 TerminalTool 的流程需求處理 _handle_cd 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            parts: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.allow_cd:
            return "[ERROR] cd 命令已禁用"
        
        if len(parts) < 2:
            # cd 無參數，回傳目前目錄
            return f"目前目錄: {self.current_dir}"
        
        target_dir = parts[1]
        
        # 處理相對路徑
        if target_dir == "..":
            new_dir = self.current_dir.parent
        elif target_dir == ".":
            new_dir = self.current_dir
        elif target_dir == "~":
            new_dir = self.workspace
        else:
            new_dir = (self.current_dir / target_dir).resolve()
        
        # 檢查是否在工作目錄內
        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return f"[ERROR] 不允許訪問工作目錄外的路徑: {new_dir}"
        
        # 檢查目錄是否存在
        if not new_dir.exists():
            return f"[ERROR] 目錄不存在: {new_dir}"
        
        if not new_dir.is_dir():
            return f"[ERROR] 不是目錄: {new_dir}"
        
        # 更新目前目錄
        self.current_dir = new_dir
        return f"[OK] 切換到目錄: {self.current_dir}"
    
    def _execute_command(self, command: str) -> str:
        """
        負責執行 TerminalTool 中的 _execute_command 流程，依照 TerminalTool 的流程需求處理 _execute_command 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            command: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            # 根據操作系統類型調整命令執行方式
            if self.os_type == "windows":
                # Windows下使用cmd.exe或直接shell=True
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.current_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=os.environ.copy()
                )
            else:
                # Unix系統（Linux/Mac）使用shell=True
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.current_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=os.environ.copy()
                )

            # 合併標準輸出和標準錯誤
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            # 檢查輸出大小
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size]
                output += f"\n\n[WARN] 輸出被截斷（超過 {self.max_output_size} 位元組）"

            # 添加回傳碼資訊
            if result.returncode != 0:
                output = f"[WARN] 命令回傳碼: {result.returncode}\n\n{output}"

            return output if output else "[OK] 命令執行成功（無輸出）"

        except subprocess.TimeoutExpired:
            return f"[ERROR] 命令執行逾時（超過 {self.timeout} 秒）"
        except Exception as e:
            return f"[ERROR] 命令執行失敗: {e}"

    def get_current_dir(self) -> str:
        """
        負責執行 TerminalTool 中的 get_current_dir 流程，依照 TerminalTool 的流程需求處理 get_current_dir 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return str(self.current_dir)

    def reset_dir(self):
        """
        負責執行 TerminalTool 中的 reset_dir 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.current_dir = self.workspace

    def get_os_type(self) -> str:
        """
        負責執行 TerminalTool 中的 get_os_type 流程，依照 TerminalTool 的流程需求處理 get_os_type 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.os_type

