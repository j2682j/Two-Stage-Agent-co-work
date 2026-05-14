"""非同步工具執行器 - HelloAgents非同步工具執行支援"""

import asyncio
import concurrent.futures
from typing import Dict, Any, List
from .registry import ToolRegistry


class AsyncToolExecutor:
    """
    負責在 tools.async_executor 中封裝 AsyncToolExecutor，封裝工具呼叫、參數處理與工具結果回傳流程。
    
    Args:
        registry: 可呼叫的工具、工具名稱或工具註冊表。
        max_workers: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        """
        負責執行 AsyncToolExecutor 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            registry: 可呼叫的工具、工具名稱或工具註冊表。
            max_workers: 控制檢索、篩選或輸出數量的數值參數。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """
        負責執行 AsyncToolExecutor 中的 execute_tool_async 流程，依照 AsyncToolExecutor 的流程需求處理 execute_tool_async 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            input_data: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        loop = asyncio.get_event_loop()
        
        def _execute():
            """
            負責執行 AsyncToolExecutor 中的 _execute 流程，依照 AsyncToolExecutor 的流程需求處理 _execute 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                無。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            return self.registry.execute_tool(tool_name, input_data)
        
        try:
            result = await loop.run_in_executor(self.executor, _execute)
            return result
        except Exception as e:
            return f"[ERROR] 工具 '{tool_name}' 非同步執行失敗: {e}"

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        負責執行 AsyncToolExecutor 中的 execute_tools_parallel 流程，依照 AsyncToolExecutor 的流程需求處理 execute_tools_parallel 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tasks: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        print(f"[INFO] 開始平行執行 {len(tasks)} 個工具任務")
        
        # 建立非同步任務
        async_tasks = []
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            input_data = task.get("input_data", "")
            
            if not tool_name:
                continue
                
            print(f"[INFO] 建立任務 {i+1}: {tool_name}")
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append((i, task, async_task))
        
        # 等待所有任務完成
        results = []
        for i, task, async_task in async_tasks:
            try:
                result = await async_task
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": result,
                    "status": "success"
                })
                print(f"[OK] 任務 {i+1} 完成: {task['tool_name']}")
            except Exception as e:
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": str(e),
                    "status": "error"
                })
                print(f"[ERROR] 任務 {i+1} 失敗: {task['tool_name']} - {e}")
        
        print(f"🎉 平行執行完成，成功: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
        return results

    async def execute_tools_batch(self, tool_name: str, input_list: List[str]) -> List[Dict[str, Any]]:
        """
        負責執行 AsyncToolExecutor 中的 execute_tools_batch 流程，依照 AsyncToolExecutor 的流程需求處理 execute_tools_batch 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            tool_name: 可呼叫的工具、工具名稱或工具註冊表。
            input_list: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks)

    def close(self):
        """
        負責執行 AsyncToolExecutor 中的 close 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.executor.shutdown(wait=True)
        print("🔒 非同步工具執行器已關閉")

    def __enter__(self):
        """
        負責執行 AsyncToolExecutor 中的 __enter__ 流程，依照 AsyncToolExecutor 的流程需求處理 __enter__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        負責執行 AsyncToolExecutor 中的 __exit__ 流程，依照 AsyncToolExecutor 的流程需求處理 __exit__ 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            exc_type: 此流程需要使用的輸入資料。
            exc_val: 此流程需要使用的輸入資料。
            exc_tb: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.close()


# 便捷函式
async def run_parallel_tools(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    負責執行 tools.async_executor 中的 run_parallel_tools 流程，依照 tools.async_executor 的流程需求處理 run_parallel_tools 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        registry: 可呼叫的工具、工具名稱或工具註冊表。
        tasks: 此流程需要使用的輸入資料。
        max_workers: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_parallel(tasks)


async def run_batch_tool(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    負責執行 tools.async_executor 中的 run_batch_tool 流程，依照 tools.async_executor 的流程需求處理 run_batch_tool 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        registry: 可呼叫的工具、工具名稱或工具註冊表。
        tool_name: 可呼叫的工具、工具名稱或工具註冊表。
        input_list: 此流程需要使用的輸入資料。
        max_workers: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_batch(tool_name, input_list)


# 同步包裝函式（為了相容性）
def run_parallel_tools_sync(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    負責執行 tools.async_executor 中的 run_parallel_tools_sync 流程，依照 tools.async_executor 的流程需求處理 run_parallel_tools_sync 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        registry: 可呼叫的工具、工具名稱或工具註冊表。
        tasks: 此流程需要使用的輸入資料。
        max_workers: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers))


def run_batch_tool_sync(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    負責執行 tools.async_executor 中的 run_batch_tool_sync 流程，依照 tools.async_executor 的流程需求處理 run_batch_tool_sync 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        registry: 可呼叫的工具、工具名稱或工具註冊表。
        tool_name: 可呼叫的工具、工具名稱或工具註冊表。
        input_list: 此流程需要使用的輸入資料。
        max_workers: 控制檢索、篩選或輸出數量的數值參數。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return asyncio.run(run_batch_tool(registry, tool_name, input_list, max_workers))


# 範例函式
async def demo_parallel_execution():
    """
    負責執行 tools.async_executor 中的 demo_parallel_execution 流程，依照 tools.async_executor 的流程需求處理 demo_parallel_execution 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    from .registry import ToolRegistry
    
    # 建立註冊表（這裡假設已經註冊了工具）
    registry = ToolRegistry()
    
    # 定義平行任務
    tasks = [
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "3 * 4"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
        {"tool_name": "my_calculator", "input_data": "10 / 2"},
    ]
    
    # 平行執行
    results = await run_parallel_tools(registry, tasks)
    
    # 顯示結果
    print("\n[INFO] 平行執行結果:")
    for result in results:
        status_icon = "[OK]" if result["status"] == "success" else "[ERROR]"
        print(f"{status_icon} {result['tool_name']}({result['input_data']}) = {result['result']}")
    
    return results


if __name__ == "__main__":
    # 執行演示
    asyncio.run(demo_parallel_execution())
