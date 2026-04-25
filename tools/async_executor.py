"""非同步工具執行器 - HelloAgents非同步工具執行支援"""

import asyncio
import concurrent.futures
from typing import Dict, Any, List
from .registry import ToolRegistry


class AsyncToolExecutor:
    """非同步工具執行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """非同步執行單個工具"""
        loop = asyncio.get_event_loop()
        
        def _execute():
            return self.registry.execute_tool(tool_name, input_data)
        
        try:
            result = await loop.run_in_executor(self.executor, _execute)
            return result
        except Exception as e:
            return f"[ERROR] 工具 '{tool_name}' 非同步執行失敗: {e}"

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        平行執行多個工具
        
        Args:
            tasks: 任務列表，每個任務包含 tool_name 和 input_data
            
        Returns:
            執行結果列表，包含任務資訊和結果
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
        批量執行同一個工具
        
        Args:
            tool_name: 工具名稱
            input_list: 輸入資料列表
            
        Returns:
            執行結果列表
        """
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks)

    def close(self):
        """關閉執行器"""
        self.executor.shutdown(wait=True)
        print("🔒 非同步工具執行器已關閉")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函式
async def run_parallel_tools(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函式：平行執行多個工具
    
    Args:
        registry: 工具註冊表
        tasks: 任務列表
        max_workers: 最大工作執行緒數
        
    Returns:
        執行結果列表
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_parallel(tasks)


async def run_batch_tool(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函式：批量執行同一個工具
    
    Args:
        registry: 工具註冊表
        tool_name: 工具名稱
        input_list: 輸入資料列表
        max_workers: 最大工作執行緒數
        
    Returns:
        執行結果列表
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_batch(tool_name, input_list)


# 同步包裝函式（為了相容性）
def run_parallel_tools_sync(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本的平行工具執行"""
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers))


def run_batch_tool_sync(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本的批量工具執行"""
    return asyncio.run(run_batch_tool(registry, tool_name, input_list, max_workers))


# 範例函式
async def demo_parallel_execution():
    """演示平行執行的範例"""
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
