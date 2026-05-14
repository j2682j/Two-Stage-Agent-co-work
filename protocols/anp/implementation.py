"""
基於 agent-connect 庫的 ANP 協議實現

使用 agent-connect 庫 (v0.3.7) 實現 Agent Network Protocol 功能。

注意：agent-connect 是一個底層的網路協議庫，提供了加密、認證等功能。
這裡我們建立一個簡化的包裝器，使其更易於使用。
"""

from typing import Dict, Any, List, Optional
import asyncio
import json


# 由於 agent-connect 的 API 比較底層，我們建立一個簡化的實現
# 實際使用時可以根據需要呼叫 agent-connect 的具體模組

class ServiceInfo:
    """
    負責在 protocols.anp.implementation 中封裝 ServiceInfo，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        service_id: 此流程需要使用的輸入資料。
        service_type: 此流程需要使用的輸入資料。
        endpoint: 此流程需要使用的輸入資料。
        service_name: 此流程需要使用的輸入資料。
        capabilities: 此流程需要使用的輸入資料。
        metadata: 目前流程所需的上下文、狀態或附加資訊。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        service_id: str,
        service_type: str,
        endpoint: str,
        service_name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        負責執行 ServiceInfo 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            service_id: 此流程需要使用的輸入資料。
            service_type: 此流程需要使用的輸入資料。
            endpoint: 此流程需要使用的輸入資料。
            service_name: 此流程需要使用的輸入資料。
            capabilities: 此流程需要使用的輸入資料。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.service_id = service_id
        self.service_type = service_type
        self.endpoint = endpoint
        self.service_name = service_name or service_id
        self.capabilities = capabilities or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        負責執行 ServiceInfo 中的 to_dict 流程，將內部資料整理成日誌、提示詞、摘要或指定的輸出格式。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "service_id": self.service_id,
            "service_type": self.service_type,
            "endpoint": self.endpoint,
            "service_name": self.service_name,
            "capabilities": self.capabilities,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInfo':
        """
        負責執行 ServiceInfo 中的 from_dict 流程，依照 ServiceInfo 的流程需求處理 from_dict 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            data: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 'ServiceInfo'。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return cls(
            service_id=data["service_id"],
            service_type=data["service_type"],
            endpoint=data["endpoint"],
            service_name=data.get("service_name"),
            capabilities=data.get("capabilities"),
            metadata=data.get("metadata", {})
        )


class ANPDiscovery:
    """
    負責在 protocols.anp.implementation 中封裝 ANPDiscovery，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(self):
        """
        負責執行 ANPDiscovery 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._services: Dict[str, ServiceInfo] = {}
        
    def register_service(self, service: ServiceInfo) -> bool:
        """
        負責執行 ANPDiscovery 中的 register_service 流程，依照 ANPDiscovery 的流程需求處理 register_service 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            service: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._services[service.service_id] = service
        return True
        
    def unregister_service(self, service_id: str) -> bool:
        """
        負責執行 ANPDiscovery 中的 unregister_service 流程，依照 ANPDiscovery 的流程需求處理 unregister_service 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            service_id: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if service_id in self._services:
            del self._services[service_id]
            return True
        return False
        
    def discover_services(
        self,
        service_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ServiceInfo]:
        """
        負責執行 ANPDiscovery 中的 discover_services 流程，依照 ANPDiscovery 的流程需求處理 discover_services 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            service_type: 此流程需要使用的輸入資料。
            filters: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[ServiceInfo]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        services = list(self._services.values())
        
        # 按類型過濾
        if service_type:
            services = [s for s in services if s.service_type == service_type]
            
        # 按元資料過濾
        if filters:
            def matches_filters(service: ServiceInfo) -> bool:
                """
                負責執行 ANPDiscovery 中的 matches_filters 流程，依照 ANPDiscovery 的流程需求處理 matches_filters 對應的資料轉換、狀態操作或結果產生。
                
                Args:
                    service: 此流程需要使用的輸入資料。
                
                Returns:
                    執行結果；若函式標註回傳型別，預期型別為 bool。
                
                限制或副作用:
                    可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
                """
                for key, value in filters.items():
                    if service.metadata.get(key) != value:
                        return False
                return True
            services = [s for s in services if matches_filters(s)]
            
        return services
        
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """
        負責執行 ANPDiscovery 中的 get_service 流程，依照 ANPDiscovery 的流程需求處理 get_service 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            service_id: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[ServiceInfo]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self._services.get(service_id)
        
    def list_all_services(self) -> List[ServiceInfo]:
        """
        負責執行 ANPDiscovery 中的 list_all_services 流程，依照 ANPDiscovery 的流程需求處理 list_all_services 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[ServiceInfo]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return list(self._services.values())


class ANPNetwork:
    """
    負責在 protocols.anp.implementation 中封裝 ANPNetwork，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        network_id: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    
    def __init__(self, network_id: str = "default"):
        """
        負責執行 ANPNetwork 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            network_id: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.network_id = network_id
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._connections: Dict[str, List[str]] = {}
        
    def add_node(self, node_id: str, endpoint: str, metadata: Optional[Dict[str, Any]] = None):
        """
        負責執行 ANPNetwork 中的 add_node 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            node_id: 目前執行或需要記錄的代理節點識別資訊。
            endpoint: 圖結構中的節點、邊或相關識別資料。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._nodes[node_id] = {
            "node_id": node_id,
            "endpoint": endpoint,
            "metadata": metadata or {},
            "status": "active"
        }
        self._connections[node_id] = []
        
    def remove_node(self, node_id: str) -> bool:
        """
        負責執行 ANPNetwork 中的 remove_node 流程，清除或移除指定資源、狀態或註冊資料，維持後續流程的一致性。
        
        Args:
            node_id: 目前執行或需要記錄的代理節點識別資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 bool。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if node_id in self._nodes:
            del self._nodes[node_id]
            del self._connections[node_id]
            # 移除其他節點到此節點的連線
            for connections in self._connections.values():
                if node_id in connections:
                    connections.remove(node_id)
            return True
        return False
        
    def connect_nodes(self, from_node: str, to_node: str):
        """
        負責執行 ANPNetwork 中的 connect_nodes 流程，依照 ANPNetwork 的流程需求處理 connect_nodes 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            from_node: 圖結構中的節點、邊或相關識別資料。
            to_node: 圖結構中的節點、邊或相關識別資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if from_node in self._connections and to_node in self._nodes:
            if to_node not in self._connections[from_node]:
                self._connections[from_node].append(to_node)
                
    def route_message(
        self,
        from_node: str,
        to_node: str,
        message: Dict[str, Any]
    ) -> Optional[List[str]]:
        """
        負責執行 ANPNetwork 中的 route_message 流程，根據任務特徵、候選答案或評分結果選擇後續節點、工具或流程分支。
        
        Args:
            from_node: 圖結構中的節點、邊或相關識別資料。
            to_node: 圖結構中的節點、邊或相關識別資料。
            message: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[List[str]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if from_node not in self._nodes or to_node not in self._nodes:
            return None
            
        # 簡單實現：直接路由
        if to_node in self._connections.get(from_node, []):
            return [from_node, to_node]
            
        # 嘗試通過一跳中轉
        for intermediate in self._connections.get(from_node, []):
            if to_node in self._connections.get(intermediate, []):
                return [from_node, intermediate, to_node]
                
        return None
        
    def broadcast_message(self, from_node: str, message: Dict[str, Any]) -> List[str]:
        """
        負責執行 ANPNetwork 中的 broadcast_message 流程，依照 ANPNetwork 的流程需求處理 broadcast_message 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            from_node: 圖結構中的節點、邊或相關識別資料。
            message: 模型、節點或工具產生的候選回覆內容。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if from_node not in self._connections:
            return []
            
        return self._connections[from_node].copy()
        
    def get_network_stats(self) -> Dict[str, Any]:
        """
        負責執行 ANPNetwork 中的 get_network_stats 流程，依照 ANPNetwork 的流程需求處理 get_network_stats 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        total_connections = sum(len(conns) for conns in self._connections.values())
        active_nodes = sum(1 for node in self._nodes.values() if node["status"] == "active")
        
        return {
            "network_id": self.network_id,
            "total_nodes": len(self._nodes),
            "active_nodes": active_nodes,
            "total_connections": total_connections,
            "nodes": list(self._nodes.keys())
        }
        
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        負責執行 ANPNetwork 中的 get_node_info 流程，依照 ANPNetwork 的流程需求處理 get_node_info 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            node_id: 目前執行或需要記錄的代理節點識別資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if node_id in self._nodes:
            node_info = self._nodes[node_id].copy()
            node_info["connections"] = self._connections[node_id].copy()
            return node_info
        return None


# 範例：建立一個簡單的 ANP 網路
def create_example_network() -> ANPNetwork:
    """
    負責執行 protocols.anp.implementation 中的 create_example_network 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 ANPNetwork。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    network = ANPNetwork(network_id="example_network")
    
    # 添加節點
    network.add_node("node1", "http://localhost:8001", {"type": "agent", "role": "coordinator"})
    network.add_node("node2", "http://localhost:8002", {"type": "agent", "role": "worker"})
    network.add_node("node3", "http://localhost:8003", {"type": "agent", "role": "worker"})
    
    # 連線節點
    network.connect_nodes("node1", "node2")
    network.connect_nodes("node1", "node3")
    network.connect_nodes("node2", "node3")
    
    return network


if __name__ == "__main__":
    # 建立範例網路
    network = create_example_network()
    print(f"🌐 ANP Network: {network.network_id}")
    print(f"📊 Network Stats:")
    stats = network.get_network_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    # 測試路由
    print("🔀 Testing message routing:")
    path = network.route_message("node1", "node2", {"type": "test", "content": "Hello"})
    print(f"   Route from node1 to node2: {' -> '.join(path) if path else 'No route found'}")
    
    # 測試廣播
    print("\n📢 Testing broadcast:")
    recipients = network.broadcast_message("node1", {"type": "broadcast", "content": "Hello all"})
    print(f"   Broadcast from node1 to: {', '.join(recipients)}")

 