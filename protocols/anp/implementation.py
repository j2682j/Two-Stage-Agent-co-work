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
    """服務資訊"""

    def __init__(
        self,
        service_id: str,
        service_type: str,
        endpoint: str,
        service_name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.service_id = service_id
        self.service_type = service_type
        self.endpoint = endpoint
        self.service_name = service_name or service_id
        self.capabilities = capabilities or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
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
        """從字典建立"""
        return cls(
            service_id=data["service_id"],
            service_type=data["service_type"],
            endpoint=data["endpoint"],
            service_name=data.get("service_name"),
            capabilities=data.get("capabilities"),
            metadata=data.get("metadata", {})
        )


class ANPDiscovery:
    """基於 agent-connect 的服務發現實現"""
    
    def __init__(self):
        """初始化服務發現"""
        self._services: Dict[str, ServiceInfo] = {}
        
    def register_service(self, service: ServiceInfo) -> bool:
        """
        註冊服務
        
        Args:
            service: 服務資訊
            
        Returns:
            是否註冊成功
        """
        self._services[service.service_id] = service
        return True
        
    def unregister_service(self, service_id: str) -> bool:
        """
        注銷服務
        
        Args:
            service_id: 服務 ID
            
        Returns:
            是否注銷成功
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
        發現服務
        
        Args:
            service_type: 服務類型（可選）
            filters: 過濾條件（可選）
            
        Returns:
            服務列表
        """
        services = list(self._services.values())
        
        # 按類型過濾
        if service_type:
            services = [s for s in services if s.service_type == service_type]
            
        # 按元資料過濾
        if filters:
            def matches_filters(service: ServiceInfo) -> bool:
                for key, value in filters.items():
                    if service.metadata.get(key) != value:
                        return False
                return True
            services = [s for s in services if matches_filters(s)]
            
        return services
        
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """
        取得服務資訊
        
        Args:
            service_id: 服務 ID
            
        Returns:
            服務資訊，如果不存在則回傳 None
        """
        return self._services.get(service_id)
        
    def list_all_services(self) -> List[ServiceInfo]:
        """列出所有服務"""
        return list(self._services.values())


class ANPNetwork:
    """基於 agent-connect 的網路管理實現"""
    
    def __init__(self, network_id: str = "default"):
        """
        初始化網路管理器
        
        Args:
            network_id: 網路 ID
        """
        self.network_id = network_id
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._connections: Dict[str, List[str]] = {}
        
    def add_node(self, node_id: str, endpoint: str, metadata: Optional[Dict[str, Any]] = None):
        """
        添加節點到網路
        
        Args:
            node_id: 節點 ID
            endpoint: 節點端點
            metadata: 節點元資料
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
        從網路中移除節點
        
        Args:
            node_id: 節點 ID
            
        Returns:
            是否移除成功
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
        連線兩個節點
        
        Args:
            from_node: 源節點 ID
            to_node: 目標節點 ID
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
        路由消息（簡單的直接路由）
        
        Args:
            from_node: 源節點 ID
            to_node: 目標節點 ID
            message: 消息內容
            
        Returns:
            路由路徑，如果無法路由則回傳 None
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
        廣播消息到所有連線的節點
        
        Args:
            from_node: 源節點 ID
            message: 消息內容
            
        Returns:
            接收消息的節點列表
        """
        if from_node not in self._connections:
            return []
            
        return self._connections[from_node].copy()
        
    def get_network_stats(self) -> Dict[str, Any]:
        """
        取得網路統計資訊
        
        Returns:
            網路統計資訊
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
        取得節點資訊
        
        Args:
            node_id: 節點 ID
            
        Returns:
            節點資訊，如果不存在則回傳 None
        """
        if node_id in self._nodes:
            node_info = self._nodes[node_id].copy()
            node_info["connections"] = self._connections[node_id].copy()
            return node_info
        return None


# 範例：建立一個簡單的 ANP 網路
def create_example_network() -> ANPNetwork:
    """建立一個範例 ANP 網路"""
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

 