"""ANP (Agent Network Protocol) 協議實現

概念性實現，提供簡潔的 API 用於：
- Agent 服務發現
- Agent 網路管理
- 負載均衡與路由

注意: 這是概念性實現，用於學習和理解 ANP 理念。
詳見文檔: docs/chapter10/ANP_CONCEPTS.md
"""

from typing import Optional, Dict, Any, List

from .implementation import (
    ANPDiscovery,
    ANPNetwork,
    ServiceInfo
)

def register_service(
    discovery: ANPDiscovery,
    service: Optional[ServiceInfo] = None,
    service_id: Optional[str] = None,
    service_type: Optional[str] = None,
    endpoint: Optional[str] = None,
    service_name: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """註冊服務的便捷函式

    支援兩種呼叫方式：
    1. 傳入 ServiceInfo 對象：
       register_service(discovery, service=service_info)

    2. 傳入參數自動構造：
       register_service(
           discovery=discovery,
           service_id="agent1",
           service_type="nlp",
           endpoint="http://localhost:8001",
           service_name="NLP Agent",
           capabilities=["text_analysis"],
           metadata={"version": "1.0"}
       )
    """
    if service is not None:
        # 方式1：直接傳入 ServiceInfo 對象
        return discovery.register_service(service)
    else:
        # 方式2：從參數構造 ServiceInfo 對象
        if not all([service_id, service_type, endpoint]):
            raise ValueError("必須提供 service_id, service_type 和 endpoint 參數")

        service_info = ServiceInfo(
            service_id=service_id,
            service_type=service_type,
            endpoint=endpoint,
            service_name=service_name,
            capabilities=capabilities,
            metadata=metadata
        )
        return discovery.register_service(service_info)

def discover_service(discovery: ANPDiscovery, service_type: str = None):
    """發現服務的便捷函式"""
    return discovery.discover_services(service_type=service_type)

__all__ = [
    "ANPDiscovery",
    "ANPNetwork",
    "ServiceInfo",
    "register_service",
    "discover_service",
]

