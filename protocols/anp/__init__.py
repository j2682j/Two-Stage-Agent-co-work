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
    """
    負責執行 protocols.anp.__init__ 中的 register_service 流程，依照 protocols.anp.__init__ 的流程需求處理 register_service 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        discovery: 此流程需要使用的輸入資料。
        service: 此流程需要使用的輸入資料。
        service_id: 此流程需要使用的輸入資料。
        service_type: 此流程需要使用的輸入資料。
        endpoint: 此流程需要使用的輸入資料。
        service_name: 此流程需要使用的輸入資料。
        capabilities: 此流程需要使用的輸入資料。
        metadata: 目前流程所需的上下文、狀態或附加資訊。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 bool。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
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
    """
    負責執行 protocols.anp.__init__ 中的 discover_service 流程，依照 protocols.anp.__init__ 的流程需求處理 discover_service 對應的資料轉換、狀態操作或結果產生。
    
    Args:
        discovery: 此流程需要使用的輸入資料。
        service_type: 此流程需要使用的輸入資料。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 未標註。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    return discovery.discover_services(service_type=service_type)

__all__ = [
    "ANPDiscovery",
    "ANPNetwork",
    "ServiceInfo",
    "register_service",
    "discover_service",
]

