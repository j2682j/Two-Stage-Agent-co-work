"""
基於官方 a2a-sdk 庫的 A2A 協議實現

使用官方 a2a-sdk 庫實現 Agent-to-Agent Protocol 功能。
官方倉庫: https://github.com/a2aproject/a2a-python
安裝: pip install a2a-sdk
"""

from typing import Dict, Any, List, Optional
import asyncio

try:
    from a2a.client import A2AClient
    from a2a.types import Message
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    A2AClient = None
    Message = None


class A2AServer:
    """
    負責在 protocols.a2a.implementation 中封裝 A2AServer，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        name: 此流程需要使用的輸入資料。
        description: 此流程需要使用的輸入資料。
        version: 此流程需要使用的輸入資料。
        capabilities: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        capabilities: Optional[Dict[str, Any]] = None
    ):
        """
        負責執行 A2AServer 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
            version: 此流程需要使用的輸入資料。
            capabilities: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.name = name
        self.description = description
        self.version = version
        self.capabilities = capabilities or {}
        self.skills = {}

    def add_skill(self, skill_name: str, func):
        """
        負責執行 A2AServer 中的 add_skill 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            skill_name: 此流程需要使用的輸入資料。
            func: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.skills[skill_name] = func
        return func

    def skill(self, skill_name: str):
        """
        負責執行 A2AServer 中的 skill 流程，依照 A2AServer 的流程需求處理 skill 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            skill_name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        def decorator(func):
            """
            負責執行 A2AServer 中的 decorator 流程，依照 A2AServer 的流程需求處理 decorator 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                func: 此流程需要使用的輸入資料。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            self.add_skill(skill_name, func)
            return func
        return decorator

    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """
        負責執行 A2AServer 中的 run 流程，啟動主要執行流程，串接輸入準備、核心處理與結果輸出。
        
        Args:
            host: 此流程需要使用的輸入資料。
            port: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError(
                "A2A server requires Flask. Install it with: pip install flask"
            )

        app = Flask(self.name)

        # 禁用 Flask 的日誌輸出（可選）
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @app.route('/info', methods=['GET'])
        def get_info():
            """
            負責執行 A2AServer 中的 get_info 流程，依照 A2AServer 的流程需求處理 get_info 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                無。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            return jsonify(self.get_info())

        @app.route('/skills', methods=['GET'])
        def list_skills():
            """
            負責執行 A2AServer 中的 list_skills 流程，依照 A2AServer 的流程需求處理 list_skills 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                無。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            return jsonify({
                "skills": list(self.skills.keys()),
                "count": len(self.skills)
            })

        @app.route('/execute/<skill_name>', methods=['POST'])
        def execute_skill(skill_name):
            """
            負責執行 A2AServer 中的 execute_skill 流程，依照 A2AServer 的流程需求處理 execute_skill 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                skill_name: 此流程需要使用的輸入資料。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            if skill_name not in self.skills:
                return jsonify({
                    "error": f"Skill '{skill_name}' not found",
                    "available_skills": list(self.skills.keys())
                }), 404

            try:
                data = request.get_json() or {}
                text = data.get('text', data.get('query', ''))

                # 呼叫技能函式
                result = self.skills[skill_name](text)

                return jsonify({
                    "skill": skill_name,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                return jsonify({
                    "error": str(e),
                    "skill": skill_name,
                    "status": "error"
                }), 500

        @app.route('/ask', methods=['POST'])
        def ask():
            """
            負責執行 A2AServer 中的 ask 流程，依照 A2AServer 的流程需求處理 ask 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                無。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            try:
                data = request.get_json() or {}
                question = data.get('question', data.get('text', ''))

                # 簡單策略：嘗試所有技能，回傳第一個非錯誤結果
                for skill_name, skill_func in self.skills.items():
                    try:
                        result = skill_func(question)
                        if result and not result.startswith("Error"):
                            return jsonify({
                                "answer": result,
                                "skill_used": skill_name,
                                "status": "success"
                            })
                    except:
                        continue

                return jsonify({
                    "answer": "No suitable skill found for this question",
                    "status": "no_match"
                })
            except Exception as e:
                return jsonify({
                    "error": str(e),
                    "status": "error"
                }), 500

        @app.route('/health', methods=['GET'])
        def health():
            """
            負責執行 A2AServer 中的 health 流程，依照 A2AServer 的流程需求處理 health 對應的資料轉換、狀態操作或結果產生。
            
            Args:
                無。
            
            Returns:
                執行結果；若函式標註回傳型別，預期型別為 未標註。
            
            限制或副作用:
                可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
            """
            return jsonify({"status": "healthy", "agent": self.name})

        # 啟動伺服器
        print(f"🚀 A2A 伺服器 '{self.name}' 啟動在 {host}:{port}")
        print(f"📋 描述: {self.description}")
        print(f"🛠️  可用技能: {list(self.skills.keys())}")
        print(f"📡 API 端點:")
        print(f"   - GET  {host}:{port}/info - 取得 Agent 資訊")
        print(f"   - GET  {host}:{port}/skills - 列出技能")
        print(f"   - POST {host}:{port}/execute/<skill> - 執行技能")
        print(f"   - POST {host}:{port}/ask - 通用問答")
        print(f"   - GET  {host}:{port}/health - 健康檢查")
        print()

        app.run(host=host, port=port, debug=False)

    def get_info(self) -> Dict[str, Any]:
        """
        負責執行 A2AServer 中的 get_info 流程，依照 A2AServer 的流程需求處理 get_info 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "protocol": "A2A",
            "skills": list(self.skills.keys())
        }


class A2AClient:
    """
    負責在 protocols.a2a.implementation 中封裝 A2AClient，封裝此模組的狀態資料與主要操作流程。
    
    Args:
        server_url: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, server_url: str):
        """
        負責執行 A2AClient 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            server_url: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.server_url = server_url.rstrip('/')

    def ask(self, question: str) -> str:
        """
        負責執行 A2AClient 中的 ask 流程，依照 A2AClient 的流程需求處理 ask 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            import requests
            response = requests.post(
                f"{self.server_url}/ask",
                json={"question": question},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("answer", "No response")
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"

    def execute_skill(self, skill_name: str, text: str = "") -> Dict[str, Any]:
        """
        負責執行 A2AClient 中的 execute_skill 流程，依照 A2AClient 的流程需求處理 execute_skill 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            skill_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            text: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            import requests
            response = requests.post(
                f"{self.server_url}/execute/{skill_name}",
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to execute skill: {str(e)}", "status": "error"}

    def get_info(self) -> Dict[str, Any]:
        """
        負責執行 A2AClient 中的 get_info 流程，依照 A2AClient 的流程需求處理 get_info 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            import requests
            response = requests.get(f"{self.server_url}/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to get agent info: {str(e)}"}

    def list_skills(self) -> List[str]:
        """
        負責執行 A2AClient 中的 list_skills 流程，依照 A2AClient 的流程需求處理 list_skills 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[str]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        try:
            import requests
            response = requests.get(f"{self.server_url}/skills", timeout=10)
            response.raise_for_status()
            return response.json().get("skills", [])
        except Exception as e:
            return []


class AgentNetwork:
    """
    負責在 protocols.a2a.implementation 中封裝 AgentNetwork，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        name: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, name: str = "Agent Network"):
        """
        負責執行 AgentNetwork 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.name = name
        self.agents = {}  # agent_name -> agent_url

    def add_agent(self, agent_name: str, agent_url: str):
        """
        負責執行 AgentNetwork 中的 add_agent 流程，將新的輸入資料合併到目前物件狀態或流程紀錄中。
        
        Args:
            agent_name: 目前執行或需要記錄的代理節點識別資訊。
            agent_url: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.agents[agent_name] = agent_url

    def get_agent(self, agent_name: str) -> A2AClient:
        """
        負責執行 AgentNetwork 中的 get_agent 流程，依照 AgentNetwork 的流程需求處理 get_agent 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            agent_name: 目前執行或需要記錄的代理節點識別資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 A2AClient。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found in network")

        return A2AClient(self.agents[agent_name])

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        負責執行 AgentNetwork 中的 list_agents 流程，依照 AgentNetwork 的流程需求處理 list_agents 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [
            {"name": name, "url": url}
            for name, url in self.agents.items()
        ]

    def discover_agents(self, urls: List[str]) -> int:
        """
        負責執行 AgentNetwork 中的 discover_agents 流程，依照 AgentNetwork 的流程需求處理 discover_agents 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            urls: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 int。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        discovered = 0
        for url in urls:
            try:
                client = A2AClient(url)
                info = client.get_info()
                if "name" in info and "error" not in info:
                    self.add_agent(info["name"], url)
                    discovered += 1
            except Exception:
                continue
        return discovered


class AgentRegistry:
    """
    負責在 protocols.a2a.implementation 中封裝 AgentRegistry，封裝代理節點的推理、工具使用、訊息傳遞或協作控制邏輯。
    
    Args:
        name: 此流程需要使用的輸入資料。
        description: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """

    def __init__(self, name: str = "Agent Registry", description: str = "Central agent registry"):
        """
        負責執行 AgentRegistry 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            name: 此流程需要使用的輸入資料。
            description: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.name = name
        self.description = description
        self.registered_agents = {}

    def register_agent(self, agent_name: str, agent_url: str, metadata: Optional[Dict[str, Any]] = None):
        """
        負責執行 AgentRegistry 中的 register_agent 流程，依照 AgentRegistry 的流程需求處理 register_agent 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            agent_name: 目前執行或需要記錄的代理節點識別資訊。
            agent_url: 此流程需要使用的輸入資料。
            metadata: 目前流程所需的上下文、狀態或附加資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.registered_agents[agent_name] = {
            "url": agent_url,
            "metadata": metadata or {},
            "registered_at": __import__("datetime").datetime.now().isoformat()
        }

    def unregister_agent(self, agent_name: str):
        """
        負責執行 AgentRegistry 中的 unregister_agent 流程，依照 AgentRegistry 的流程需求處理 unregister_agent 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            agent_name: 目前執行或需要記錄的代理節點識別資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if agent_name in self.registered_agents:
            del self.registered_agents[agent_name]

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        負責執行 AgentRegistry 中的 list_agents 流程，依照 AgentRegistry 的流程需求處理 list_agents 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 List[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return [
            {"name": name, **info}
            for name, info in self.registered_agents.items()
        ]

    def find_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        負責執行 AgentRegistry 中的 find_agent 流程，依照查詢條件找出符合需求的資料並回傳給呼叫端。
        
        Args:
            agent_name: 目前執行或需要記錄的代理節點識別資訊。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Optional[Dict[str, Any]]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.registered_agents.get(agent_name)

    def get_info(self) -> Dict[str, Any]:
        """
        負責執行 AgentRegistry 中的 get_info 流程，依照 AgentRegistry 的流程需求處理 get_info 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 Dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return {
            "name": self.name,
            "description": self.description,
            "protocol": "A2A",
            "type": "registry",
            "registered_agents": len(self.registered_agents)
        }


# 範例：建立一個簡單的 A2A Agent
def create_example_agent() -> A2AServer:
    """
    負責執行 protocols.a2a.implementation 中的 create_example_agent 流程，建立後續流程需要的物件、資料結構或輸出區塊。
    
    Args:
        無。
    
    Returns:
        執行結果；若函式標註回傳型別，預期型別為 A2AServer。
    
    限制或副作用:
        可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
    """
    if not A2A_AVAILABLE:
        raise ImportError(
            "Cannot create example agent: a2a-sdk library not available. "
            "Install it with: pip install a2a-sdk"
        )

    server = A2AServer(
        name="Example A2A Agent",
        description="A simple example A2A agent",
        version="1.0.0",
        capabilities={"chat": True, "calculation": True}
    )

    # 添加計算技能
    def calculator_skill(text: str) -> str:
        """
        負責執行 protocols.a2a.implementation 中的 calculator_skill 流程，依照 protocols.a2a.implementation 的流程需求處理 calculator_skill 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        # 從文字中提取表達式
        import re
        match = re.search(r'calculate\s+(.+)', text, re.IGNORECASE)
        if match:
            expression = match.group(1).strip()
            try:
                # 安全的表達式求值（僅支援基本運算）
                allowed_chars = set("0123456789+-*/() .")
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters in expression"
                result = eval(expression)
                return f"The result is: {result}"
            except Exception as e:
                return f"Calculation error: {str(e)}"
        return "Please provide an expression to calculate"

    server.add_skill("calculate", calculator_skill)

    # 添加問候技能
    def greeting_skill(text: str) -> str:
        """
        負責執行 protocols.a2a.implementation 中的 greeting_skill 流程，依照 protocols.a2a.implementation 的流程需求處理 greeting_skill 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            text: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 str。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        import re
        match = re.search(r'hello|hi|greet', text, re.IGNORECASE)
        if match:
            return "Hello! I'm an A2A agent. How can I help you today?"
        return "Hi there!"

    server.add_skill("greet", greeting_skill)

    return server


if __name__ == "__main__":
    try:
        # 建立並執行範例 Agent
        agent = create_example_agent()
        print(f"🚀 Starting {agent.name}...")
        print(f"📝 {agent.description}")
        print(f"🔌 Protocol: A2A")
        print(f"📡 Version: {agent.version}")
        print(f"🛠️ Skills: {list(agent.skills.keys())}")
        print()
        agent.run(host="0.0.0.0", port=5000)
    except ImportError as e:
        print(f"❌ {e}")
        print("💡 Install the A2A SDK: pip install a2a-sdk")
        print("📖 Official repository: https://github.com/a2aproject/a2a-python")

i