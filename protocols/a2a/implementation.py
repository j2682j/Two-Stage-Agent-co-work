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
    """A2A 伺服器（使用 Flask 提供 HTTP API）"""

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        capabilities: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 A2A 伺服器

        Args:
            name: Agent 名稱
            description: Agent 描述
            version: Agent 版本
            capabilities: Agent 能力描述
        """
        self.name = name
        self.description = description
        self.version = version
        self.capabilities = capabilities or {}
        self.skills = {}

    def add_skill(self, skill_name: str, func):
        """添加技能到伺服器"""
        self.skills[skill_name] = func
        return func

    def skill(self, skill_name: str):
        """裝飾器方式添加技能"""
        def decorator(func):
            self.add_skill(skill_name, func)
            return func
        return decorator

    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """執行伺服器（使用 Flask 提供 HTTP API）"""
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
            """取得 Agent 資訊"""
            return jsonify(self.get_info())

        @app.route('/skills', methods=['GET'])
        def list_skills():
            """列出所有技能"""
            return jsonify({
                "skills": list(self.skills.keys()),
                "count": len(self.skills)
            })

        @app.route('/execute/<skill_name>', methods=['POST'])
        def execute_skill(skill_name):
            """執行指定技能"""
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
            """通用問答介面（自動選擇技能）"""
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
            """健康檢查"""
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
        """取得伺服器資訊"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "protocol": "A2A",
            "skills": list(self.skills.keys())
        }


class A2AClient:
    """A2A 客戶端（通過 HTTP 與 A2AServer 通信）"""

    def __init__(self, server_url: str):
        """
        初始化 A2A 客戶端

        Args:
            server_url: 伺服器 URL（例如：http://localhost:5000）
        """
        self.server_url = server_url.rstrip('/')

    def ask(self, question: str) -> str:
        """
        向 Agent 提問（通用介面）

        Args:
            question: 問題文字

        Returns:
            Agent 的回答
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
        執行指定技能

        Args:
            skill_name: 技能名稱
            text: 輸入文字

        Returns:
            執行結果
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
        """取得 Agent 資訊"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to get agent info: {str(e)}"}

    def list_skills(self) -> List[str]:
        """列出 Agent 的技能"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/skills", timeout=10)
            response.raise_for_status()
            return response.json().get("skills", [])
        except Exception as e:
            return []


class AgentNetwork:
    """基於官方 a2a-sdk 庫的 Agent 網路（概念性實現）"""

    def __init__(self, name: str = "Agent Network"):
        """
        初始化 Agent 網路

        Args:
            name: 網路名稱
        """
        self.name = name
        self.agents = {}  # agent_name -> agent_url

    def add_agent(self, agent_name: str, agent_url: str):
        """
        添加 Agent 到網路

        Args:
            agent_name: Agent 名稱
            agent_url: Agent URL
        """
        self.agents[agent_name] = agent_url

    def get_agent(self, agent_name: str) -> A2AClient:
        """
        取得網路中的 Agent

        Args:
            agent_name: Agent 名稱

        Returns:
            A2A 客戶端實例
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found in network")

        return A2AClient(self.agents[agent_name])

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有 Agent"""
        return [
            {"name": name, "url": url}
            for name, url in self.agents.items()
        ]

    def discover_agents(self, urls: List[str]) -> int:
        """
        從 URL 列表中發現 Agent

        Args:
            urls: URL 列表

        Returns:
            發現的 Agent 數量
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
    """基於官方 a2a-sdk 庫的 Agent 註冊中心（概念性實現）"""

    def __init__(self, name: str = "Agent Registry", description: str = "Central agent registry"):
        """
        初始化 Agent 註冊中心

        Args:
            name: 註冊中心名稱
            description: 註冊中心描述
        """
        self.name = name
        self.description = description
        self.registered_agents = {}

    def register_agent(self, agent_name: str, agent_url: str, metadata: Optional[Dict[str, Any]] = None):
        """註冊 Agent"""
        self.registered_agents[agent_name] = {
            "url": agent_url,
            "metadata": metadata or {},
            "registered_at": __import__("datetime").datetime.now().isoformat()
        }

    def unregister_agent(self, agent_name: str):
        """注銷 Agent"""
        if agent_name in self.registered_agents:
            del self.registered_agents[agent_name]

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有註冊的 Agent"""
        return [
            {"name": name, **info}
            for name, info in self.registered_agents.items()
        ]

    def find_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """查找特定 Agent"""
        return self.registered_agents.get(agent_name)

    def get_info(self) -> Dict[str, Any]:
        """取得註冊中心資訊"""
        return {
            "name": self.name,
            "description": self.description,
            "protocol": "A2A",
            "type": "registry",
            "registered_agents": len(self.registered_agents)
        }


# 範例：建立一個簡單的 A2A Agent
def create_example_agent() -> A2AServer:
    """建立一個範例 A2A Agent"""
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
        """計算數學表達式"""
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
        """生成問候語"""
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