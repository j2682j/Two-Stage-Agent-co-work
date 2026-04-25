# Two-Stage-Agent-co-work

Two-Stage-Agent-co-work 是一個實驗型的 two-stage agent network 專案，目標是把多個 agent 的候選答案、judge 評分、工具證據、memory 與 RAG 串成一條可觀測的推理與評估流程。

目前專案重點集中在：

- Stage 1：多節點 agent network 產生候選推理與答案。
- Backward / Judge：用 judge-aware scoring 回推節點重要性。
- Stage 2：挑選 top-k 節點進一步使用工具、搜尋、memory、RAG 補強答案。
- Final Decision：用 solver-first + critics 的方式整合 stage2 候選結果。
- Evaluation：支援 GAIA / BFCL 等 benchmark adapter 與 evaluation tool。
- Memory / RAG：提供 working、episodic、semantic memory，以及 Qdrant-backed RAG pipeline。

> 注意：這個 repo 目前仍在整理期。核心模組可用，但 package import、pytest 測試入口、依賴宣告與 lint 狀態仍有待收斂。詳細改善步驟請看 [OPTIMIZATION_EXECUTION_PLAN.md](OPTIMIZATION_EXECUTION_PLAN.md)。

## 專案結構

```text
agents/                 基礎 agent 類型與範例 agent
builder/                evidence、search query、trace builder
context/                上下文組裝
core/                   LLM wrapper、config、message、基礎 agent
decisionmaker/          final decision maker
evaluation/             benchmark adapters 與 GAIA/BFCL evaluators
memory/                 memory manager、memory types、storage、RAG pipeline
network/                two-stage agent network 核心流程
parser/                 stage / ranking / decision parser
prompt/                 prompt builders
protocols/              MCP / A2A / ANP protocol integration
rl/                     RL dataset、reward、trainer
tools/                  tool registry 與內建工具
test/                   目前包含 GAIA smoke runner 與本地 GAIA data
utils/                  logging、serialization、network helper utilities
```

## 核心流程

`network.AgentNetwork` 是主要 orchestrator。典型流程如下：

1. `forward(question)`
   - 建立多輪 agent node。
   - 第一、二輪會啟動多個 agent 產生初始答案。
   - 後續輪次可透過 listwise ranking 選擇要啟動的節點。
   - 收集 active nodes 的 stage1 answers。
2. `backward(stage1_result)`
   - 對最後一層 active nodes 執行 `Stage1Judge`。
   - 根據 judge score、acceptable、approved answer、revised answer 調整 importance。
   - 將 importance 沿著 network edge 往前回推。
   - 用 `Stage1ResultSelector` 選出 judge-aware stage1 result。
3. `run_stage2(question, top_k_indices)`
   - 選出 importance 較高的 top-k nodes。
   - 在 stage2 prompt 中加入 search / RAG / memory / calculator 等 evidence。
   - 產出 stage2 candidate outputs。
4. `forward_two_stage(question)`
   - 串起 stage1、backward、stage2、final decision。
   - 最後交給 `VerticalSolverFirstDecisionMaker` 做 solver + critics 整合。

## 快速開始

此專案目前主要在 Windows + PowerShell + Python 3.12 環境下開發，repo 內已有 `venv312`。

```powershell
cd C:\paper
.\venv312\Scripts\python.exe --version
```

目前已知 `venv312` 使用 Python `3.12.10`。

### 檢查目前環境

```powershell
.\venv312\Scripts\python.exe -m pip check
.\venv312\Scripts\python.exe -m ruff check . --statistics
.\venv312\Scripts\python.exe -m py_compile $(rg --files -g '*.py' -g '!venv312/**' -g '!test/data/**')
```

目前已知狀態：

- `pip check` 會回報 `fastmcp` 與 `websockets` 版本衝突。
- `pytest` 尚未安裝在 `venv312`。
- `ruff check .` 目前有大量 lint issues，包含一批可自動修的 typing/import/f-string 問題，以及少量高風險 correctness issues。

## 最小使用範例

從 repo root 執行時，可以直接使用目前的 top-level import 方式：

```powershell
.\venv312\Scripts\python.exe -c "from network.agent_network import AgentNetwork; print('ok')"
```

建立 network：

```python
from network.agent_network import AgentNetwork

network = AgentNetwork(
    agents=4,
    rounds=5,
    enable_stage1_tools=False,
    enable_shared_memory=True,
    memory_mode="disabled",
)

answer = network.forward_two_stage("What is the capital of France?")
print(answer)
```

實際執行會呼叫設定的 LLM / Ollama / local model，請先確認 `.env` 或環境變數已設定好。

## LLM 設定

核心 LLM wrapper 在 `core/llm.py`，支援 OpenAI-compatible client。常見環境變數：

```powershell
$env:LLM_MODEL_ID="your-model"
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_API_KEY="ollama"
$env:LLM_TIMEOUT="120"
```

也支援 provider-specific variables，例如：

- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`
- `MODELSCOPE_API_KEY`
- `KIMI_API_KEY` / `MOONSHOT_API_KEY`
- `ZHIPU_API_KEY` / `GLM_API_KEY`
- `OLLAMA_HOST`
- `VLLM_HOST`

## Tools

內建工具集中在 `tools/builtin/`：

- `calculator.py`：安全 AST calculator。
- `search_tool.py`：搜尋、fetch、rerank 與多 backend 搜尋整合。
- `memory_tool.py`：memory 讀寫與 debug 工具。
- `rag_tool.py`：RAG context retrieval。
- `terminal_tool.py`：終端工具，目前需要安全性收斂後再作為預設工具使用。
- `gaia_evaluation_tool.py` / `bfcl_evaluation_tool.py`：benchmark evaluation wrappers。

`network.ToolManager` 會註冊 calculator、search、memory，並在 RAG 初始化成功時註冊 RAG。

## Memory 與 RAG

Memory 入口大致如下：

```text
AgentNetwork
  -> ToolManager
    -> MemoryTool
      -> MemoryManager
```

支援的 memory 類型：

- `working`：短期工作記憶。
- `episodic`：案例與執行經驗。
- `semantic`：可重用 lesson / error pattern。

`AgentNetwork` 目前支援三種 memory mode：

- `disabled`：不將 memory 放入推理流程。
- `stage1_first_round_only`：只在 stage1 第一輪加入 reflection memory。
- `final_decision`：stage1 與 final decision 都可使用 memory context。

RAG 主要透過 `RAGTool` 與 `memory/rag/pipeline.py` 使用 Qdrant collection。若本機 Qdrant 可用，初始化時會連線到 configured collection。

## GAIA Evaluation

本地 GAIA 資料在：

```text
test/data/gaia/2023/
```

可用 dataset loader 驗證本地 metadata：

```powershell
.\venv312\Scripts\python.exe -c "from evaluation.benchmarks.gaia.dataset import GAIADataset; ds=GAIADataset(local_data_dir='test/data/gaia', level=1); data=ds.load(); print(len(data))"
```

目前 `test/test_gaia.py` 比較像 GAIA smoke runner，而不是 pytest unit test。建議後續將它移到 `scripts/` 或 `evaluation/runners/`，再建立真正可離線執行的 `tests/`。

## 開發檢查

建議每次整理前先跑：

```powershell
.\venv312\Scripts\python.exe -m py_compile $(rg --files -g '*.py' -g '!venv312/**' -g '!test/data/**')
.\venv312\Scripts\python.exe -m ruff check . --select F821,E722,B006,B023,B904
.\venv312\Scripts\python.exe -m pip check
```

等安裝 `pytest` 並整理測試入口後，再加入：

```powershell
.\venv312\Scripts\python.exe -m pytest --collect-only -q
.\venv312\Scripts\python.exe -m pytest
```

## 目前已知限制

- `pyproject.toml` 尚未完整宣告 runtime / dev / optional dependencies。
- `pytest` 尚未安裝在目前 `venv312`。
- `import paper` 從父層目錄執行仍會失敗，因為專案內混用了 package-relative 與 top-level imports。
- `memory.__all__` 與 `tools.__all__` 目前包含未定義或未啟用的 symbol。
- `ruff check .` 有大量風格與 typing cleanup 工作。
- `TerminalTool` 目前允許 shell/interpreter 並使用 `shell=True`，若要讓 agent 自動呼叫，建議先收緊安全邊界。
- GAIA runner 現在放在 `test/`，但它不是乾淨的 pytest 測試。

## 建議整理順序

1. 補齊 `pyproject.toml` dependency groups，安裝 `pytest`。
2. 解決 `fastmcp` / `websockets` 版本衝突。
3. 修正 package import 與 `__all__` 匯出。
4. 將 GAIA smoke runner 移出 `test/`，建立真正的 pytest tests。
5. 優先修掉高風險 lint：`F821`、`E722`、`B006`、`B023`、`B904`。
6. 再進行可自動修的 ruff cleanup。
7. 統一 logging / trace 格式。
8. 分階段拆分大型模組，例如 `search_tool.py`、RAG pipeline、agent network flow。

## 相關文件

- [OPTIMIZATION_EXECUTION_PLAN.md](OPTIMIZATION_EXECUTION_PLAN.md)：目前檢查結果與分階段優化計畫。
- [PROJECT_CLEANUP_REVIEW.md](PROJECT_CLEANUP_REVIEW.md)：早期 cleanup review 文件。若 git 狀態顯示刪除與未追蹤並存，請先確認檔案追蹤狀態。
