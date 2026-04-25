# Project Cleanup Review

## 目的

這份文件的重點不是改功能，而是幫這個專案變得：

- 更容易讀
- 更容易維護
- 更容易測試
- 更容易分辨「核心程式碼」和「執行產物」

我會先給整體觀察，再給一個建議的整理順序。這份順序刻意設計成「可以一階段一階段做」，避免一次大改把系統弄亂。

---

## 目前專案最明顯的整潔度問題

### 1. Repo hygiene 還沒建立

目前 repo 根目錄沒有看到這些基礎檔案：

- `.gitignore`
- `.gitattributes`
- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`

這會直接造成幾個問題：

- `venv312/`、`__pycache__/`、`memory_data/`、`test_gaia_latest.log`、`test_gaia_compact.log` 這些執行產物很容易混進 repo
- 新成員不容易知道專案該怎麼安裝與執行
- formatter / linter / test command 沒有統一入口

### 2. 目錄命名風格不一致 (Checked)

目前目錄同時混用：

- `network/`
- `decisionmaker/`
- `agents/`
- `builder/`
- `evaluation/`
- `tools/`

也就是有些是 PascalCase，有些是 snake_case。這種混用會讓專案看起來像是不同時期的程式碼直接堆在一起。

### 3. 同時存在兩套 agent 主線

從結構來看，目前有兩條路：

- 舊的通用 agent 路線：`agents/`、`core/agent.py`
- 現在真正 benchmark 主線：`network/agent_network.py`

這本身不一定錯，但目前沒有很明確的分層標記，會讓人不知道：

- 哪一條是正式主線
- 哪一條是 legacy
- 哪些檔案之後還會維護

### 4. 核心檔案偏大，責任過重

目前幾個明顯偏大的檔案：

- `tools/builtin/search_tool.py`：931 行
- `memory/lesson_rule.py`：797 行
- `test/test_gaia.py`：496 行
- `evaluation/gaia_adapter.py`：495 行
- `network/agent_network.py`：463 行
- `network/agent_neuron.py`：455 行

這些檔案不只是長，還混合了多種責任。例如：

- `agent_network.py` 同時在做 orchestration、trace 保存、stage control、fallback
- `test_gaia.py` 同時在做 runner、logging、memory debug、result export
- `search_tool.py` 同時在做 backend dispatch、結果整理、全文抓取、rerank、格式化

### 5. 測試資料、下載快取、執行結果混在一起

`test/` 底下目前不只放測試腳本，還放了：

- `test/data/gaia/2023/...`
- `test/data/gaia/.cache/huggingface/...`
- `gaia_evaluation_results.json`

這會讓 `test/` 同時扮演：

- 測試程式
- fixture
- dataset cache
- output directory

維護成本會高很多。

### 6. logging / trace 邏輯散在多處

目前有大量 `print()` 分散在：

- `network/agent_network.py`
- `network/agent_neuron.py`
- `test/test_gaia.py`
- `tools/builtin/search_tool.py`

這在開發期很好用，但後期會開始出現：

- 哪個 log 是產品行為，哪個是 debug 雜訊，不容易分
- 不同 entrypoint 輸出的格式不一致
- 很多 log 難以結構化保存

### 7. 產物目錄放在 repo 根目錄

目前根目錄可以看到：

- `test_gaia_latest.log`
- `test_gaia_compact.log`
- `memory_data/`
- `venv312/`
- `=5.0.0`

這些都會讓 repo 首頁看起來很亂，也模糊掉「真正需要維護的程式碼」。

---

## 整理方向總原則

這個專案不適合直接大重構。我建議用下面三個原則：

1. 先整理 repo 邊界，再整理模組邊界
2. 先切責任，再改命名
3. 先讓主線更清楚，再處理 legacy

換句話說，不要一開始就搬整個資料夾。先把「哪些是原始碼、哪些是資料、哪些是產物」釐清，後面才會順。

---

## 建議的整理順序

## Phase 1: 先做 Repo Hygiene 

這一階段不碰功能，只做環境與檔案邊界整理。

### 要做的事

- 新增 `.gitignore` (checked)
- 新增 `pyproject.toml` (checked)
- 建立統一的 formatter / linter / test 指令
- 把執行產物集中到單一輸出目錄，例如：(checked)
  - `result/logs/`
  - `result/eval/`
  - `result/memory/`
- 把本地虛擬環境 `venv312/` 從 repo 規則中排除 (checked)
- 把 `memory_data/` 視為本地資料，而不是原始碼 (checked)
- 清掉不明檔案，例如根目錄的 `=5.0.0` (checked)

### 這一階段的成果

- 根目錄只剩：原始碼、文件、設定檔
- 新人進 repo 一眼就知道什麼該讀，什麼不用碰

---

## Phase 2: 把目錄分成「核心主線」和「支線 / legacy」

這一階段要先把專案語意講清楚。

### 我建議的分類

- `network/`
  - 真正的兩階段推理主線
- `decision/`
  - final decision maker
- `memory/`
  - memory system
- `evaluation/`
  - benchmark adapter / evaluator
- `tools/`
  - 可注入工具
- `legacy_agents/`
  - 舊版通用 agent 實作

### 具體方向

- 將 `network/`、`decisionmaker/` 維持一致的小寫命名
- 將 `agents/` 裡現在不在主線上的內容明確標成 `legacy`
- README 第一段就先寫清楚：
  - 正式主線入口在哪裡
  - 哪些模組只是歷史保留

### 這一階段的成果

- 不再讓讀 code 的人同時面對兩條不清楚的架構線

---

## Phase 3: 拆大檔，先拆 orchestration 層

這一階段優先處理「長而重」的核心檔案。

### 第一批建議拆的檔案

- `network/agent_network.py`
- `network/agent_neuron.py`
- `test/test_gaia.py`

### 拆法建議

`agent_network.py` 可以拆成：

- `network/orchestrator.py`
- `network/stage1_flow.py`
- `network/stage2_flow.py`
- `network/runtime_state.py`

`agent_neuron.py` 可以拆成：

- `network/neuron.py`
- `network/neuron_stage1.py`
- `network/neuron_stage2.py`
- `network/neuron_state.py`

`test_gaia.py` 可以拆成：

- `test/gaia_smoke_runner.py`
- `test/gaia_logging.py`
- `test/gaia_memory_trace.py`

### 這一階段的成果

- orchestration 邏輯不再全塞在單一檔案
- 看 stage1 / stage2 / log / trace 時不需要一直上下跳

---

## Phase 4: 拆 rule-heavy 與 tool-heavy 檔案

這一階段處理規則特別多、容易膨脹的檔案。

### 優先對象

- `memory/lesson_rule.py`
- `tools/builtin/search_tool.py`

### 建議拆法

`lesson_rule.py` 拆成：

- `memory/lesson_types.py`
- `memory/lesson_parsing.py`
- `memory/lesson_scoring.py`
- `memory/lesson_retrieval_profile.py`

`search_tool.py` 拆成：

- `tools/builtin/search/backends.py`
- `tools/builtin/search/rerank.py`
- `tools/builtin/search/fetch.py`
- `tools/builtin/search/formatting.py`
- `tools/builtin/search/tool.py`

### 這一階段的成果

- 單個檔案不再同時承擔 backend、ranking、fetch、formatting
- 之後要調 search 策略時，修改範圍會明確很多

---

## Phase 5: 統一 logging / trace 策略

這一階段很重要，因為目前 debug 訊息散得很開。

### 建議方向

- 把即時開發 log 和評測 trace 分開
- 統一用 logger，不要在核心流程裡直接散佈大量 `print()`
- 訂出 3 個層級：
  - `console summary`
  - `debug log`
  - `structured trace`

### 具體做法

- 新增 `logging_config.py`
- 新增 `trace/` 或 `builder/trace/` 下的統一輸出 builder
- `test_gaia.py` 只負責決定寫哪一種 log，不負責組裝所有內容

### 這一階段的成果

- log 更一致
- evaluator / runner / runtime 不會各自長出自己的輸出格式

---

## Phase 6: 統一 configuration 與依賴注入方式

目前很多設定是直接從 `AgentNetwork(...)` 參數一路傳到 runtime、tool manager、memory tool。

### 建議方向

- 建立明確的 config objects，例如：
  - `NetworkConfig`
  - `MemoryConfig`
  - `SearchConfig`
  - `EvaluationConfig`
- 讓 `AgentNetwork` 接收較高階設定物件，而不是太多平行參數

### 好處

- 初始化介面會乾淨很多
- 不容易出現「這個 flag 在 network 存一份，runtime 又存一份」的狀況

---

## Phase 7: 重整測試目錄與資料目錄

目前 `test/` 太像「所有東西都先塞進來」的地方。

### 建議方向

- `tests/`
  - 純測試程式
- `datasets/`
  - 本地 benchmark data
- `artifacts/`
  - log / eval outputs
- `scripts/`
  - 手動執行 runner

### 具體例子

- `test/test_gaia.py` 移到 `scripts/test_gaia.py` 或 `scripts/run_gaia_smoke.py`
- `test/data/gaia/` 移到 `datasets/gaia/`
- `gaia_evaluation_results.json` 移到 `artifacts/eval/`

### 這一階段的成果

- 測試、資料、輸出不再混在一起

---

## Phase 8: 補文件，不是補長文件

這一階段不是寫更多 README，而是補「真的能降低理解成本的文件」。

### 最值得補的 4 份

- `docs/architecture.md`
- `docs/runtime-flow.md`
- `docs/evaluation-flow.md`
- `docs/memory-flow.md`

### 每份文件應該回答的問題

- 這個模組的單一責任是什麼
- 它依賴誰
- 它輸入什麼，輸出什麼
- 哪些是正式主線，哪些是實驗性功能

---

## 我最建議的實作順序

如果你要最有效率，我會照這個順序做：

1. 建 `.gitignore`、`pyproject.toml`、`artifacts/`
2. 把根目錄的 log / venv / memory data 排除掉
3. 把 `network/`、`decisionmaker/` 維持一致命名
4. 在 README 標明主線與 legacy
5. 拆 `test_gaia.py`
6. 拆 `agent_network.py`
7. 拆 `search_tool.py`
8. 拆 `lesson_rule.py`
9. 導入統一 logger / trace
10. 最後才處理更深層的 config refactor

---

## 不建議一開始就做的事

- 不要先全面 rename 所有檔案
- 不要一開始就把 `agents/` 全刪掉
- 不要先做大規模 import path 重寫
- 不要在沒有 `.gitignore` 前就開始大量搬資料

先把 repo 邊界整理乾淨，再動架構，風險最低。

---

## 一句話總結

這個專案目前最大的整潔度問題，不是功能太多，而是：

- 原始碼、資料、產物混在一起
- 新舊架構並存但沒有明確標記
- 核心大檔責任過重

最好的整理方式不是一次大改，而是：

**先清 repo 邊界，再清主線模組，再拆大檔。**
