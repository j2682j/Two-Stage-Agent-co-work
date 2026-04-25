# 專案說明報告

本專案是一個以多代理協作為核心的兩階段推理系統，主要目標是讓多個小模型先各自提出初步答案，再透過評分、篩選、工具使用與最終裁決流程，得到較穩定的最終答案。整體設計特別重視以下幾件事：

- 讓 `stage1` 承擔多樣化探索與初步推理
- 讓 `stage2` 專注在少數高品質候選的查證與修正
- 用 judge 與 final decision 機制降低單一節點誤答的影響
- 把 memory 與 RAG 視為可切換的輔助能力，而不是永遠強塞進 prompt

---

## 1. Network 架構說明

### 1.1 整體架構概觀

系統主體由以下幾層組成：

- `AgentNetwork`
  - 負責建立整個多代理網路
  - 控制 `forward / backward / run_stage2 / forward_two_stage`
  - 持有共享的 `ToolManager`、`MemoryTool`、`NetworkRuntime`
- `AgentNeuron`
  - 代表單一代理節點
  - 在不同 round 中產生答案、解析答案、維護邊權重
- `AgentNetworkHelper`
  - 承接初始化、收尾、top-k 選取、final decision 前處理等網路級輔助工作
- `Stage1Judge`
  - 負責對 stage1 候選進行評分
- `Stage1ResultSelector`
  - 在 backward 後，根據 judge 資訊重新選出較可信的 stage1 結果
- `VerticalSolverFirstDecisionMaker`
  - 對 stage2 候選做 judge、critic、solver revision 與最終答案決策
- `ToolManager`
  - 系統的共享工具入口
  - 管理 calculator、search、memory、RAG
- `MemoryTool`
  - 記憶系統的共享入口
  - 負責對外 action 入口與參數處理
- `MemoryManager`
  - 核心記憶管理邏輯
  - 管理 working / episodic / semantic 等記憶類型
- `EvidenceBuilder`
  - 統一蒐集 calculator、search、RAG、memory 等 evidence

---

### 1.2 AgentNetwork 的角色與工作內容

`AgentNetwork` 是整個系統的 orchestrator，主要負責：

- 初始化多輪 `AgentNeuron`
- 建立共享 runtime
- 在 `forward()` 中控制 stage1 的節點啟動順序
- 在 `backward()` 中計算重要度分數
- 在 `run_stage2()` 中只針對 top-k 節點做第二階段推理
- 在 `forward_two_stage()` 中串起完整兩階段流程

目前 `AgentNetwork` 本體刻意保持精簡，主要只留下：

- `__init__`
- `forward`
- `backward`
- `run_stage2`
- `forward_two_stage`

其餘初始化與輔助流程盡量收斂到 `AgentNetworkHelper`。

---

### 1.3 AgentNeuron 的角色與工作內容

`AgentNeuron` 代表網路中的單一節點，主要負責：

- 接收題目與前序節點回覆
- 在 `activate()` 中執行 stage1 推理
- 在 `activate_stage2()` 中執行 stage2 推理
- 呼叫 prompt builder 組 prompt
- 呼叫 parser 解析模型輸出
- 維護從前序節點進來的邊權重
- 保存本節點的 reasoning / final answer / stage2 output

在設計上，`AgentNeuron` 比較像「可被 `AgentNetwork` 調度的推理單元」，而不是整個流程的決策者。

---

## 2. 兩階段流程：Forward 與 Backward 的詳細步驟

### 2.1 Stage 1：`forward()`

`forward(question)` 的核心目標是：讓多個節點在多輪互相參考的情況下，形成一批可評估的候選答案。

詳細流程如下：

1. 重置所有節點狀態
2. 第 0 輪：
   - 啟動第一批 `agents` 個節點
   - 每個節點獨立回答問題
3. 第 1 輪：
   - 啟動第二批 `agents` 個節點
   - 每個節點可以讀取前一輪可用節點的回覆
4. 第 2 輪之後：
   - 若代理數量夠多，會先對上一輪回答做 listwise ranking / activation 選擇
   - 只讓較有潛力的局部路徑繼續擴展
5. 每個節點在 `activate()` 中：
   - 先蒐集 `formers`
   - 視設定決定是否加入 stage1 第一輪反思記憶
   - 視設定決定是否啟用 stage1 tools
   - 組出 stage1 prompt
   - 呼叫模型產生回答
   - 解析 `REASONING / FINAL_ANSWER / WEIGHTS`
   - 將 weights 回寫到對應邊上
6. 最後由 active 節點的答案形成 stage1 候選集

若系統開啟 stage1 共識機制，某些輪次中可提前結束；否則會完整跑完預定 rounds。

---

### 2.2 Backward：`backward()`

`backward(result)` 的目標是：根據最後一層 active 節點的品質，將重要度往前傳回整張網路。

流程如下：

1. 從最後一個有 active node 的 round 開始
2. 對最後一層 active node：
   - 呼叫 `Stage1Judge.evaluate_stage1_candidate(...)`
   - 取得：
     - `is_acceptable`
     - `score`
     - `approved_answer`
     - `suggested_fix`
     - `revised_answer`
     - `judge_reasoning`
3. 將 judge score 經 `adjust_stage1_importance(...)` 修正後，形成該節點的 raw importance
4. 將最後一層 importance 正規化
5. 對更早的 round，依照邊權重往前回傳：

\[
I_i = \sum_{j \in \text{children}(i)} w_{ij} \cdot I_j
\]

其中：

- \(I_i\)：節點 \(i\) 的 importance
- \(w_{ij}\)：從節點 \(i\) 指向子節點 \(j\) 的邊權重

6. 最後再透過 `Stage1ResultSelector` 根據 judge-aware 規則，重新選擇較可信的 stage1 result

---

### 2.3 Stage 2：`run_stage2()`

`run_stage2()` 只會對 top-k 節點做第二階段處理。

流程如下：

1. 根據 `importance` 選出 top-k 節點
2. 對每個 top-k node：
   - 執行 `activate_stage2()`
   - 進一步使用 tools / search / RAG
   - 產生較完整的 stage2 answer
3. 收集每個 stage2 candidate 的：
   - `answer`
   - `reply`
   - `tool_usage`
   - `success`
   - `error`

這些 stage2 traces 會交給 final decision maker。

---

### 2.4 完整流程：`forward_two_stage()`

`forward_two_stage()` 的順序如下：

1. `forward(question)` 取得初步 stage1 結果
2. `backward(stage1_result)` 算出 importance
3. 根據 importance 選出 top-k
4. 對 top-k 跑 `run_stage2()`
5. 呼叫 final decision maker 整合 stage2 候選
6. 若 final decision 失敗，退回 stage1 result

---

## 3. Importance Score 的意義，以及 Judge 如何給分

### 3.1 Importance Score 的意義

Importance Score 代表：

- 某個 stage1 節點在整體推理圖中有多重要
- 它對最後答案形成的貢獻程度有多高
- 它是否值得進入 stage2

Importance 不是單純看這個節點有沒有作答，而是綜合考慮：

- 這個節點的 stage1 judge score
- 它是否被 judge 視為 acceptable
- 它是否有 approved answer
- 後續節點是否依賴它

---

### 3.2 最後一層節點的初始 importance

對最後一個 active round 的節點，會先算 adjusted score：

\[
s_i = f(\text{judge score}, \text{acceptable}, \text{approved answer}, \text{revised answer})
\]

再做正規化：

\[
I_i = \frac{s_i}{\sum_{k \in \text{active last round}} s_k}
\]

若所有 \(s_i \le 0\)，則平均分配。

---

### 3.3 Stage1Judge 如何給分

`Stage1Judge` 的評分重點有三個：

1. reasoning 是否合理
2. final answer 是否正確
3. final answer 的單位是否符合題目要求

#### 評分區間

- `0 ~ 3`
  - reasoning 或 answer 大幅錯誤
  - 或 final unit 明顯不對
- `4 ~ 6`
  - 部分有用，但仍有重大缺陷
- `7 ~ 8`
  - 大致正確，但仍有小問題或解釋不足
- `9 ~ 10`
  - reasoning 正確、答案正確、單位也正確

#### 額外修正規則

`adjust_stage1_importance(...)` 會再根據 judge 結果做加權：

- 若不 acceptable，低分會被大幅折損
- 若 acceptable 且分數高，會有小幅增益
- 有 `approved_answer` 會再加一些 bonus

因此 importance 不等於 judge raw score，而是 judge raw score 的「策略化版本」。

---

## 4. Final Decision 的產生方式

Final decision 由 [vertical_solver_first_decision_maker.py](/C:/paper/decisionmaker/vertical_solver_first_decision_maker.py) 負責，流程如下：

### 4.1 先過濾成功的 stage2 候選

只保留：

- `success=True`
- 有可用 answer / reply 的候選

若沒有成功候選，final decision 直接失敗。

---

### 4.2 先做 stage2 candidate judge rerank

對每個 successful output，會再做一次候選評估：

- `is_acceptable`
- `score`
- `revised_answer`
- `judge_reasoning`

這是 final decision 前的第一層過濾。

---

### 4.3 選 solver candidate

目前 solver 會從 stage2 候選中選一個主答案，排序準則近似：

\[
\text{score tuple} =
(
\mathbb{1}[\text{acceptable}],
\text{stage2 judge score},
\text{importance}
)
\]

也就是優先考慮：

1. 是否 acceptable
2. stage2 judge score
3. 該節點在 stage1 的 importance

---

### 4.4 其他候選作為 critics

選中的 solver candidate 之外，其餘 successful stage2 outputs 會作為 critics：

- 每個 critic 會拿 solver answer 與自己的 answer 做比較
- 若認為 solver answer 不夠好，就提出 critique 與 revised answer

---

### 4.5 Solver revision

若有可信 critique，solver 會再進行 revision：

- 讀取 critique
- 視情況修正自己的 final answer

這形成一個小型的「solver-first, critic-after」垂直決策流程。

---

### 4.6 失敗回退

若 final decision 沒有成功得到結果，系統會退回：

- `stage1_result`

確保整個 pipeline 不至於因 final decision 失敗而完全無答案。

---

## 5. 如何使用 Memory 與 RAG

### 5.1 Memory 的設計定位

目前 memory 的設計不是把舊答案直接拿來查表，而是：

- 保存反思 lesson
- 保存過去案例
- 在需要的時候做 retrieval

目前 shared memory 的主要入口是：

- `ToolManager`
  - `MemoryTool`
    - `MemoryManager`

`MemoryManager` 不再由 `AgentNetwork` 單獨初始化。

---

### 5.2 Memory 類型

目前實作中主要有：

- `working`
  - 短期工作記憶
- `episodic`
  - 案例型記憶、過去經驗
- `semantic`
  - lesson、規則、可泛化知識

在 GAIA 這類 benchmark 中，較適合保存的是：

- `episodic`: 這次答錯的案例摘要
- `semantic`: 可泛化的 lesson / error pattern

---

### 5.3 Memory 的三種使用模式

目前 `AgentNetwork` 支援：

#### 1. `memory_mode="disabled"`

- 完全不使用 memory
- 系統純粹依靠目前題目本身作答

#### 2. `memory_mode="final_decision"`

- stage1 第一輪會使用 reflection memory
- final decision 前也會額外檢索 memory reflection

#### 3. `memory_mode="stage1_first_round_only"`

- 只有第一輪 `AgentNeuron` 會在回答前讀取反思記憶
- final decision 不使用 memory

這個設計是為了方便做 A/B 實驗，觀察 memory 注入位置對表現的影響。

---

### 5.4 Stage1 第一輪如何使用 memory

若模式為 `stage1_first_round_only`，第一輪 node 在 `activate()` 時會：

1. 根據題目做 memory retrieval
2. 從 semantic memory 中取出少量 lesson
3. 壓縮成短的結構化 context，例如：

```text
Relevant reflection rules:
- error_type=surface_form_guess | lesson=Do not infer the answer from surface wording alone. | use_when=Use for token, label, code name, or hidden mapping questions where surface text may be misleading.
```

再放進 stage1 prompt。

這種做法的目的，是讓反思記憶成為「錯誤避免規則」，而不是讓 prompt 無限制變長。

---

### 5.5 Final decision 模式如何使用 memory

若模式為 `final_decision`，會保留 stage1 的 reflection 使用方式，並且在 final decision 前再額外做一次 memory reflection：

1. 根據當前問題建立 lesson / case 查詢
2. 檢索：
   - `semantic`
   - `episodic`
   - `working`
3. 組成：
   - `Relevant memory lessons`
   - `Relevant memory cases`
4. 傳給 decision maker 與 critic / solver revision prompt

此時 memory 被視為：

- lesson
- error-avoidance rule
- case reference

而不是舊答案查表。

---

### 5.6 RAG 的使用方式

目前 RAG 是透過：

- `AgentNeuron`
- `EvidenceBuilder`
- `ToolManager.rag_tool`

這條路徑使用。

RAG 的特性如下：

1. `ToolManager` 建立共享 `RAGTool`
2. `EvidenceBuilder` 先做 tool routing
3. 若 `use_rag=True`，就呼叫：
   - `rag_tool.get_relevant_context(query=question, limit=3)`
4. 回傳內容會包成：

```text
RAG evidence:
...
```

5. 最後併進 `tool_context`

因此目前：

- `RAG` 會主動進 prompt
- `memory` 不一定會主動進 prompt，而是依 `memory_mode` 決定

---

## 6. 補充：目前實務上值得注意的坑

雖然本報告以目前穩定架構為主，但在開發過程中，我們已經遇到幾類典型問題：

### 6.1 parser 與格式脆弱性

- 曾出現 regex 壞掉，導致 stage1 fallback parse 直接例外
- 修正方式：
  - 收斂 parser 結構
  - 把 `base_parser` 與 `stage_parser` 分工整理清楚

### 6.2 stage1 / stage2 success 欄位混亂

- 曾同時存在 `success` 與 `stage2_success`
- 導致候選被錯誤覆蓋
- 修正方式：
  - 收斂成單一 canonical `success`

### 6.3 shared memory 入口不一致

- 曾經同時有 `runtime.memory_manager` 與 `tool_manager.memory_tool.memory_manager`
- 容易造成兩套記憶來源
- 修正方式：
  - 以 `MemoryTool` 為唯一共享入口

### 6.4 memory lesson 過度泛化

- 若把上一題的反思直接套到下一題，可能會導致錯誤遷移
- 因此目前 memory 更適合：
  - 當 error-avoidance rule
  - 不適合作為答案查表

---

## 7. 總結

這個專案目前已形成一套相對清楚的兩階段多代理推理流程：

- `stage1` 做探索與多樣化候選生成
- `backward` 用 judge-aware importance 評估每個節點的重要性
- `stage2` 對 top-k 候選進一步查證與修正
- `final decision` 用 solver-first + critics 的方式整合最終答案
- `memory` 與 `RAG` 以可切換、可實驗的方式接入

若未來要繼續強化，最值得投入的方向通常會是：

- 讓 stage1 judge 與 importance 更穩
- 讓 final decision 更能辨別 critic 品質
- 讓 memory retrieval 更精準對應 error pattern
- 讓 RAG 與 search 在 evidence 層的角色更清楚分工
