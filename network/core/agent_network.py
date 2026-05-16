import random
from typing import Any, List, Optional

from builder.evidence_builder import EvidenceBuilder
from ..runners.backward_scorer import BackwardScorer
from .agentnetwork_helper import AgentNetworkHelper
from .agent_neuron import AgentNeuron, NeuronEdge
from .agentneuron_helper import AgentNeuronHelper
from ..runners.early_stop_checker import EarlyStopChecker
from ..policies.early_stop import EarlyStopPolicy
from .network_runtime import NetworkRuntime
from ..runners.stage1_round_runner import Stage1RoundRunner
from ..stage1_judge import Stage1Judge
from ..stage1_result_selector import Stage1ResultSelector
from ..runners.stage2_runner import Stage2Runner
from .task_context import TaskContext
from ..services.tool_manager import ToolManager
from decisionmaker import VerticalSolverFirstDecisionMaker
from memory.base import MemoryConfig


class AgentNetwork:
    """
    多代理推理網路，負責協調 Stage 1 多輪回答、Backward 評分、
    Stage 2 top-k 修正，以及 Final decision 彙整。

    Args:
        model_pool: 可供節點抽樣使用的模型名稱清單。
        agents: 每一輪的 agent 數量。
        rounds: Stage 1 最大推理輪數。
        seed: 隨機抽樣與節點順序使用的種子。
        enable_stage1_tools: 是否允許 Stage 1 使用工具證據。
        enable_shared_memory: 是否啟用 graph memory 相關服務。
        shared_memory_user_id: graph memory namespace 或使用者識別。
        memory_config: 記憶系統設定。
        memory_mode: Stage 1 記憶注入模式。
        debug_print_stage1_first_round_prompt: 是否列印第一輪 Stage 1 prompt。
        enable_stage1_attachment_after_first_round: 是否在第一輪後仍注入附件證據。
        stage1_parallel_workers: Stage 1 每輪 node 平行執行的 worker 數。
        backward_judge_workers: Backward 階段 Stage 1 judge 平行評估的 worker 數。
        stage2_parallel_workers: Stage 2 top-k agent 平行執行的 worker 數。
        final_decision_critic_workers: Final decision critic 平行執行的 worker 數。
    """

    def __init__(
        self,
        model_pool: List = ["nemotron-mini:4b", "phi3:3.8b", "qwen3:4b", "gemma3:4b"],
        agents: int = 4,
        rounds: int = 5,
        seed: int = 2026,
        enable_stage1_tools: bool = False,
        enable_shared_memory: bool = False,
        shared_memory_user_id: str = "network_shared",
        memory_config: Optional[MemoryConfig] = None,
        memory_mode: str = "disabled",
        debug_print_stage1_first_round_prompt: bool = False,
        enable_stage1_attachment_after_first_round: bool = False,
        stage1_parallel_workers: int | None = None,
        stage2_parallel_workers: int | None = None,
        final_decision_critic_workers: int | None = None,
        backward_judge_workers: int | None = None,
    ):
        """
        初始化 AgentNetwork 的模型池、節點數、記憶設定、工具管理器、
        runner、judge、early-stop policy 與 runtime。

        Args:
            model_pool: 可供節點抽樣使用的模型名稱清單。
            agents: 每一輪的 agent 數量。
            rounds: Stage 1 最大推理輪數。
            seed: 隨機抽樣與節點順序使用的種子。
            enable_stage1_tools: 是否允許 Stage 1 使用工具證據。
            enable_shared_memory: 是否啟用 graph memory 相關服務。
            shared_memory_user_id: graph memory namespace 或使用者識別。
            memory_config: 記憶系統設定。
            memory_mode: Stage 1 記憶注入模式。
            debug_print_stage1_first_round_prompt: 是否列印第一輪 Stage 1 prompt。
            enable_stage1_attachment_after_first_round: 是否在第一輪後仍注入附件證據。
            stage1_parallel_workers: Stage 1 每輪 node 平行執行的 worker 數。
            backward_judge_workers: Backward 階段 Stage 1 judge 平行評估的 worker 數。
            stage2_parallel_workers: Stage 2 top-k agent 平行執行的 worker 數。
            final_decision_critic_workers: Final decision critic 平行執行的 worker 數。
        """
        self.model_pool = model_pool
        self.agents = agents
        self.rounds = rounds

        self.rng = random.Random(seed)
        self.enable_shared_memory = enable_shared_memory
        self.shared_memory_user_id = shared_memory_user_id
        valid_memory_modes = {"disabled", "stage1_first_round_only"}
        if memory_mode not in valid_memory_modes:
            raise ValueError(
                f"Unsupported memory_mode: {memory_mode}. Expected one of "
                f"{sorted(valid_memory_modes)}"
            )
        self.memory_mode = memory_mode
        self.debug_print_stage1_first_round_prompt = debug_print_stage1_first_round_prompt
        self.enable_stage1_attachment_after_first_round = enable_stage1_attachment_after_first_round
        self.stage1_parallel_workers = stage1_parallel_workers
        self.stage2_parallel_workers = stage2_parallel_workers
        self.final_decision_critic_workers = final_decision_critic_workers
        self.backward_judge_workers = backward_judge_workers
        
        self.top_k = 3
        self.enable_stage2 = True
        self.enable_stage1_tools = enable_stage1_tools
        self.last_top_k_indices = []
        self.last_stage2_outputs = []
        self.last_final_decision = None
        self.last_early_stop_decision = None
        self.last_early_stop_trace = []
        self.current_question = None
        self.current_task_context: TaskContext = TaskContext()
        self.last_stage1_result = None
        self.last_importance = None
        self.last_stage1_activation_trace = []

        self.network_helper = AgentNetworkHelper()
        self.stage1_judge = Stage1Judge()
        self.stage1_result_selector = Stage1ResultSelector(helper=self.network_helper)
        self.final_decision_maker = VerticalSolverFirstDecisionMaker(
            critic_parallel_workers=self.final_decision_critic_workers,
        )
        self.early_stop_policy = EarlyStopPolicy()
        self.early_stop_checker = EarlyStopChecker(self.early_stop_policy)
        self.stage1_round_runner = Stage1RoundRunner()
        self.backward_scorer = BackwardScorer()
        self.stage2_runner = Stage2Runner()

        self.memory_config = self.network_helper.clone_memory_config(memory_config)
        self.tool_manager = ToolManager(
            memory_config=self.memory_config,
            shared_memory_user_id=self.shared_memory_user_id,
            enable_shared_memory=self.enable_shared_memory,
        )
        self.memory_tool = getattr(self.tool_manager, "memory_tool", None)
        

        self.runtime = self.init_runtime()
        self.init_nn()

    
    def init_nn(self) -> None:
        """
        初始化 Stage 1 的多輪節點與相鄰輪之間的邊。

        每一輪會依 `model_pool` 抽樣建立 `agents` 個節點，並將前一輪
        節點連到下一輪節點，供 backward importance 傳遞使用。
        """
        self.nodes, self.edges = [], []

        first_round_models = self.network_helper.sample_model_name_for_round(self, self.agents)
        for model_name in first_round_models:
            node = AgentNeuron(model_name=model_name)
            node.rng = self.rng
            node.runtime = self.runtime
            self.nodes.append(node)

        agents_last_round = self.nodes[: self.agents]

        for _rid in range(1, self.rounds):
            round_models = self.network_helper.sample_model_name_for_round(self, self.agents)
            new_round_nodes = []

            for model_name in round_models:
                node = AgentNeuron(model_name=model_name)
                node.rng = self.rng
                node.runtime = self.runtime
                self.nodes.append(node)
                new_round_nodes.append(node)

                for a1 in agents_last_round:
                    self.edges.append(NeuronEdge(a1, node))

            agents_last_round = new_round_nodes

        self.activation = AgentNeuronHelper().listwise_ranker_2
        self.activation_cost = 1
    
    
    
    def init_runtime(self) -> NetworkRuntime:
        """
        建立 NetworkRuntime，並掛上 EvidenceBuilder 與目前工具、記憶設定。

        Returns:
            已初始化並可供整個 workflow 共用的 `NetworkRuntime`。
        """
        runtime = NetworkRuntime(
            self.tool_manager,
            memory_tool=self.memory_tool,
            memory_config=self.memory_config,
            shared_memory_user_id=self.shared_memory_user_id,
            enable_shared_memory=self.enable_shared_memory,
            memory_mode=self.memory_mode,
            debug_print_stage1_first_round_prompt=self.debug_print_stage1_first_round_prompt,
        )
        runtime.evidence_builder = EvidenceBuilder(
            tool_manager=self.tool_manager,
            memory_tool=self.memory_tool,
            runtime=runtime,
        )
        runtime.enable_stage1_tools = self.enable_stage1_tools
        runtime.enable_stage1_attachment_after_first_round = self.enable_stage1_attachment_after_first_round
        return runtime

    def set_task_context(self, context: dict[str, Any] | TaskContext | None) -> TaskContext:
        task_context = TaskContext.from_dict(context)
        if not task_context.memory_namespace:
            data = task_context.to_dict()
            data["memory_namespace"] = str(self.shared_memory_user_id or "").strip()
            task_context = TaskContext.from_dict(data)
        self.current_task_context = task_context
        if task_context.question:
            self.current_question = task_context.question
        if self.runtime is not None:
            self.runtime.set_task_context(task_context)
        return task_context

    def get_task_context(self) -> TaskContext:
        if isinstance(getattr(self, "current_task_context", None), TaskContext):
            return self.current_task_context
        task_context = TaskContext()
        self.current_task_context = task_context
        return task_context

    @property
    def current_context(self) -> dict[str, Any]:
        return self.get_task_context().to_dict()

    @current_context.setter
    def current_context(self, context: dict[str, Any] | TaskContext | None) -> None:
        task_context = TaskContext.from_dict(context)
        self.current_task_context = task_context
        if task_context.question:
            self.current_question = task_context.question
        if hasattr(self, "runtime") and self.runtime is not None:
            self.runtime.set_task_context(task_context)

    

    def _forward_legacy(self, question):
        """
        舊版 Stage 1 forward 流程。

        目前正式流程已移到 `Stage1RoundRunner`，此方法保留作為對照與回退。

        Args:
            question: 使用者問題或 benchmark 題目。

        Returns:
            `(majority_answer, response_count, completions, prompt_tokens, completion_tokens)`。
        """
        def build_activation_trace_entry(round_id: int, node_indices: list[int]) -> dict:
            """
            建立單一 Stage 1 輪次的 activation trace。

            Args:
                round_id: Stage 1 輪次。
                node_indices: 此輪被啟用的 node index。

            Returns:
                包含 node 回答、reply、啟用狀態與注入 context 的 trace dict。
            """
            return {
                "round": round_id,
                "node_indices": list(node_indices),
                "nodes": [
                    {
                        "idx": idx,
                        "model_name": getattr(self.nodes[idx], "model_name", None),
                        "active": bool(getattr(self.nodes[idx], "active", False)),
                        "answer": self.nodes[idx].get_answer(),
                        "reply": self.nodes[idx].get_reply(),
                        "stage1_reflection_context": getattr(self.nodes[idx], "stage1_reflection_context", ""),
                        "stage1_attachment_context": getattr(self.nodes[idx], "stage1_attachment_context", ""),
                    }
                    for idx in node_indices
                ],
            }

        def get_completions():
            """
            彙整每個 agent 在各輪的 reply。

            Returns:
                依 agent 分組的多輪 reply；未啟用的節點以 `None` 表示。
            """
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(self.nodes[idx].get_reply())
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.network_helper.set_allnodes_deactivated(self)
        self.last_stage1_activation_trace = []
        self.last_early_stop_decision = None
        self.last_early_stop_trace = []
        assert self.rounds > 2

        loop_indices = list(range(self.agents))
        self.rng.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(f"Round 0, node {idx + 1} starting reply")
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        self.last_stage1_activation_trace.append(build_activation_trace_entry(0, activated_indices))


        loop_indices = list(range(self.agents, self.agents * 2))
        self.rng.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print("=" * 20)
            print(f"Round 1, node {idx + 1} starting reply")
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        self.last_stage1_activation_trace.append(build_activation_trace_entry(1, activated_indices))
        stage1_stopped = self._check_stage1_early_stop(1)


        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents * 2))
        for rid in range(2, self.rounds):
            if stage1_stopped:
                break
            print("=" * 20)
            print(f"Round {rid}")
            print(f"Previous activated nodes: {activated_indices}\n")
            if self.agents > 3:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))
                self.rng.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]

                if self.runtime is not None and hasattr(self.runtime, "measure"):
                    with self.runtime.measure(
                        "stage1_activation_ranker_llm",
                        stage="stage1_activation_ranker",
                        category="llm_call",
                        event_type="llm_call",
                        metadata={"agent_id": "activation_ranker", "model_name": "gpt-oss:20b", "round": rid},
                        input_summary=str(question or "")[:240],
                    ) as latency:
                        tops, prompt_tokens, completion_tokens = self.activation(
                            shuffled_replies,
                            question,
                            "gpt-oss:20b",
                        )
                        latency.metadata["token_usage"] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        }
                        latency.metadata["output_summary"] = str(tops)
                else:
                    tops, prompt_tokens, completion_tokens = self.activation(
                        shuffled_replies,
                        question,
                        "gpt-oss:20b",
                    )
                if self.runtime is not None:
                    self.runtime.record_token_usage(
                        {
                            "stage": "stage1_activation_ranker",
                            "agent_id": "activation_ranker",
                            "model_name": "gpt-oss:20b",
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "round": rid,
                        }
                    )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                idx_mask = list(map(lambda x: idxs[indices[x]] % self.agents, tops))
                resp_cnt += self.activation_cost

            loop_indices = list(range(self.agents * rid, self.agents * (rid + 1)))
            self.rng.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    print(rid, idx)
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
            self.last_stage1_activation_trace.append(build_activation_trace_entry(rid, idxs))
            if self._check_stage1_early_stop(rid):
                break

        completions = get_completions()

        active_answer_indices = [
            idx for idx, node in enumerate(self.nodes)
            if getattr(node, "active", False) and str(node.get_answer() or "").strip()
        ]
        if self.last_early_stop_decision and self.last_early_stop_decision.get("representative"):
            majority_answer = self.last_early_stop_decision.get("representative")
        else:
            answers = [self.nodes[idx].get_answer() for idx in active_answer_indices]
            clusters = self.network_helper.cluster_answers(answers)
            majority_answer = self.network_helper.select_cluster_representative(clusters)

        return majority_answer, resp_cnt, completions, total_prompt_tokens, total_completion_tokens

    def forward(self, question):
        result = self.stage1_round_runner.run(self, question)
        return (
            result.majority_answer,
            result.response_count,
            result.completions,
            result.prompt_tokens,
            result.completion_tokens,
        )

    def _check_stage1_early_stop(self, round_id: int) -> bool:
        return self.early_stop_checker.check_stage1_round(self, round_id)

    def _backward_legacy(self, result):
        """
        舊版 backward 評分流程。

        目前正式流程已移到 `BackwardScorer`。此方法會從最後一個有
        active node 的輪次開始，用 Stage 1 judge 評估候選，再沿著邊把
        importance 往前傳遞。

        Args:
            result: Stage 1 forward 選出的候選答案。

        Returns:
            所有節點的 importance 分數清單。
        """
        flag_last = False
        for rid in range(self.rounds - 1, -1, -1):
            if not flag_last:
                if len([idx for idx in range(self.agents * rid, self.agents * (rid + 1)) if self.nodes[idx].active]) > 0:
                    flag_last = True
                else:
                    continue

                active_indices = [
                    idx
                    for idx in range(self.agents * rid, self.agents * (rid + 1))
                    if self.nodes[idx].active
                ]

                scored = {}
                total_score = 0.0
                for idx in active_indices:
                    node = self.nodes[idx]
                    if self.runtime is not None and hasattr(self.runtime, "measure"):
                        with self.runtime.measure(
                            "stage1_judge_llm",
                            stage="stage1_judge",
                            category="llm_call",
                            event_type="llm_call",
                            metadata={
                                "agent_id": f"stage1_judge_node_{idx}",
                                "model_name": getattr(self.stage1_judge, "judge_model_name", "unknown"),
                                "node_idx": idx,
                            },
                            input_summary=str(self.current_question or "")[:240],
                        ) as latency:
                            evaluation = self.stage1_judge.evaluate_stage1_candidate(
                                question=self.current_question or "",
                                reasoning=node.reasoning,
                                final_answer=node.get_answer(),
                            )
                            prompt_tokens = int(evaluation.get("prompt_tokens", 0) or 0)
                            completion_tokens = int(evaluation.get("completion_tokens", 0) or 0)
                            latency.metadata["token_usage"] = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            }
                            latency.metadata["output_summary"] = str(evaluation.get("judge_reasoning", "") or "")[:240]
                    else:
                        evaluation = self.stage1_judge.evaluate_stage1_candidate(
                            question=self.current_question or "",
                            reasoning=node.reasoning,
                            final_answer=node.get_answer(),
                        )
                    raw_score = max(float(evaluation.get("score", 0.0)), 0.0)
                    score = self.stage1_judge.adjust_stage1_importance(evaluation)
                    node.stage1_judge_is_acceptable = evaluation.get("is_acceptable", False)
                    node.stage1_judge_score = raw_score
                    node.stage1_judge_adjusted_score = score
                    node.stage1_judge_approved_answer = evaluation.get("approved_answer", "")
                    node.stage1_judge_suggested_fix = evaluation.get("suggested_fix", "")
                    node.stage1_judge_revised_answer = evaluation.get("revised_answer", "")
                    node.stage1_judge_reasoning = evaluation.get("judge_reasoning", "")
                    node.stage1_judge_used_fallback = evaluation.get("used_fallback", False)
                    if self.runtime is not None:
                        self.runtime.record_token_usage(
                            {
                                "stage": "stage1_judge",
                                "agent_id": f"stage1_judge_node_{idx}",
                                "model_name": getattr(self.stage1_judge, "judge_model_name", "unknown"),
                                "prompt_tokens": evaluation.get("prompt_tokens", 0),
                                "completion_tokens": evaluation.get("completion_tokens", 0),
                                "node_idx": idx,
                            }
                        )
                    scored[idx] = score
                    total_score += score

                if total_score <= 0 and active_indices:
                    uniform_score = 1 / len(active_indices)
                    for idx in range(self.agents * rid, self.agents * (rid + 1)):
                        self.nodes[idx].importance = uniform_score if idx in active_indices else 0
                else:
                    for idx in range(self.agents * rid, self.agents * (rid + 1)):
                        if idx in scored and total_score > 0:
                            self.nodes[idx].importance = scored[idx] / total_score
                        else:
                            self.nodes[idx].importance = 0
            else:
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += edge.weight * edge.a2.importance

        if self.runtime is not None and hasattr(self.runtime, "measure"):
            with self.runtime.measure(
                "stage1_result_selector_llm",
                stage="stage1_result_selector",
                category="llm_call",
                event_type="llm_call",
                metadata={
                    "agent_id": "stage1_result_selector",
                    "model_name": getattr(self.stage1_result_selector, "judge_model_name", "unknown"),
                },
                input_summary=str(self.current_question or "")[:240],
            ) as latency:
                refined_result = self.stage1_result_selector.select_stage1_result_with_judge(
                    self.nodes,
                    question=self.current_question,
                    fallback_answer=result,
                )
                prompt_tokens = int(getattr(self.stage1_result_selector, "last_prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(self.stage1_result_selector, "last_completion_tokens", 0) or 0)
                latency.metadata["token_usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                latency.metadata["output_summary"] = str(refined_result or "")[:240]
        else:
            refined_result = self.stage1_result_selector.select_stage1_result_with_judge(
                self.nodes,
                question=self.current_question,
                fallback_answer=result,
            )
        if self.runtime is not None:
            self.runtime.record_token_usage(
                {
                    "stage": "stage1_result_selector",
                    "agent_id": "stage1_result_selector",
                    "model_name": getattr(self.stage1_result_selector, "judge_model_name", "unknown"),
                    "prompt_tokens": getattr(self.stage1_result_selector, "last_prompt_tokens", 0),
                    "completion_tokens": getattr(self.stage1_result_selector, "last_completion_tokens", 0),
                }
            )
        self.last_stage1_result = refined_result
        return [node.importance for node in self.nodes]

    def backward(self, result):
        return self.backward_scorer.score(self, result)

    def _run_stage2_legacy(
        self,
        question: str,
        top_k_indices: list[int],
        stage1_result: str = None,
        importance: list[float] = None,
    ):
        """
        舊版 Stage 2 top-k agent 執行流程。

        目前正式流程已移到 `Stage2Runner`，並支援 top-k agents 平行化。

        Args:
            question: 使用者問題或 benchmark 題目。
            top_k_indices: 從 backward importance 選出的 top-k node index。
            stage1_result: Stage 1 的代表答案。
            importance: 所有節點的 importance 分數。

        Returns:
            Stage 2 每個 top-k agent 的輸出 trace。
        """
        tool_manager = self.network_helper.ensure_tool_manager(self)
        stage2_outputs = []
        runtime = getattr(self, "runtime", None)
        shared_search_bundle = None
        if runtime is not None:
            runtime.clear_stage2_shared_state()
            runtime.current_stage2_stage1_result = stage1_result
            runtime.current_stage2_top_k_answers = [
                str(self.nodes[idx].get_answer() or "").strip()
                for idx in top_k_indices
                if str(self.nodes[idx].get_answer() or "").strip()
            ]
            runtime.current_stage2_judge_scores = [
                float(getattr(self.nodes[idx], "stage1_judge_score", 0.0) or 0.0)
                for idx in top_k_indices
            ]
            shared_search_bundle = runtime.prepare_shared_stage2_search(
                question=question,
                router_model_name=None,
            )
            if shared_search_bundle is not None:
                print(
                    "[SHARED-SEARCH] "
                    f"enabled={bool(shared_search_bundle.get('enabled'))} "
                    f"queries={shared_search_bundle.get('queries', [])}"
                )

        for idx in top_k_indices:
            node = self.nodes[idx]
            print("=" * 20)
            print(f"Stage2 Agent Start - idx={idx}, model={getattr(node, 'model_name', None)}")
            print("=" * 20)
            try:
                evidence_bundle = self.stage2_runner._build_agent_evidence_bundle(
                    self,
                    idx=idx,
                    question=question,
                    tool_manager=tool_manager,
                    shared_search_bundle=shared_search_bundle,
                )
                result = node.activate_stage2(
                    question=question,
                    stage1_result=stage1_result,
                    importance=importance[idx] if importance is not None else None,
                    evidence_bundle=evidence_bundle,
                )

                stage2_outputs.append(
                    {
                        "agent_idx": idx,
                        "model_name": getattr(node, "model_name", None),
                        "answer": result.get("answer"),
                        "reply": result.get("reply"),
                        "tool_usage": result.get("tool_usage", []),
                        "routing": result.get("routing", {}),
                        "attachment_context": getattr(node, "stage2_attachment_context", ""),
                        "search_context": getattr(node, "stage2_search_context", ""),
                        "solver_context": getattr(node, "stage2_solver_context", ""),
                        "memory_context": getattr(node, "stage2_memory_context", ""),
                        "rag_context": getattr(node, "stage2_rag_context", ""),
                        "success": result.get("success", True),
                        "error": result.get("error"),
                    }
                )

            except Exception as e:
                result = {
                    "answer": None,
                    "reply": None,
                    "tool_usage": [],
                    "success": False,
                    "error": str(e),
                }
                stage2_outputs.append(
                    {
                        "agent_idx": idx,
                        "model_name": getattr(node, "model_name", None),
                        "answer": None,
                        "reply": None,
                        "tool_usage": [],
                        "routing": {},
                        "attachment_context": getattr(node, "stage2_attachment_context", ""),
                        "search_context": getattr(node, "stage2_search_context", ""),
                        "solver_context": getattr(node, "stage2_solver_context", ""),
                        "memory_context": getattr(node, "stage2_memory_context", ""),
                        "rag_context": getattr(node, "stage2_rag_context", ""),
                        "success": False,
                        "error": str(e),
                    }
                )

            print("=" * 20)
            print(f"Stage2 Agent End - idx={idx}, success={result.get('success', False)}")
            print("=" * 20)

        return stage2_outputs

    def run_stage2(
        self,
        question: str,
        top_k_indices: list[int],
        stage1_result: str = None,
        importance: list[float] = None,
    ):
        return self.stage2_runner.run(
            self,
            question=question,
            top_k_indices=top_k_indices,
            stage1_result=stage1_result,
            importance=importance,
        )

    def forward_two_stage(self, question: str, context: dict[str, Any] | TaskContext | None = None):
        """
        執行完整兩階段推理流程。

        流程包含：設定 TaskContext、Stage 1 forward、Backward 評分、
        top-k agent 選擇、Stage 2 修正，以及 Final decision 彙整。

        Args:
            question: 使用者問題或 benchmark 題目。
            context: 題目相關的 TaskContext 或可轉成 TaskContext 的 dict。

        Returns:
            包含 final result、Stage 1/Stage 2 trace、importance、token usage
            與 early-stop 資訊的結果 dict。
        """
        base_context = context if context is not None else getattr(self, "current_task_context", TaskContext()).to_dict()
        task_context = TaskContext.from_dict(base_context).with_question(question)
        self.set_task_context(task_context)
        self.current_question = question
        if self.runtime is not None:
            self.runtime.prepare_shared_attachment_evidence(question)

        stage1_result, resp_cnt, completions, prompt_tokens, completion_tokens = self.forward(question)
        self.last_stage1_result = stage1_result
        print("Stage 1 result:", stage1_result)

        importance = self.backward(stage1_result)
        stage1_result = self.last_stage1_result
        self.last_importance = importance
        print("Stage 1 result after judge-aware selection:", stage1_result)
        print("Importance scores:", importance)

        top_k_indices = self.network_helper.select_top_k_agents(
            self,
            importance=importance,
            top_k=self.top_k,
        )
        self.last_top_k_indices = top_k_indices

        if not self.enable_stage2:
            return {
                "final_result": stage1_result,
                "stage1_result": stage1_result,
                "top_k_indices": top_k_indices,
                "stage2_outputs": [],
                "resp_cnt": resp_cnt,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completions": completions,
                "importance": importance,
                "early_stop_decision": self.last_early_stop_decision,
                "early_stop_trace": self.last_early_stop_trace,
            }

        if not top_k_indices:
            return {
                "final_result": stage1_result,
                "stage1_result": stage1_result,
                "top_k_indices": top_k_indices,
                "stage2_outputs": [],
                "resp_cnt": resp_cnt,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completions": completions,
                "importance": importance,
                "early_stop_decision": self.last_early_stop_decision,
                "early_stop_trace": self.last_early_stop_trace,
            }

        stage2_outputs = self.run_stage2(
            question=question,
            top_k_indices=top_k_indices,
            stage1_result=stage1_result,
            importance=importance,
        )
        self.last_stage2_outputs = stage2_outputs

        final_result = self.network_helper.finalize_stage2_results(self, stage2_outputs)

        if not final_result:
            final_result = stage1_result

        decision_final_result = (self.last_final_decision or {}).get("final_result")
        if decision_final_result is not None and final_result != decision_final_result:
            print(
                "[WARN] forward_two_stage final_result differs from last_final_decision['final_result']: "
                f"returned={final_result!r}, decision={decision_final_result!r}"
            )

        return {
            "final_result": final_result,
            "stage1_result": stage1_result,
            "top_k_indices": top_k_indices,
            "stage2_outputs": stage2_outputs,
            "resp_cnt": resp_cnt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "completions": completions,
            "importance": importance,
            "early_stop_decision": self.last_early_stop_decision,
            "early_stop_trace": self.last_early_stop_trace,
        }
