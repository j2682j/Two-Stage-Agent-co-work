import random
from typing import List, Optional

from builder.evidence_builder import EvidenceBuilder
from .agentnetwork_helper import AgentNetworkHelper
from .agent_neuron import AgentNeuron, NeuronEdge
from .agentneuron_helper import AgentNeuronHelper
from .network_runtime import NetworkRuntime
from .stage1_judge import Stage1Judge
from .stage1_result_selector import Stage1ResultSelector
from .tool_manager import ToolManager
from decisionmaker import VerticalSolverFirstDecisionMaker
from memory import MemoryConfig


class AgentNetwork:
    """
    AgentNetwork 會串接多個 AgentNeuron，負責：
    - forward: 讓多個節點逐輪推理並收集 stage1 候選結果
    - backward: 根據 judge 與邊權重回傳每個節點的重要性分數
    """

    def __init__(
        self,
        model_pool: List = ["nemotron-mini:4b", "minicpm3_4b:latest", "qwen3:4b", "gemma3:4b"],
        agents: int = 4,
        rounds: int = 5,
        seed: int = 2026,
        enable_stage1_tools: bool = False,
        enable_shared_memory: bool = True,
        shared_memory_user_id: str = "network_shared",
        memory_config: Optional[MemoryConfig] = None,
        memory_mode: str = "disabled",
        debug_print_stage1_first_round_prompt: bool = False,
    ):
        self.model_pool = model_pool
        self.agents = agents
        self.rounds = rounds

        self.rng = random.Random(seed)
        self.enable_shared_memory = enable_shared_memory
        self.shared_memory_user_id = shared_memory_user_id
        valid_memory_modes = {"disabled", "final_decision", "stage1_first_round_only"}
        if memory_mode not in valid_memory_modes:
            raise ValueError(
                f"Unsupported memory_mode: {memory_mode}. Expected one of "
                f"{sorted(valid_memory_modes)}"
            )
        self.memory_mode = memory_mode
        self.debug_print_stage1_first_round_prompt = debug_print_stage1_first_round_prompt
        
        self.top_k = 3
        self.enable_stage2 = True
        self.enable_stage1_tools = enable_stage1_tools
        self.last_top_k_indices = []
        self.last_stage2_outputs = []
        self.last_final_decision = None
        self.current_question = None
        self.last_stage1_result = None
        self.last_importance = None
        self.last_stage1_activation_trace = []

        self.network_helper = AgentNetworkHelper()
        self.stage1_judge = Stage1Judge()
        self.stage1_result_selector = Stage1ResultSelector(helper=self.network_helper)
        self.final_decision_maker = VerticalSolverFirstDecisionMaker()

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
        return runtime

    

    def forward(self, question):
        def build_activation_trace_entry(round_id: int, node_indices: list[int]) -> dict:
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
                    }
                    for idx in node_indices
                ],
            }

        def get_completions():
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
        assert self.rounds > 2

        loop_indices = list(range(self.agents))
        self.rng.shuffle(loop_indices)

        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            print(f"第 0 輪，第{idx + 1}個Node 開始回覆")
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
            print(f"第 1 輪，第{idx + 1}個Node 開始回覆")
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
        self.last_stage1_activation_trace.append(build_activation_trace_entry(1, activated_indices))


        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents * 2))
        for rid in range(2, self.rounds):
            print("=" * 20)
            print(f"第 {rid} 輪")
            print(f"全部實際被 activate 且有回覆的 node: {activated_indices}\n")
            if self.agents > 3:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))
                self.rng.shuffle(indices)
                shuffled_replies = [replies[idx] for idx in indices]

                tops, prompt_tokens, completion_tokens = self.activation(
                    shuffled_replies,
                    question,
                    "gpt-oss:20b",
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

        completions = get_completions()

        active_answer_indices = [
            idx for idx, node in enumerate(self.nodes)
            if getattr(node, "active", False) and str(node.get_answer() or "").strip()
        ]
        answers = [self.nodes[idx].get_answer() for idx in active_answer_indices]
        clusters = self.network_helper.cluster_answers(answers)
        majority_answer = self.network_helper.select_cluster_representative(clusters)

        return majority_answer, resp_cnt, completions, total_prompt_tokens, total_completion_tokens

    def backward(self, result):
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

        refined_result = self.stage1_result_selector.select_stage1_result_with_judge(
            self.nodes,
            question=self.current_question,
            fallback_answer=result,
        )
        self.last_stage1_result = refined_result
        return [node.importance for node in self.nodes]

    def run_stage2(
        self,
        question: str,
        top_k_indices: list[int],
        stage1_result: str = None,
        importance: list[float] = None,
    ):
        tool_manager = self.network_helper.ensure_tool_manager(self)
        stage2_outputs = []
        runtime = getattr(self, "runtime", None)
        shared_search_bundle = None
        if runtime is not None:
            runtime.clear_stage2_shared_state()
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
                result = node.activate_stage2(
                    question=question,
                    tool_manager=tool_manager,
                    stage1_result=stage1_result,
                    importance=importance[idx] if importance is not None else None,
                    shared_search_bundle=shared_search_bundle,
                )

                stage2_outputs.append(
                    {
                        "agent_idx": idx,
                        "model_name": getattr(node, "model_name", None),
                        "answer": result.get("answer"),
                        "reply": result.get("reply"),
                        "tool_usage": result.get("tool_usage", []),
                        "search_context": getattr(node, "stage2_search_context", ""),
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
                        "search_context": getattr(node, "stage2_search_context", ""),
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

    def forward_two_stage(self, question: str):
        self.current_question = question
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
                "[WARN] forward_two_stage final_result 與 last_final_decision['final_result'] 不一致: "
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
        }
