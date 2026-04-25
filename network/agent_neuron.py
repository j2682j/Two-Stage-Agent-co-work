import random
from typing import Any

from parser import AgentReplyParser, Stage1ReplyParser, Stage2ReplyParser, try_parse_json
from .agentneuron_helper import AgentNeuronHelper
from builder.evidence_builder import EvidenceBuilder


class AgentNeuron:
    def __init__(self, model_name, parse_json=try_parse_json):
        self.model_name = model_name
        self.rng = random.Random(2026)
        self.runtime = None
        self.reasoning = ""
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.to_edges = []
        self.from_edges = []
        self.question = None
        self.reply_parser = AgentReplyParser(parse_json=parse_json)
        self.stage1_reply_parser = Stage1ReplyParser()
        self.stage2_reply_parser = Stage2ReplyParser()
        self.helper = AgentNeuronHelper()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.stage1_judge_is_acceptable = False
        self.stage1_judge_score = 0.0
        self.stage1_judge_adjusted_score = 0.0
        self.stage1_reflection_context = ""

        self.stage2_reply = None
        self.stage2_answer = ""
        self.stage2_reasoning = ""
        self.stage2_tool_usage = []
        self.stage2_success = False
        self.stage2_error = None
        self.stage2_search_context = ""
        self.stage2_memory_context = ""
        self.stage2_rag_context = ""

    def _build_stage_evidence(self, question: str, stage: str) -> dict[str, Any]:
        runtime = getattr(self, "runtime", None)
        tool_manager = getattr(runtime, "tool_manager", None)
        builder = getattr(runtime, "evidence_builder", None)
        if builder is None and tool_manager is not None:
            builder = EvidenceBuilder(
                tool_manager=tool_manager,
                memory_tool=getattr(runtime, "memory_tool", None),
                runtime=runtime,
            )

        if builder is None:
            return {
                "tool_usage": [],
                "tool_context": "",
                "memory_context": "",
                "rag_context": "",
            }

        evidence = builder.build(
            question=question,
            agent_id=getattr(self, "model_name", "unknown_agent"),
            stage=stage,
            router_model_name=self.model_name,
        )

        if runtime is not None:
            runtime.record_tool_trace(
                {
                    "agent_id": getattr(self, "model_name", "unknown_agent"),
                    "stage": stage,
                    "question": question,
                    "tool_usage": evidence.get("tool_usage", []),
                    "routing": evidence.get("routing", {}),
                }
            )

        return evidence

    def parse_reply(
        self,
        reply: str,
        expected_weight_count: int | None,
        require_final_answer: bool = True,
    ) -> dict[str, Any]:
        return self.reply_parser.parse(reply, expected_weight_count, require_final_answer=require_final_answer)

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def deactivate(self):
        self.reasoning = ""
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.stage1_judge_is_acceptable = False
        self.stage1_judge_score = 0.0
        self.stage1_judge_adjusted_score = 0.0
        self.stage1_reflection_context = ""

        self.stage2_reply = None
        self.stage2_answer = ""
        self.stage2_reasoning = ""
        self.stage2_tool_usage = []
        self.stage2_success = False
        self.stage2_error = None
        self.stage2_search_context = ""
        self.stage2_memory_context = ""
        self.stage2_rag_context = ""

    def get_context(self):
        # 先放入 system prompt，再蒐集目前可用的前序 agent 回覆
        contexts = [{"role": "system", "content": self.helper.SYSTEM_PROMPT}]
        formers = [
            ({"reasoning": edge.a1.reasoning, "final_answer": edge.a1.answer}, eid)
            for eid, edge in enumerate(self.from_edges)
            if edge.a1.reply is not None and edge.a1.active
        ]
        return contexts, formers

    def activate(self, question):
        self.question = question
        self.active = True

        contexts, formers = self.get_context()

        original_idxs = [mess[1] for mess in formers]
        print(f"原始順序: {original_idxs}")
        self.rng.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]
        print(f"打亂後順序: {shuffled_idxs}")

        tool_context = ""
        reflection_context = ""
        self.stage1_reflection_context = ""
        runtime = getattr(self, "runtime", None)
        memory_tool = getattr(runtime, "memory_tool", None)
        is_first_round = len(formers) == 0
        memory_mode = getattr(runtime, "memory_mode", "disabled")
        stage1_reflection_enabled = memory_mode in {"stage1_first_round_only", "final_decision"}
        if stage1_reflection_enabled and is_first_round:
            try:
                reflection_context = self.helper.build_stage1_reflection_context(
                    question=question,
                    memory_tool=memory_tool,
                    runtime=runtime,
                )
                if reflection_context.strip():
                    print(f"[{self.model_name}] reflection_context:\n{reflection_context}\n")
            except Exception as e:
                print(f"[WARN] stage1 reflection context failed for {self.model_name}: {e}")
        self.stage1_reflection_context = reflection_context.strip()

        if getattr(runtime, "enable_stage1_tools", False):
            try:
                evidence = self._build_stage_evidence(question, stage="stage1")
                tool_context = evidence.get("tool_context", "") or ""
            except Exception as e:
                print(f"[WARN] stage1 evidence building failed for {self.model_name}: {e}")

        stage1_message = self.helper.build_stage1_message(
            question,
            formers,
            tool_context=tool_context,
            reflection_context=reflection_context,
        )
        if (
            is_first_round
            and runtime is not None
            and isinstance(stage1_message, dict)
            and stage1_message.get("role") == "user"
        ):
            runtime.last_stage1_first_round_prompt = str(stage1_message.get("content", ""))
            if getattr(runtime, "debug_print_stage1_first_round_prompt", False):
                print(f"[{self.model_name}] first_round_stage1_prompt:\n{runtime.last_stage1_first_round_prompt}\n")
        contexts.append(stage1_message)
        self.reply, self.prompt_tokens, self.completion_tokens = self.helper.generate_answer(
            contexts,
            self.model_name,
        )

        try:
            parsed = self.stage1_reply_parser.parse(
                self.reply,
                expected_weight_count=len(formers),
            )
            print(f"[解析後的Stage1 Agent回答]: {parsed}")
        except Exception as e:
            print("[parse_reply] first attempt failed")
            print(f"[parse_reply] error: {type(e).__name__}: {e}")

            repair_prompt = self.helper.build_repair_prompt(len(formers))
            repair_contexts = contexts + [
                {"role": "assistant", "content": self.reply},
                {"role": "user", "content": repair_prompt},
            ]

            retry_reply, retry_prompt_tokens, retry_completion_tokens = self.helper.generate_answer(
                repair_contexts,
                self.model_name,
            )

            self.prompt_tokens += retry_prompt_tokens
            self.completion_tokens += retry_completion_tokens
            self.reply = retry_reply

            print("[INFO] 已完成修復重試")
            try:
                parsed = self.stage1_reply_parser.parse(
                    self.reply,
                    expected_weight_count=len(formers),
                )
                print(f"[重試解析後的Stage1 Agent回答]: {parsed}")
            except Exception as retry_error:
                fallback_reasoning = self.stage1_reply_parser.extract_reasoning(
                    self.reply
                ).strip()
                fallback_answer = self.stage1_reply_parser.extract_final_answer(
                    self.reply
                )
                fallback_weights = self.stage1_reply_parser.fallback_weights(
                    len(formers)
                )

                if fallback_answer:
                    print("[WARN] repair 後仍無法完整解析，改用保守容錯結果")
                    parsed = {
                        "reasoning": fallback_reasoning,
                        "final_answer": str(fallback_answer).strip(),
                        "weights": fallback_weights,
                    }
                else:
                    print("[WARN] repair 後仍無法解析，將此節點標記為失敗並跳過")
                    print(f"[WARN] parse_reply error: {type(retry_error).__name__}: {retry_error}")
                    self.reasoning = ""
                    self.answer = ""
                    self.active = False
                    for edge in self.from_edges:
                        edge.weight = 0
                    return

        self.reasoning = parsed["reasoning"]
        self.answer = parsed["final_answer"]
        weights = parsed["weights"]
        print("=" * 20)
        print("Agent回覆已解析")
        print("=" * 20)

        if len(weights) != len(formers):
            raise ValueError(
                f"Weight count mismatch: expected {len(formers)}, got {len(weights)}.\n"
                f"Weights: {weights}\n"
                f"Reply: {self.reply}"
            )

        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights = [weight for _, weight, _ in sorted_pairs]
        formers = [(former, eid) for eid, _, former in sorted_pairs]

        for idx, (_, eid) in enumerate(formers):
            self.from_edges[eid].weight = weights[idx] / 5 if 0 < weights[idx] <= 5 else (1 if weights[idx] > 5 else 0)

        total = sum(self.from_edges[eid].weight for _, eid in formers)
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        elif formers:
            # 如果沒有有效權重，就平均分配給所有前序節點
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

    def activate_stage2(
        self,
        question: str,
        tool_manager=None,
        stage1_result: str = None,
        importance: float = None,
        shared_search_bundle: dict[str, Any] | None = None,
    ):
        self.question = question
        self.active = True
        self.stage2_reply = None
        self.stage2_answer = ""
        self.stage2_reasoning = ""
        self.stage2_tool_usage = []
        self.stage2_success = False
        self.stage2_error = None
        self.stage2_search_context = ""
        self.stage2_memory_context = ""
        self.stage2_rag_context = ""

        tool_usage = []
        tool_context = ""

        runtime = getattr(self, "runtime", None)
        builder = getattr(runtime, "evidence_builder", None)
        if builder is None and tool_manager is not None:
            builder = EvidenceBuilder(
                tool_manager=tool_manager,
                memory_tool=getattr(runtime, "memory_tool", None),
                runtime=runtime,
            )

        if builder is not None:
            try:
                evidence = builder.build(
                    question=question,
                    agent_id=getattr(self, "model_name", "unknown_agent"),
                    stage="stage2",
                    router_model_name=self.model_name,
                    shared_search_bundle=shared_search_bundle,
                )

                tool_usage = evidence["tool_usage"]
                tool_context = evidence["tool_context"]
                self.stage2_tool_usage = tool_usage
                self.stage2_search_context = evidence["search_context"].replace("Search evidence:\n", "", 1)
                self.stage2_memory_context = evidence["memory_context"].replace("Memory evidence:\n", "", 1)
                self.stage2_rag_context = evidence["rag_context"].replace("RAG evidence:\n", "", 1)

                if evidence["search_context"]:
                    print(f"[{self.model_name}] search_context after summary:\n{evidence['search_context']}\n")
                if self.stage2_memory_context.strip():
                    print(f"[{self.model_name}] memory_context:\n{self.stage2_memory_context}\n")
                if self.stage2_rag_context.strip():
                    print(f"[{self.model_name}] rag_context:\n{self.stage2_rag_context}\n")

                if runtime is not None:
                    runtime.record_tool_trace(
                        {
                            "agent_id": getattr(self, "model_name", "unknown_agent"),
                            "stage": "stage2",
                            "question": question,
                            "tool_usage": tool_usage,
                            "routing": evidence.get("routing", {}),
                        }
                    )

            except Exception as e:
                self.stage2_tool_usage = tool_usage
                self.stage2_success = False
                self.stage2_error = str(e)
                self.stage2_reply = None
                self.stage2_reasoning = ""
                self.stage2_answer = ""
                self.stage2_memory_context = ""
                self.stage2_rag_context = ""

                return {
                    "answer": None,
                    "reply": None,
                    "tool_usage": tool_usage,
                    "success": False,
                    "error": str(e),
                }

        if not tool_context:
            tool_context = "No tool result available."

        contexts = self.helper.build_stage2_prompts(
            question=question,
            stage1_result=stage1_result,
            importance=importance,
            tool_context=tool_context,
        )

        try:
            self.stage2_reply, self.prompt_tokens, self.completion_tokens = self.helper.generate_answer(
                contexts,
                self.model_name,
            )

            parsed = self.parse_reply(self.stage2_reply, expected_weight_count=None)
            print(f"[解析後的Stage 2 Agent回答]: {parsed}")
            self.stage2_reasoning = parsed["reasoning"]
            self.stage2_answer = parsed["final_answer"]
            self.stage2_success = True
            self.stage2_error = None

            return {
                "answer": self.stage2_answer,
                "reply": self.stage2_reply,
                "tool_usage": tool_usage,
                "success": True,
                "error": None,
            }

        except Exception as e:
            fallback = self.stage2_reply_parser.parse_fallback(self.stage2_reply)
            print(f"[重試解析後的Stage 2 Agent回答]: {fallback}")
            if fallback is not None:
                self.stage2_reasoning = fallback["reasoning"]
                self.stage2_answer = fallback["final_answer"]
                self.stage2_tool_usage = tool_usage
                self.stage2_success = True
                self.stage2_error = None

                return {
                    "answer": self.stage2_answer,
                    "reply": self.stage2_reply,
                    "tool_usage": tool_usage,
                    "success": True,
                    "error": None,
                }

            self.stage2_reasoning = ""
            self.stage2_answer = ""
            self.stage2_tool_usage = tool_usage
            self.stage2_success = False
            self.stage2_error = str(e)
            self.stage2_reply = None

            return {
                "answer": None,
                "reply": None,
                "tool_usage": tool_usage,
                "success": False,
                "error": str(e),
            }

    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        structured_formers = [former for former, _ in formers]
        contexts.append(self.helper.build_stage1_message(self.question, structured_formers))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class NeuronEdge:
    def __init__(self, a1, a2):
        # a1 -> a2 的連線邊，weight 表示 a1 對 a2 的影響權重
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0


