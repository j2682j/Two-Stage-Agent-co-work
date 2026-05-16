import random
from typing import Any

from parser import AgentReplyParser, Stage1ReplyParser, Stage2ReplyParser, try_parse_json
from prompt.contracts import resolve_prompt_contract
from .agentneuron_helper import AgentNeuronHelper
from builder.evidence_builder import EvidenceBuilder
from .stage2_evidence_bundle import Stage2EvidenceBundle
from .trace_record import summarize_text


class AgentNeuron:
    """
    負責在 network.agent_neuron 中封裝 AgentNeuron，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
        parse_json: 此流程需要使用的輸入資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, model_name, parse_json=try_parse_json):
        """
        負責執行 AgentNeuron 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            model_name: 用來呼叫模型或外部服務的模型名稱、客戶端或相關設定。
            parse_json: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        self.stage1_attachment_context = ""

        self.stage2_reply = None
        self.stage2_answer = ""
        self.stage2_reasoning = ""
        self.stage2_tool_usage = []
        self.stage2_success = False
        self.stage2_error = None
        self.stage2_search_context = ""
        self.stage2_memory_context = ""
        self.stage2_rag_context = ""
        self.stage2_attachment_context = ""
        self.stage2_solver_context = ""
        self.stage2_routing = {}

    def _build_stage_evidence(
        self,
        question: str,
        stage: str,
        *,
        include_routed_tools: bool = True,
        include_attachment: bool = True,
    ) -> dict[str, Any]:
        """
        負責執行 AgentNeuron 中的 _build_stage_evidence 流程，依照 AgentNeuron 的流程需求處理 _build_stage_evidence 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            stage: 目前執行的階段、輪次或流程位置。
            include_routed_tools: 控制是否啟用此項資料、功能或處理分支的布林開關。
            include_attachment: 控制是否啟用此項資料、功能或處理分支的布林開關。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
            include_routed_tools=include_routed_tools,
            include_attachment=include_attachment,
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
        """
        負責執行 AgentNeuron 中的 parse_reply 流程，解析輸入內容並萃取後續流程需要使用的結構化資料。
        
        Args:
            reply: 模型、節點或工具產生的候選回覆內容。
            expected_weight_count: 此流程需要使用的輸入資料。
            require_final_answer: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 dict[str, Any]。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.reply_parser.parse(reply, expected_weight_count, require_final_answer=require_final_answer)

    def get_reply(self):
        """
        負責執行 AgentNeuron 中的 get_reply 流程，依照 AgentNeuron 的流程需求處理 get_reply 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.reply

    def get_answer(self):
        """
        負責執行 AgentNeuron 中的 get_answer 流程，依照 AgentNeuron 的流程需求處理 get_answer 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        return self.answer

    def deactivate(self):
        """
        負責執行 AgentNeuron 中的 deactivate 流程，依照 AgentNeuron 的流程需求處理 deactivate 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        self.stage1_attachment_context = ""

        self.stage2_reply = None
        self.stage2_answer = ""
        self.stage2_reasoning = ""
        self.stage2_tool_usage = []
        self.stage2_success = False
        self.stage2_error = None
        self.stage2_search_context = ""
        self.stage2_memory_context = ""
        self.stage2_rag_context = ""
        self.stage2_attachment_context = ""
        self.stage2_solver_context = ""
        self.stage2_routing = {}

    def _record_token_usage(self, stage: str, *, extra: dict[str, Any] | None = None) -> None:
        """
        負責執行 AgentNeuron 中的 _record_token_usage 流程，依照 AgentNeuron 的流程需求處理 _record_token_usage 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            stage: 目前執行的階段、輪次或流程位置。
            extra: 此流程需要使用的輸入資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return
        runtime.record_token_usage(
            {
                "stage": stage,
                "agent_id": getattr(self, "model_name", "unknown_agent"),
                "model_name": getattr(self, "model_name", "unknown"),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                **dict(extra or {}),
            }
        )

    def _generate_answer_with_trace(
        self,
        messages: list[dict[str, Any]],
        *,
        stage: str,
        call_name: str,
    ) -> tuple[str, int, int]:
        runtime = getattr(self, "runtime", None)
        metadata = {
            "agent_id": getattr(self, "model_name", "unknown_agent"),
            "model_name": getattr(self, "model_name", "unknown"),
            "message_count": len(messages or []),
        }
        input_summary = summarize_text((messages or [{}])[-1].get("content", "") if messages else "")
        if runtime is None or not hasattr(runtime, "measure"):
            return self.helper.generate_answer(messages, self.model_name)

        with runtime.measure(
            call_name,
            stage=stage,
            category="llm_call",
            event_type="llm_call",
            metadata=metadata,
            input_summary=input_summary,
        ) as latency:
            content, prompt_tokens, completion_tokens = self.helper.generate_answer(messages, self.model_name)
            latency.metadata["token_usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            latency.metadata["output_summary"] = summarize_text(content)
            return content, prompt_tokens, completion_tokens

    def get_context(self):
        # 先放入 system prompt，再蒐集目前可用的前序 agent 回覆
        """
        負責執行 AgentNeuron 中的 get_context 流程，依照 AgentNeuron 的流程需求處理 get_context 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        contexts = [{"role": "system", "content": self.helper.SYSTEM_PROMPT}]
        formers = [
            ({"reasoning": edge.a1.reasoning, "final_answer": edge.a1.answer}, eid)
            for eid, edge in enumerate(self.from_edges)
            if edge.a1.reply is not None and edge.a1.active
        ]
        return contexts, formers

    def activate(self, question):
        """
        負責執行 AgentNeuron 中的 activate 流程，依照 AgentNeuron 的流程需求處理 activate 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        self.stage1_attachment_context = ""
        runtime = getattr(self, "runtime", None)
        task_context = runtime.get_task_context() if runtime is not None and hasattr(runtime, "get_task_context") else None
        prompt_contract = resolve_prompt_contract(task_context, question=question)
        memory_tool = getattr(runtime, "memory_tool", None)
        is_first_round = len(formers) == 0
        memory_mode = getattr(runtime, "memory_mode", "disabled")
        stage1_reflection_enabled = memory_mode == "stage1_first_round_only"
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

        include_stage1_attachment = (
            runtime.should_include_stage1_attachment(is_first_round)
            if runtime is not None and hasattr(runtime, "should_include_stage1_attachment")
            else False
        )
        if getattr(runtime, "enable_stage1_tools", False) or include_stage1_attachment:
            try:
                evidence = self._build_stage_evidence(
                    question,
                    stage="stage1",
                    include_routed_tools=getattr(runtime, "enable_stage1_tools", False),
                    include_attachment=include_stage1_attachment,
                )
                tool_context = evidence.get("tool_context", "") or ""
                self.stage1_attachment_context = (
                    evidence.get("attachment_context", "").replace("Attachment evidence:\n", "", 1)
                )
                if self.stage1_attachment_context.strip():
                    print(f"[{self.model_name}] stage1 attachment_context:\n{self.stage1_attachment_context}\n")
            except Exception as e:
                print(f"[WARN] stage1 evidence building failed for {self.model_name}: {e}")

        stage1_message = self.helper.build_stage1_message(
            question,
            formers,
            tool_context=tool_context,
            reflection_context=reflection_context,
            contract=prompt_contract,
            task_context=task_context,
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
        self.reply, self.prompt_tokens, self.completion_tokens = self._generate_answer_with_trace(
            contexts,
            stage="stage1_agent",
            call_name="stage1_agent_llm",
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

            repair_prompt = self.helper.build_repair_prompt(
                len(formers),
                contract=prompt_contract,
                task_context=task_context,
                question=question,
            )
            repair_contexts = contexts + [
                {"role": "assistant", "content": self.reply},
                {"role": "user", "content": repair_prompt},
            ]

            retry_reply, retry_prompt_tokens, retry_completion_tokens = self._generate_answer_with_trace(
                repair_contexts,
                stage="stage1_agent_repair",
                call_name="stage1_agent_repair_llm",
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
                    self._record_token_usage("stage1_agent", extra={"node_success": False, "parse_failed": True})
                    self.reasoning = ""
                    self.answer = ""
                    self.active = False
                    for edge in self.from_edges:
                        edge.weight = 0
                    return

        self.reasoning = parsed["reasoning"]
        self.answer = parsed["final_answer"]
        self._record_token_usage("stage1_agent", extra={"node_success": True, "parse_failed": False})
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
        stage1_result: str = None,
        importance: float = None,
        evidence_bundle: Stage2EvidenceBundle | None = None,
    ):
        """
        負責執行 AgentNeuron 中的 activate_stage2 流程，依照 AgentNeuron 的流程需求處理 activate_stage2 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            question: 目前要處理的任務、問題或查詢文字。
            stage1_result: 評估、推理或工具執行後產生的結果與分數資料。
            importance: 此流程需要使用的輸入資料。
            evidence_bundle: Stage2Runner 預先整理好的 Stage-2 evidence bundle。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
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
        self.stage2_attachment_context = ""
        self.stage2_solver_context = ""
        self.stage2_routing = {}

        tool_usage = []
        tool_context = ""

        runtime = getattr(self, "runtime", None)
        task_context = runtime.get_task_context() if runtime is not None and hasattr(runtime, "get_task_context") else None
        prompt_contract = resolve_prompt_contract(task_context, question=question)
        if evidence_bundle is None:
            error = "Stage2EvidenceBundle is required for activate_stage2."
            self.stage2_error = error
            return {
                "answer": None,
                "reply": None,
                "tool_usage": [],
                "routing": {},
                "success": False,
                "error": error,
            }

        if evidence_bundle.error:
            self.stage2_tool_usage = []
            self.stage2_success = False
            self.stage2_error = evidence_bundle.error
            return {
                "answer": None,
                "reply": None,
                "tool_usage": [],
                "routing": evidence_bundle.routing,
                "success": False,
                "error": evidence_bundle.error,
            }

        tool_usage = evidence_bundle.tool_usage
        tool_context = evidence_bundle.tool_context_or_empty()
        self.stage2_tool_usage = tool_usage
        self.stage2_attachment_context = evidence_bundle.attachment_context.replace("Attachment evidence:\n", "", 1)
        self.stage2_search_context = evidence_bundle.search_context.replace("Search evidence:\n", "", 1)
        self.stage2_memory_context = evidence_bundle.memory_context.replace("Memory evidence:\n", "", 1)
        self.stage2_rag_context = evidence_bundle.rag_context.replace("RAG evidence:\n", "", 1)
        self.stage2_solver_context = evidence_bundle.solver_context.replace("Python solver guidance:\n", "", 1)
        self.stage2_routing = evidence_bundle.routing

        if self.stage2_attachment_context.strip():
            print(f"[{self.model_name}] attachment_context:\n{self.stage2_attachment_context}\n")
        if evidence_bundle.search_context:
            print(f"[{self.model_name}] search_context after summary:\n{evidence_bundle.search_context}\n")
        if self.stage2_memory_context.strip():
            print(f"[{self.model_name}] memory_context:\n{self.stage2_memory_context}\n")
        if self.stage2_rag_context.strip():
            print(f"[{self.model_name}] rag_context:\n{self.stage2_rag_context}\n")
        if self.stage2_solver_context.strip():
            print(f"[{self.model_name}] solver_context:\n{self.stage2_solver_context}\n")

        if not tool_context:
            tool_context = "No tool result available."

        contexts = self.helper.build_stage2_prompts(
            question=question,
            stage1_result=stage1_result,
            importance=importance,
            tool_context=tool_context,
            contract=prompt_contract,
            task_context=task_context,
        )

        try:
            self.stage2_reply, self.prompt_tokens, self.completion_tokens = self._generate_answer_with_trace(
                contexts,
                stage="stage2_agent",
                call_name="stage2_agent_llm",
            )

            parsed = self.parse_reply(self.stage2_reply, expected_weight_count=None)
            print(f"[解析後的Stage 2 Agent回答]: {parsed}")
            self.stage2_reasoning = parsed["reasoning"]
            self.stage2_answer = parsed["final_answer"]
            self.stage2_success = True
            self.stage2_error = None
            self._record_token_usage("stage2_agent", extra={"node_success": True})

            return {
                "answer": self.stage2_answer,
                "reply": self.stage2_reply,
                "tool_usage": tool_usage,
                "routing": self.stage2_routing,
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
                self._record_token_usage("stage2_agent", extra={"node_success": True, "parse_fallback": True})

                return {
                    "answer": self.stage2_answer,
                    "reply": self.stage2_reply,
                    "tool_usage": tool_usage,
                    "routing": self.stage2_routing,
                    "success": True,
                    "error": None,
                }

            self.stage2_reasoning = ""
            self.stage2_answer = ""
            self.stage2_tool_usage = tool_usage
            self.stage2_success = False
            self.stage2_error = str(e)
            self.stage2_reply = None
            self._record_token_usage("stage2_agent", extra={"node_success": False, "error": str(e)})

            return {
                "answer": None,
                "reply": None,
                "tool_usage": tool_usage,
                "routing": self.stage2_routing,
                "success": False,
                "error": str(e),
            }

    def get_conversation(self):
        """
        負責執行 AgentNeuron 中的 get_conversation 流程，依照 AgentNeuron 的流程需求處理 get_conversation 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        if not self.active:
            return []

        contexts, formers = self.get_context()
        structured_formers = [former for former, _ in formers]
        runtime = getattr(self, "runtime", None)
        task_context = runtime.get_task_context() if runtime is not None and hasattr(runtime, "get_task_context") else None
        contexts.append(
            self.helper.build_stage1_message(
                self.question,
                structured_formers,
                contract=resolve_prompt_contract(task_context, question=self.question or ""),
                task_context=task_context,
            )
        )
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class NeuronEdge:
    """
    負責在 network.agent_neuron 中封裝 NeuronEdge，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        a1: 圖結構中的節點、邊或相關識別資料。
        a2: 圖結構中的節點、邊或相關識別資料。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def __init__(self, a1, a2):
        # a1 -> a2 的連線邊，weight 表示 a1 對 a2 的影響權重
        """
        負責執行 NeuronEdge 中的 __init__ 流程，初始化物件所需的設定、依賴與內部狀態，讓後續方法可以沿用同一份執行上下文。
        
        Args:
            a1: 圖結構中的節點、邊或相關識別資料。
            a2: 圖結構中的節點、邊或相關識別資料。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        """
        負責執行 NeuronEdge 中的 zero_weight 流程，依照 NeuronEdge 的流程需求處理 zero_weight 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 未標註。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.weight = 0
