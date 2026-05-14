from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from memory.graph.insight_graph import InsightGraph
from memory.graph.interaction_graph import AgentMessage
from memory.graph.network_memory import NetworkMemory


class GraphMemoryLifecycleMockTest(unittest.TestCase):
    """
    負責在 test.test_graph_memory_lifecycle_mock 中封裝 GraphMemoryLifecycleMockTest，管理記憶圖、任務紀錄、檢索結果或跨任務經驗的狀態與操作。
    
    Args:
        無明確建構參數，可能透過 dataclass 欄位或預設值建立物件。
    
    Returns:
        類別本身不直接回傳值；建立實例後可透過其方法操作狀態與流程。
    
    限制或副作用:
        方法可能更新內部狀態、讀寫檔案、呼叫外部服務或產生日誌，需依使用情境確認。
    """
    def setUp(self) -> None:
        """
        負責執行 GraphMemoryLifecycleMockTest 中的 setUp 流程，更新記憶圖、互動狀態、節點邊關係或追蹤紀錄。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self.tmpdir = Path(tempfile.mkdtemp(prefix="graph_memory_mock_"))
        self.memory = NetworkMemory(
            namespace="mock_lifecycle",
            working_dir=self.tmpdir,
            auto_connect=False,
            use_qdrant=False,
        )

    def tearDown(self) -> None:
        """
        負責執行 GraphMemoryLifecycleMockTest 中的 tearDown 流程，依照 GraphMemoryLifecycleMockTest 的流程需求處理 tearDown 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seed_historical_success_and_insight(self) -> None:
        """
        負責執行 GraphMemoryLifecycleMockTest 中的 _seed_historical_success_and_insight 流程，依照 GraphMemoryLifecycleMockTest 的流程需求處理 _seed_historical_success_and_insight 對應的資料轉換、狀態操作或結果產生。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        success = self.memory.init_task_context(
            task_id="hist_success_probability",
            task_main="historical probability task",
            task_description="A random probability position advance task solved with explicit state transitions.",
            extra_fields={
                "source": "mock_seed",
                "score": 1.0,
                "final_answer": "2/3",
                "expected_answer": "2/3",
            },
        )
        success.start_state(
            state_id="hist_success_probability:stage1",
            state_type="agent_round",
            stage="stage1_round0",
        )
        self.memory.add_agent_node(
            AgentMessage(
                agent_name="solver_a",
                message="Define states S0, S1 and transition probabilities before computing the answer.",
            )
        )
        self.memory.move_memory_state(
            action="state_transition_model",
            observation="Correctly modeled probability transitions and answered 2/3.",
            reward=1.0,
        )
        saved = self.memory.save_task_context(label=True, feedback="Environment accepted the answer.")
        self.assertEqual(saved.label, "successful")

        self.memory.insight_graph.upsert_insight(
            {
                "insight_id": "insight_state_transition",
                "rule": "For random probability tasks, define states and transitions before choosing a numeric answer.",
                "task_type": "stochastic_process",
                "strategy": "Build a state transition model before solving.",
                "trigger_terms": ["random", "probability", "position", "advance"],
                "checklist": ["Define states", "Write transitions", "Verify requested format"],
                "failure_modes": ["missing_state_transition_model"],
                "score": 3.0,
                "positive_correlation_tasks": ["hist_success_probability"],
                "evidence_task_ids": ["hist_success_probability"],
            }
        )

    def test_complete_task_lifecycle(self) -> None:
        """
        負責執行 GraphMemoryLifecycleMockTest 中的 test_complete_task_lifecycle 流程，評估候選結果是否符合任務需求並回傳判定資訊。
        
        Args:
            無。
        
        Returns:
            執行結果；若函式標註回傳型別，預期型別為 None。
        
        限制或副作用:
            可能讀取或更新物件狀態、檔案、外部服務或日誌；請依呼叫場景確認副作用。
        """
        self._seed_historical_success_and_insight()

        # 1. Task start: establish current interaction graph.
        current = self.memory.init_task_context(
            task_id="mock_current_probability",
            task_main="mock current GAIA task",
            task_description="A random probability position advance task asks for the best odds.",
            extra_fields={"source": "mock_test"},
        )
        self.assertEqual(current.task_id, "mock_current_probability")
        self.assertIs(self.memory.current_task_context, current)

        # 2-5. Query Graph retrieval, success rerank, insight retrieval, prompt injection.
        retrieval_bundle = self.memory.retrieve_context(
            task_id=current.task_id,
            input_text=current.task_description or "",
            source="mock_stage1",
            limit=3,
            injection_target="stage1_round0",
        )
        retrieval = retrieval_bundle["retrieval"]
        self.assertTrue(retrieval["similar_task_records"])
        self.assertTrue(retrieval["similar_successes"])
        self.assertEqual(retrieval["similar_successes"][0]["task_id"], "hist_success_probability")
        self.assertTrue(retrieval_bundle["insights"])
        self.assertIn("Stage-1 Memory Strategy", retrieval_bundle["guidance"])
        self.assertIn("Strategy reminder", retrieval_bundle["guidance"])

        stage2_bundle = self.memory.retrieve_context(
            task_id=current.task_id,
            input_text=current.task_description or "",
            source="mock_stage2",
            limit=3,
            injection_target="stage2_top_k",
        )
        self.assertIn("Stage-2 Repair Memory", stage2_bundle["guidance"])
        self.assertIn("Repair strategy", stage2_bundle["guidance"])

        # 6-7. Record agent interaction graph and reply state chain.
        current.start_state(
            state_id="mock_current_probability:stage1_round0",
            state_type="agent_round",
            stage="stage1_round0",
        )
        node_a = self.memory.add_agent_node(
            AgentMessage(agent_name="agent_a", message="I guess the answer is 1/2 without modeling states.")
        )
        self.memory.move_memory_state(
            action="stage1_first_answer",
            observation="Surface numeric guess was produced.",
            reward=-1.0,
            extra_fields={"stage_result": "1/2"},
        )
        node_b = self.memory.add_agent_node(
            AgentMessage(agent_name="stage2_agent_b", message="Use state transition modeling and repair answer to 2/3."),
            upstream_agent_ids=[node_a],
        )
        self.memory.move_memory_state(
            action="stage2_repair",
            observation="State transitions were modeled and final answer changed to 2/3.",
            reward=1.0,
            extra_fields={"stage_result": "2/3", "upstream_node": node_b},
        )
        self.assertIn("stage1_first_answer", current.task_trajectory)
        self.assertIn("stage2_repair", current.task_trajectory)

        # 8. Complete workflow save_task_context(label=final_done, feedback=final_feedback).
        final_done = False
        final_feedback = "Expected 2/3, but submitted 1/2. Failure: missing state transition model."
        saved_current = self.memory.save_task_context(
            label=final_done,
            feedback=final_feedback,
            score=0.0,
            metadata={"final_answer": "1/2", "expected_answer": "2/3"},
        )
        self.assertEqual(saved_current.label, "failed")
        self.assertIn("Environment feedback", saved_current.task_description or "")

        # 9-11. Sparsify bad states, extract key steps, detect failure reason.
        state_rewards = [state.graph.get("reward") for state in saved_current.chain_of_states.chain_of_states]
        self.assertNotIn(-1.0, state_rewards)
        self.assertTrue(saved_current.extra_fields["key_steps"])
        self.assertIn("stage2_repair", saved_current.extra_fields["clean_traj"])
        self.assertEqual(saved_current.extra_fields["fail_reason"], "missing_state_transition_model")

        # 12-13. Query graph and local vector/task record stores are updated.
        self.assertIn("mock_current_probability", self.memory.query_task_graph._memory_tasks)
        self.assertIn("mock_current_probability", self.memory._task_records)
        self.assertTrue(self.memory.task_record_store_path.exists())
        self.assertIn("mock_current_probability", self.memory.task_vector_index.records)

        # 14. Failed task creates or updates insight candidates.
        self.assertTrue(
            any(
                "missing_state_transition_model" in record.failure_modes
                for record in self.memory.insight_graph._memory_insights.values()
            )
        )

        # 15. Backward feedback updates used insights.
        used = self.memory.retrieve_context(
            task_id=current.task_id,
            input_text=current.task_description or "",
            source="mock_feedback",
            limit=3,
            injection_target="stage2_top_k",
        )
        self.assertTrue(used["insights"])
        target_insight_id = used["insights"][0]["insight_id"]
        before = self.memory.insight_graph._memory_insights[target_insight_id].score
        self.memory.backward(True, task_id=current.task_id, stage2_changed_answer=True)
        after = self.memory.insight_graph._memory_insights[target_insight_id].score
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
