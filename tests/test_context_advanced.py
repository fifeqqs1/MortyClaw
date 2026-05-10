import unittest
import json
import os
import sys
import tempfile
from unittest.mock import patch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mortyclaw.core.context import trim_context_messages, AgentState
from mortyclaw.core.context.window import (
    _FallbackEncoder,
    _estimate_message_tokens,
    _estimate_messages_tokens,
    _resolve_token_encoder,
    classify_context_pressure,
    compact_context_messages_deterministic,
    estimate_text_tokens,
)


class TestContextTrimming(unittest.TestCase):

    def test_trim_with_system_message_keep_all(self):
        """测试保留所有消息的情况（不超过阈值）"""
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="用户消息1"),
            AIMessage(content="AI消息1"),
            HumanMessage(content="用户消息2"),
            AIMessage(content="AI消息2")
        ]

        kept, discarded = trim_context_messages(messages, trigger_turns=10, keep_turns=10)

        # 由于回合数(2) < 触发阈值(10)，不应裁剪
        self.assertEqual(len(kept), 5)  # 包含系统消息
        self.assertEqual(len(discarded), 0)

    def test_trim_with_system_message_discard_some(self):
        """测试裁剪部分消息的情况"""
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="用户消息1"),
            AIMessage(content="AI消息1"),
            HumanMessage(content="用户消息2"),
            AIMessage(content="AI消息2"),
            HumanMessage(content="用户消息3"),
            AIMessage(content="AI消息3"),
            HumanMessage(content="用户消息4"),
            AIMessage(content="AI消息4"),
            HumanMessage(content="用户消息5"),
            AIMessage(content="AI消息5")
        ]

        kept, discarded = trim_context_messages(messages, trigger_turns=3, keep_turns=2)

        # 由于回合数(5) > 触发阈值(3)，应裁剪
        # 保留最后2个回合 + 系统消息 = 5条消息
        self.assertEqual(len(kept), 5)
        self.assertEqual(len(discarded), 6)  # 前3个回合的消息

        # 验证系统消息在保留的消息中
        self.assertIsInstance(kept[0], SystemMessage)

        # 验证保留的是最后2个回合
        self.assertIsInstance(kept[1], HumanMessage)
        self.assertIsInstance(kept[2], AIMessage)
        self.assertIsInstance(kept[3], HumanMessage)
        self.assertIsInstance(kept[4], AIMessage)

    def test_trim_without_system_message(self):
        """测试没有系统消息时的裁剪"""
        messages = [
            HumanMessage(content="用户消息1"),
            AIMessage(content="AI消息1"),
            HumanMessage(content="用户消息2"),
            AIMessage(content="AI消息2"),
            HumanMessage(content="用户消息3"),
            AIMessage(content="AI消息3")
        ]

        kept, discarded = trim_context_messages(messages, trigger_turns=2, keep_turns=1)

        # 回合数(3) > 触发阈值(2)，保留最后1个回合
        self.assertEqual(len(kept), 2)  # 最后一个回合(Human+AI)
        self.assertEqual(len(discarded), 4)  # 前2个回合

    def test_trim_only_system_message(self):
        """测试只有系统消息的情况"""
        messages = [
            SystemMessage(content="系统消息")
        ]

        kept, discarded = trim_context_messages(messages, trigger_turns=1, keep_turns=1)

        self.assertEqual(len(kept), 1)
        self.assertEqual(len(discarded), 0)
        self.assertIsInstance(kept[0], SystemMessage)

    def test_trim_empty_messages(self):
        """测试空消息列表"""
        messages = []

        kept, discarded = trim_context_messages(messages, trigger_turns=1, keep_turns=1)

        self.assertEqual(len(kept), 0)
        self.assertEqual(len(discarded), 0)

    def test_trim_with_tool_messages(self):
        """测试包含工具消息的裁剪"""
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="用户消息1"),
            AIMessage(content="AI消息1"),
            ToolMessage(content="工具结果1", tool_call_id="1"),
            HumanMessage(content="用户消息2"),
            AIMessage(content="AI消息2"),
            ToolMessage(content="工具结果2", tool_call_id="2"),
            HumanMessage(content="用户消息3"),
            AIMessage(content="AI消息3")
        ]

        kept, discarded = trim_context_messages(messages, trigger_turns=2, keep_turns=1)

        # 3个回合(每回合可能包含多个消息) > 阈值2，保留最后1个回合
        # 最后一个回合：HumanMessage + AIMessage
        # 所以前面两回合的所有消息都被丢弃
        self.assertEqual(len(discarded), 6)  # 前两个回合加上系统消息
        self.assertEqual(len(kept), 3)  # 最后一个回合的Human + AI

    def test_turn_calculation_logic(self):
        """测试回合计算逻辑"""
        messages = [
            HumanMessage(content="用户消息1"),
            AIMessage(content="AI消息1a"),
            ToolMessage(content="工具结果1a", tool_call_id="1a"),
            AIMessage(content="AI消息1b"),
            ToolMessage(content="工具结果1b", tool_call_id="1b"),
            HumanMessage(content="用户消息2"),
            AIMessage(content="AI消息2"),
            HumanMessage(content="用户消息3"),
            AIMessage(content="AI消息3a"),
            ToolMessage(content="工具结果3a", tool_call_id="3a"),
            AIMessage(content="AI消息3b")
        ]

        # 测试回合是如何计算的
        # 回合1: Human1, AI1a, Tool1a, AI1b, Tool1b
        # 回合2: Human2, AI2
        # 回合3: Human3, AI3a, Tool3a, AI3b
        # 总共3个回合

        kept, discarded = trim_context_messages(messages, trigger_turns=2, keep_turns=1)

        # 3回合 > 阈值2，保留最后1回合
        self.assertEqual(len(kept), 4)  # Human3, AI3a, Tool3a, AI3b
        self.assertEqual(len(discarded), 7)  # 前两个回合的所有消息

    def test_trim_large_single_turn_by_message_budget(self):
        """测试单个 slow-path 大回合也会按 message 数裁剪，而不是无限膨胀"""
        messages = [SystemMessage(content="系统消息"), HumanMessage(content="继续分析项目并推进当前步骤")]
        for index in range(1, 11):
            messages.append(AIMessage(content=f"AI消息{index}"))
            messages.append(ToolMessage(content=f"工具结果{index}", tool_call_id=str(index)))

        kept, discarded = trim_context_messages(
            messages,
            trigger_turns=40,
            keep_turns=10,
            trigger_messages=12,
            keep_messages=6,
        )

        self.assertEqual(len(kept), 7)  # system + 6 non-system messages
        self.assertEqual(len(discarded), 15)
        self.assertIsInstance(kept[0], SystemMessage)
        self.assertIsInstance(kept[1], HumanMessage)
        self.assertEqual(kept[1].content, "继续分析项目并推进当前步骤")
        self.assertEqual(kept[-1].content, "工具结果10")

    def test_trim_large_single_turn_preserves_latest_user_message_outside_tail(self):
        """测试即使最新 HumanMessage 已经不在尾部，也会强制保留它"""
        messages = [HumanMessage(content="请继续当前任务")]
        for index in range(1, 9):
            messages.append(AIMessage(content=f"AI消息{index}"))

        kept, discarded = trim_context_messages(
            messages,
            trigger_turns=40,
            keep_turns=10,
            trigger_messages=6,
            keep_messages=4,
        )

        self.assertEqual(len(kept), 4)
        self.assertEqual(kept[0].content, "请继续当前任务")
        self.assertEqual([message.content for message in kept[1:]], ["AI消息6", "AI消息7", "AI消息8"])
        self.assertEqual(len(discarded), 5)

    def test_token_budget_does_not_trim_below_threshold(self):
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="你好"),
            AIMessage(content="你好，有什么可以帮你？"),
        ]

        kept, discarded = trim_context_messages(
            messages,
            trigger_tokens=200,
            keep_tokens=160,
            reserve_tokens=20,
            model_name="gpt-4o-mini",
        )

        self.assertEqual(len(kept), 3)
        self.assertEqual(discarded, [])

    def test_token_budget_preserves_tail_while_trimming_early_context(self):
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="用户消息1"),
            AIMessage(content="A" * 80),
            HumanMessage(content="用户消息2"),
            AIMessage(content="B" * 80),
            HumanMessage(content="用户消息3"),
            AIMessage(content="C" * 80),
        ]
        encoder = _resolve_token_encoder("gpt-4o-mini")
        total_tokens = _estimate_messages_tokens(messages, encoder=encoder)
        retained_tail_tokens = _estimate_messages_tokens(messages[-3:], encoder=encoder)

        kept, discarded = trim_context_messages(
            messages,
            trigger_tokens=total_tokens - 1,
            keep_tokens=retained_tail_tokens + 20,
            reserve_tokens=20,
            model_name="gpt-4o-mini",
        )

        self.assertIsInstance(kept[0], SystemMessage)
        self.assertEqual(kept[-2].content, "用户消息3")
        self.assertGreater(len(discarded), 0)
        self.assertTrue(any(message.content == "A" * 80 for message in discarded))
        self.assertTrue(any(message.content == "B" * 80 for message in kept + discarded))

    def test_token_budget_preserves_latest_human_message_even_if_not_in_tail(self):
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="请继续当前任务"),
            AIMessage(content="A" * 120),
            AIMessage(content="B" * 120),
            AIMessage(content="C" * 120),
        ]

        kept, discarded = trim_context_messages(
            messages,
            trigger_tokens=130,
            keep_tokens=105,
            reserve_tokens=20,
            model_name="gpt-4o-mini",
        )

        self.assertEqual(kept[0].content, "系统消息")
        self.assertEqual(kept[1].content, "请继续当前任务")
        self.assertGreaterEqual(len(discarded), 1)

    def test_tool_message_has_extra_token_overhead(self):
        encoder = _resolve_token_encoder("gpt-4o-mini")
        ai_cost = _estimate_message_tokens(AIMessage(content="结果"), encoder=encoder)
        tool_cost = _estimate_message_tokens(ToolMessage(content="结果", tool_call_id="1"), encoder=encoder)
        self.assertEqual(tool_cost - ai_cost, 12)

    def test_resolve_token_encoder_prefers_model_specific_encoding(self):
        with patch("mortyclaw.core.context.window.tiktoken.encoding_for_model") as mock_encoding_for_model:
            mock_encoder = _FallbackEncoder()
            mock_encoding_for_model.return_value = mock_encoder
            encoder = _resolve_token_encoder.__wrapped__("gpt-4o-mini")

        self.assertIs(encoder, mock_encoder)
        mock_encoding_for_model.assert_called_once_with("gpt-4o-mini")

    def test_resolve_token_encoder_falls_back_to_generic_encoding(self):
        with (
            patch("mortyclaw.core.context.window.tiktoken.encoding_for_model", side_effect=KeyError("unknown")),
            patch("mortyclaw.core.context.window.tiktoken.get_encoding") as mock_get_encoding,
        ):
            mock_encoder = _FallbackEncoder()
            mock_get_encoding.return_value = mock_encoder
            encoder = _resolve_token_encoder.__wrapped__("unknown-model")

        self.assertIs(encoder, mock_encoder)
        mock_get_encoding.assert_called_with("o200k_base")

    def test_reserve_tokens_causes_earlier_token_trim(self):
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="用户消息1"),
            AIMessage(content="A" * 60),
            HumanMessage(content="用户消息2"),
            AIMessage(content="B" * 60),
        ]

        kept_without_reserve, discarded_without_reserve = trim_context_messages(
            messages,
            trigger_tokens=120,
            keep_tokens=120,
            reserve_tokens=0,
            model_name="gpt-4o-mini",
        )
        kept_with_reserve, discarded_with_reserve = trim_context_messages(
            messages,
            trigger_tokens=120,
            keep_tokens=120,
            reserve_tokens=50,
            model_name="gpt-4o-mini",
        )

        self.assertEqual(discarded_without_reserve, [])
        self.assertGreater(len(discarded_with_reserve), 0)
        self.assertLess(len(kept_with_reserve), len(kept_without_reserve))

    def test_context_pressure_uses_budget_ratio_and_extra_context(self):
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="用户消息"),
            AIMessage(content="A" * 60),
        ]
        model_name = "gpt-4o-mini"
        message_tokens = _estimate_messages_tokens(
            messages,
            encoder=_resolve_token_encoder(model_name),
        )
        extra_text = "B" * 120
        reserve_tokens = 40
        extra_tokens = estimate_text_tokens(extra_text, model_name=model_name)
        budget_tokens = max(1, int((message_tokens + extra_tokens + reserve_tokens) / 0.6))

        pressure = classify_context_pressure(
            messages,
            model_name=model_name,
            budget_tokens=budget_tokens,
            reserve_tokens=reserve_tokens,
            extra_texts=[extra_text],
            layer2_trigger_ratio=0.5,
            layer3_trigger_ratio=0.7,
        )

        self.assertEqual(pressure["level"], "medium")
        self.assertEqual(pressure["message_tokens"], message_tokens)
        self.assertEqual(pressure["extra_tokens"], extra_tokens)
        self.assertEqual(pressure["reserve_tokens"], reserve_tokens)
        self.assertAlmostEqual(float(pressure["usage_ratio"]), 0.6, delta=0.06)

    def test_context_pressure_reaches_high_ratio(self):
        messages = [
            HumanMessage(content="用户消息"),
            AIMessage(content="A" * 120),
        ]
        model_name = "gpt-4o-mini"
        message_tokens = _estimate_messages_tokens(
            messages,
            encoder=_resolve_token_encoder(model_name),
        )
        budget_tokens = max(1, int(message_tokens / 0.75))

        pressure = classify_context_pressure(
            messages,
            model_name=model_name,
            budget_tokens=budget_tokens,
            layer2_trigger_ratio=0.5,
            layer3_trigger_ratio=0.7,
        )

        self.assertEqual(pressure["level"], "high")
        self.assertGreaterEqual(float(pressure["usage_ratio"]), 0.7)

    def test_token_budget_counts_artifact_messages_without_recompressing_them(self):
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="继续任务"),
            ToolMessage(
                content=(
                    "<persisted-output>\n"
                    "完整输出已保存到：/tmp/artifact.txt\n"
                    "预览（前 10 字符）：\npreview text\n"
                    "</persisted-output>"
                ),
                tool_call_id="call-artifact",
                additional_kwargs={
                    "mortyclaw_artifact": {
                        "artifact_persisted": True,
                        "artifact_path": "/tmp/artifact.txt",
                        "artifact_size": 12000,
                        "preview_chars": 10,
                    }
                },
            ),
            AIMessage(content="最后回复"),
        ]

        kept, discarded = trim_context_messages(
            messages,
            trigger_tokens=1,
            keep_tokens=500,
            reserve_tokens=0,
            model_name="gpt-4o-mini",
        )

        self.assertEqual(discarded, [])
        self.assertEqual(kept[2].content, messages[2].content)
        self.assertEqual(trim_context_messages._last_stats["artifact_messages_seen"], 1)
        self.assertEqual(trim_context_messages._last_stats["tool_results_pruned"], 0)

    def test_token_budget_repairs_missing_tool_result_with_stub(self):
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="旧请求"),
            AIMessage(tool_calls=[{"id": "call-old", "name": "search_project_code", "args": {"query": "old"}}], content=""),
            AIMessage(content="中间分析" + ("A" * 600)),
            ToolMessage(content="旧工具结果" + ("B" * 600), tool_call_id="call-old"),
            HumanMessage(content="最新请求"),
            AIMessage(content="最新答复"),
        ]

        kept, discarded = trim_context_messages(
            messages,
            trigger_tokens=200,
            keep_tokens=120,
            reserve_tokens=0,
            model_name="gpt-4o-mini",
        )

        self.assertGreater(len(discarded), 0)
        repaired_results = [
            message for message in kept
            if isinstance(message, ToolMessage) and "上下文压缩说明" in str(message.content)
        ]
        self.assertEqual(len(repaired_results), 1)
        self.assertEqual(repaired_results[0].tool_call_id, "call-old")
        self.assertGreaterEqual(trim_context_messages._last_stats["tool_pairs_repaired"], 1)

    def test_deterministic_compaction_replaces_old_tool_group_with_artifact_stub(self):
        with tempfile.TemporaryDirectory() as workspace:
            with patch("mortyclaw.core.runtime.tool_results.RUNTIME_ARTIFACTS_DIR", workspace):
                from mortyclaw.core.runtime.tool_results import restore_context_artifact

                messages = [
                    HumanMessage(content="分析项目", id="user-1"),
                    AIMessage(
                        content="",
                        id="ai-tool",
                        tool_calls=[{
                            "id": "call-read",
                            "name": "read_project_file",
                            "args": {"filepath": "src/app.py", "start_line": 1, "end_line": 80},
                        }],
                    ),
                    ToolMessage(
                        content="def app():\n" + ("print('x')\n" * 1000),
                        tool_call_id="call-read",
                        name="read_project_file",
                        id="tool-read",
                    ),
                    HumanMessage(content="继续", id="user-2"),
                    AIMessage(content="最后结论", id="ai-final"),
                ]

                result = compact_context_messages_deterministic(
                    messages,
                    thread_id="ctx-thread",
                    turn_id="ctx-turn",
                    model_name="gpt-4o-mini",
                    protect_tail_groups=1,
                )

                self.assertEqual(result.stats["stubbed_count"], 1)
                self.assertEqual(result.stats["artifact_count"], 1)
                self.assertTrue(any("restore_context_artifact" in str(message.content) for message in result.stub_messages))
                self.assertFalse(any(isinstance(message, ToolMessage) and message.tool_call_id == "call-read" for message in result.kept_messages))
                stub_text = str(result.stub_messages[0].content)
                self.assertIn("content_hash", stub_text)
                self.assertIn("original_tokens", stub_text)
                self.assertIn("src/app.py", stub_text)
                ref_id = result.stubbed_groups[0]["tools"][0]["artifact_ref"]
                restored = restore_context_artifact(ref_id, preview_chars=200)
                self.assertIn('"ok": true', restored)
                self.assertIn("def app", restored)

                missing = restore_context_artifact("not-registered")
                self.assertIn('"ok": false', missing)

    def test_deterministic_compaction_discards_duplicate_search_group(self):
        messages = [
            HumanMessage(content="查配置", id="user-1"),
            AIMessage(content="", id="ai-search-1", tool_calls=[{"id": "call-search-1", "name": "search_project_code", "args": {"query": "config"}}]),
            ToolMessage(content="old result", tool_call_id="call-search-1", name="search_project_code", id="tool-search-1"),
            AIMessage(content="", id="ai-search-2", tool_calls=[{"id": "call-search-2", "name": "search_project_code", "args": {"query": "config"}}]),
            ToolMessage(content="new result", tool_call_id="call-search-2", name="search_project_code", id="tool-search-2"),
            HumanMessage(content="继续", id="user-2"),
        ]

        result = compact_context_messages_deterministic(
            messages,
            thread_id="ctx-thread",
            turn_id="ctx-turn",
            model_name="gpt-4o-mini",
            protect_tail_groups=1,
        )

        removed_ids = {getattr(message, "id", "") for message in result.safely_discarded}
        self.assertIn("ai-search-1", removed_ids)
        self.assertIn("tool-search-1", removed_ids)

    def test_deterministic_compaction_keeps_tool_protocol_legal(self):
        messages = [
            HumanMessage(content="查入口", id="user-1"),
            AIMessage(
                content="",
                id="ai-read",
                tool_calls=[{"id": "call-read", "name": "read_project_file", "args": {"filepath": "src/app.py"}}],
            ),
            ToolMessage(
                content="file content\n" + ("x\n" * 400),
                tool_call_id="call-read",
                name="read_project_file",
                id="tool-read",
            ),
            HumanMessage(content="继续", id="user-2"),
        ]

        result = compact_context_messages_deterministic(
            messages,
            thread_id="ctx-thread",
            turn_id="ctx-turn",
            model_name="gpt-4o-mini",
            persist_artifacts=False,
            protect_tail_groups=1,
        )

        tool_call_ids = {
            tool_call.get("id")
            for message in result.kept_messages
            if isinstance(message, AIMessage)
            for tool_call in (getattr(message, "tool_calls", []) or [])
        }
        result_ids = {
            getattr(message, "tool_call_id", "")
            for message in result.kept_messages
            if isinstance(message, ToolMessage)
        }
        self.assertTrue(tool_call_ids.issubset(result_ids))
        self.assertFalse(any(getattr(message, "id", "") == "ai-read" for message in result.kept_messages))
        self.assertTrue(any("context_tool_stub" in str(getattr(message, "additional_kwargs", {})) for message in result.kept_messages))

    def test_restore_context_artifact_rejects_manifest_path_outside_root(self):
        with tempfile.TemporaryDirectory() as workspace, tempfile.NamedTemporaryFile(delete=False) as outside:
            outside.write(b"secret")
            outside_path = outside.name
            try:
                with patch("mortyclaw.core.runtime.tool_results.RUNTIME_ARTIFACTS_DIR", workspace):
                    from mortyclaw.core.runtime.tool_results import restore_context_artifact

                    manifest_dir = os.path.join(workspace, "context", "thread", "turn")
                    os.makedirs(manifest_dir, exist_ok=True)
                    with open(os.path.join(manifest_dir, "manifest.jsonl"), "w", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "ref_id": "ctx_outside",
                            "artifact_path": outside_path,
                            "content_hash": "",
                        }, ensure_ascii=False) + "\n")

                    restored = restore_context_artifact("ctx_outside")
                    self.assertIn('"ok": false', restored)
                    self.assertIn("outside runtime artifact root", restored)
            finally:
                try:
                    os.remove(outside_path)
                except OSError:
                    pass


class TestAgentState(unittest.TestCase):

    def test_agent_state_initialization(self):
        """测试AgentState的初始化"""
        initial_state = AgentState(
            messages=[],
            summary=""
        )

        self.assertEqual(initial_state["messages"], [])
        self.assertEqual(initial_state["summary"], "")

    def test_agent_state_with_messages(self):
        """测试带消息的AgentState"""
        messages = [
            HumanMessage(content="用户消息"),
            AIMessage(content="AI消息")
        ]

        state = AgentState(
            messages=messages,
            summary="测试摘要"
        )

        self.assertEqual(len(state["messages"]), 2)
        self.assertEqual(state["summary"], "测试摘要")


if __name__ == '__main__':
    unittest.main()
