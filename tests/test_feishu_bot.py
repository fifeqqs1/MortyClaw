import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from mortyclaw.core.integrations.feishu_bot import (
    FeishuBotSettings,
    build_feishu_channel,
    extract_agent_reply,
    feishu_thread_id,
    split_feishu_reply,
)


class FeishuBotSettingsTests(unittest.TestCase):
    def test_settings_use_existing_feishu_credentials(self):
        env = {
            "FEISHU_APP_ID": "cli_test",
            "FEISHU_APP_SECRET": "secret",
            "FEISHU_BOT_DM_ENABLED": "0",
            "FEISHU_BOT_GROUP_ENABLED": "1",
            "FEISHU_BOT_REQUIRE_MENTION": "true",
            "FEISHU_BOT_MAX_REPLY_CHARS": "2800",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = FeishuBotSettings.from_env()

        self.assertEqual(settings.app_id, "cli_test")
        self.assertFalse(settings.dm_enabled)
        self.assertTrue(settings.group_enabled)
        self.assertTrue(settings.require_mention)
        self.assertEqual(settings.max_reply_chars, 2800)

    def test_missing_credentials_fail_before_connection(self):
        with self.assertRaisesRegex(RuntimeError, "FEISHU_APP_ID"):
            FeishuBotSettings(app_id="", app_secret="").validate()

    def test_channel_policy_matches_bot_settings(self):
        channel = build_feishu_channel(
            FeishuBotSettings(
                app_id="cli_test",
                app_secret="secret",
                dm_enabled=True,
                group_enabled=False,
                require_mention=True,
            )
        )

        self.assertEqual(channel._config.policy.dm_policy, "open")
        self.assertEqual(channel._config.policy.group_policy, "disabled")
        self.assertTrue(channel._config.policy.require_mention)
        self.assertEqual(channel._config.security.mode, "strict")


class FeishuBotMessageTests(unittest.TestCase):
    def test_thread_id_is_stable_and_does_not_expose_chat_id(self):
        first = feishu_thread_id("oc_sensitive_chat")
        second = feishu_thread_id("oc_sensitive_chat")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("feishu-"))
        self.assertNotIn("oc_sensitive_chat", first)

    def test_long_reply_prefers_paragraph_boundaries(self):
        chunks = split_feishu_reply("第一段内容\n\n第二段很长的内容", max_chars=8)

        self.assertEqual("".join(chunks).replace("\n", ""), "第一段内容第二段很长的内容")
        self.assertTrue(all(len(chunk) <= 8 for chunk in chunks))

    def test_extracts_fast_reply(self):
        reply = extract_agent_reply(
            "fast_agent",
            {"messages": [AIMessage(content="你好，我是 MortyClaw。")]},
        )

        self.assertEqual(reply, "你好，我是 MortyClaw。")

    def test_ignores_slow_intermediate_step(self):
        reply = extract_agent_reply(
            "slow_agent",
            {
                "messages": [
                    AIMessage(
                        content="中间分析",
                        additional_kwargs={"mortyclaw_response_kind": "step_result"},
                    )
                ]
            },
        )

        self.assertEqual(reply, "")

    def test_extracts_approval_prompt_for_chat_confirmation(self):
        reply = extract_agent_reply(
            "approval_gate",
            {
                "messages": [
                    AIMessage(
                        content="这项飞书写入需要确认，请回复确认。",
                        additional_kwargs={"mortyclaw_response_kind": "final_answer"},
                    )
                ]
            },
        )

        self.assertIn("回复确认", reply)


if __name__ == "__main__":
    unittest.main()
