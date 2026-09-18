import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.tools import tool

from mortyclaw.core.harness.gateway import GatewayTool, MortyClawGateway, _PUBLIC_NAMES, _schema_for
from mortyclaw.core.harness.settings import HarnessSettings
from mortyclaw.core.harness.storage import HarnessStore
from mortyclaw.core.harness.runtime import _redact
from mortyclaw.core.storage.store import RuntimeStore
from mortyclaw.core.tools.meta import ToolMeta, attach_tool_meta


@tool("read_sample")
def read_sample(value: str) -> str:
    """Read a sample value."""
    return f"read:{value}"


@tool("write_sample")
def write_sample(value: str) -> str:
    """Write a sample value."""
    return f"write:{value}"


class HarnessStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = HarnessStore(RuntimeStore(str(Path(self.temp.name) / "runtime.sqlite3")))

    def tearDown(self):
        self.temp.cleanup()

    def test_session_mapping_is_stable_and_reset_increments_generation(self):
        first = self.store.session_id("thread-a")
        self.assertEqual(first, self.store.session_id("thread-a"))
        second = self.store.reset("thread-a")
        self.assertNotEqual(first, second)
        self.assertNotIn("thread-a", first)

    def test_context_token_is_hashed_expires_and_checks_workspace(self):
        token = self.store.issue_context_token(
            thread_id="thread-a", turn_id="turn-a", source="cli", workspace=self.temp.name,
        )
        context = self.store.validate_context_token(token, workspace=self.temp.name)
        self.assertEqual(context["turn_id"], "turn-a")
        self.assertIsNone(self.store.validate_context_token(token, workspace=str(Path(self.temp.name) / "other")))
        with self.store.store._connect() as conn:
            raw = conn.execute("SELECT token_hash FROM harness_context_tokens").fetchone()[0]
        self.assertNotEqual(raw, token)

    def test_approval_deduplicates_same_tool_and_arguments(self):
        context = {"thread_id": "t", "turn_id": "u", "source": "cli", "workspace": self.temp.name}
        first = self.store.stage_approval(context=context, tool_name="write", arguments={"x": 1}, risk_reason="high")
        second = self.store.stage_approval(context=context, tool_name="write", arguments={"x": 1}, risk_reason="high")
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(first[2])
        self.assertFalse(second[2])
        self.assertEqual(len(self.store.pending_batch("t", "u")["operations"]), 1)

    def test_session_lease_excludes_other_process_owner(self):
        session_id = self.store.session_id("thread-lease")
        self.assertTrue(self.store.acquire_lease(session_id, "owner-a", 60))
        self.assertFalse(self.store.acquire_lease(session_id, "owner-b", 60))
        self.store.release_lease(session_id, "owner-a")
        self.assertTrue(self.store.acquire_lease(session_id, "owner-b", 60))


class GatewayPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = HarnessStore(RuntimeStore(str(Path(self.temp.name) / "runtime.sqlite3")))
        self.gateway = MortyClawGateway(store=self.store)
        read_meta = ToolMeta.build(name="read_sample", capabilities={"external_read"}, risk_level="low")
        write_meta = ToolMeta.build(name="write_sample", capabilities={"external_write"}, risk_level="high", requires_approval=True)
        attach_tool_meta(read_sample, read_meta)
        attach_tool_meta(write_sample, write_meta)
        self.gateway._tools = {
            "read_sample": GatewayTool("read_sample", read_sample, read_meta),
            "write_sample": GatewayTool("write_sample", write_sample, write_meta),
        }
        self.token = self.store.issue_context_token(
            thread_id="thread-a", turn_id="turn-a", source="cli", workspace=self.temp.name,
        )

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_read_executes_but_write_waits_for_approval(self):
        read_result = await self.gateway.call("read_sample", {"context_token": self.token, "value": "x"})
        self.assertEqual(read_result["result"], "read:x")
        write_result = await self.gateway.call("write_sample", {"context_token": self.token, "value": "x"})
        self.assertEqual(write_result["status"], "approval_required")
        batch = self.store.get_batch(write_result["batch_id"])
        self.assertEqual(batch["operations"][0]["status"], "pending")
        execution = await self.gateway.execute_approved(write_result["batch_id"])
        self.assertEqual(execution["status"], "completed")
        self.assertEqual(execution["results"][0]["result"], "write:x")

    async def test_missing_context_token_is_rejected(self):
        result = await self.gateway.call("read_sample", {"value": "x"})
        self.assertEqual(result["error"], "invalid_context_token")

    async def test_configured_secret_is_never_staged(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-sensitive-config-value"}, clear=False):
            result = await self.gateway.call(
                "write_sample",
                {"context_token": self.token, "value": "test-sensitive-config-value"},
            )
        self.assertEqual(result["error"], "sensitive_argument_rejected")
        self.assertIsNone(self.store.pending_batch("thread-a"))

    def test_tool_schema_requires_context_token(self):
        schema = _schema_for(read_sample)
        self.assertIn("context_token", schema["required"])
        self.assertIn("context_token", schema["properties"])


class HarnessSettingsTests(unittest.TestCase):
    def test_builtin_gateway_names_use_governed_namespaces(self):
        allowed = ("memory_", "task_", "project_")
        self.assertTrue(all(name.startswith(allowed) for name in _PUBLIC_NAMES.values()))

    def test_runtime_redaction_removes_keys_and_bearer_tokens(self):
        value = _redact(
            "api_key=sk-visible-secret Bearer gateway-secret",
            extra_secrets=("gateway-secret",),
        )
        self.assertNotIn("sk-visible-secret", value)
        self.assertNotIn("gateway-secret", value)

    def test_official_old_base_migrates_key_without_forwarding_base(self):
        env = {
            "DEFAULT_PROVIDER": "other", "DEFAULT_MODEL": "deepseek-flash",
            "OPENAI_API_KEY": "secret", "OPENAI_API_BASE": "https://api.deepseek.com",
        }
        from unittest.mock import patch
        with patch.dict(os.environ, env, clear=True):
            settings = HarnessSettings.from_env()
        self.assertEqual(settings.api_key, "secret")
        self.assertIsNone(settings.base_url)
        self.assertEqual(settings.model, "deepseek-v4-flash")
        self.assertNotIn("secret", json.dumps(settings.public_status()))


if __name__ == "__main__":
    unittest.main()
