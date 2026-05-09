import unittest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestConfig(unittest.TestCase):

    def test_config_import(self):
        """测试配置模块导入"""
        from mortyclaw.core.config import (
            WORKSPACE_DIR,
            MEMORY_DIR,
            MEMORY_DB_PATH,
            PERSONAS_DIR,
            SCRIPTS_DIR,
            OFFICE_DIR,
            SKILLS_DIR,
            DB_PATH,
            TASKS_FILE,
        )

        # 验证配置项存在
        self.assertIsInstance(WORKSPACE_DIR, str)
        self.assertIsInstance(MEMORY_DIR, str)
        self.assertIsInstance(MEMORY_DB_PATH, str)
        self.assertIsInstance(PERSONAS_DIR, str)
        self.assertIsInstance(SCRIPTS_DIR, str)
        self.assertIsInstance(OFFICE_DIR, str)
        self.assertIsInstance(SKILLS_DIR, str)
        self.assertIsInstance(DB_PATH, str)
        self.assertIsInstance(TASKS_FILE, str)

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "openai-key",
        "ALIYUN_API_KEY": "aliyun-key",
        "OPENAI_API_BASE": "https://openai.example/v1",
        "ALIYUN_BASE_URL": "https://dashscope.example/v1",
    }, clear=True)
    def test_provider_env_resolution_prefers_provider_specific_aliases(self):
        from mortyclaw.core.provider import (
            resolve_compatible_provider_api_key,
            resolve_compatible_provider_base_url,
        )

        self.assertEqual(resolve_compatible_provider_api_key("openai"), "openai-key")
        self.assertEqual(resolve_compatible_provider_api_key("aliyun"), "aliyun-key")
        self.assertEqual(resolve_compatible_provider_base_url("openai"), "https://openai.example/v1")
        self.assertEqual(resolve_compatible_provider_base_url("aliyun"), "https://dashscope.example/v1")

    def test_provider_env_resolution_keeps_openai_key_as_fallback(self):
        from mortyclaw.core import provider as provider_module

        with patch.dict(provider_module.os.environ, {"OPENAI_API_KEY": "fallback-key"}, clear=True):
            self.assertEqual(
                provider_module.resolve_compatible_provider_api_key("aliyun"),
                "fallback-key",
            )


class TestSkillLoader(unittest.TestCase):

    def test_skill_loader_import(self):
        """测试技能加载器模块导入"""
        try:
            from mortyclaw.core.skill_loader import load_dynamic_skills
            # 确保函数存在
            self.assertTrue(callable(load_dynamic_skills))
        except ImportError as e:
            # 如果导入失败，可能是因为依赖问题，但仍需确认模块结构
            self.fail(f"无法导入技能加载器: {e}")

    @patch('os.path.exists', return_value=False)
    @patch('os.listdir', side_effect=FileNotFoundError())
    def test_load_dynamic_skills_no_directory(self, mock_listdir, mock_exists):
        """测试技能加载器 - 不存在的目录"""
        from mortyclaw.core.skill_loader import load_dynamic_skills

        skills = load_dynamic_skills()
        self.assertEqual(skills, [])

    @patch('os.path.exists', return_value=True)
    @patch('os.listdir', return_value=[])
    def test_load_dynamic_skills_empty_directory(self, mock_listdir, mock_exists):
        """测试技能加载器 - 空目录"""
        from mortyclaw.core.skill_loader import load_dynamic_skills

        skills = load_dynamic_skills()
        self.assertEqual(skills, [])


if __name__ == '__main__':
    unittest.main()
