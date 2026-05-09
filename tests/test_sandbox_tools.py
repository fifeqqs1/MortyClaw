import unittest
from unittest.mock import patch, mock_open
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mortyclaw.core.tools.sandbox_tools import (
    list_office_files,
    read_office_file,
    write_office_file,
    execute_office_shell,
    _get_safe_path
)
from mortyclaw.core.config import OFFICE_DIR


class TestSandboxTools(unittest.TestCase):

    def test_get_safe_path_normal(self):
        """测试正常路径连接"""
        # _get_safe_path 是内部函数，不受装饰器影响，可以直接调用
        # 注意：OFFICE_DIR 是模块级常量，patch 需要在导入前或使用正确的路径
        original_office_dir = OFFICE_DIR
        try:
            # 使用实际 OFFICE_DIR 测试
            result = _get_safe_path('subdir/file.txt')
            expected = os.path.abspath(os.path.join(OFFICE_DIR, 'subdir/file.txt'))
            self.assertEqual(result, expected)
        finally:
            pass

    def test_get_safe_path_traversal_attempt(self):
        """测试路径遍历攻击"""
        with self.assertRaises(PermissionError):
            _get_safe_path('../../forbidden/file.txt')

    @patch('mortyclaw.core.tools.sandbox_tools.os.path.exists', return_value=True)
    @patch('mortyclaw.core.tools.sandbox_tools.os.listdir', return_value=['file1.txt', 'subdir'])
    @patch('mortyclaw.core.tools.sandbox_tools.os.path.isdir', side_effect=lambda x: x.endswith('subdir'))
    def test_list_office_files(self, mock_isdir, mock_listdir, mock_exists):
        """测试列出办公文件功能"""
        # 工具需要通过 .invoke() 调用
        result = list_office_files.invoke({"sub_dir": ""})

        # 验证函数调用了正确的路径检查
        mock_exists.assert_called_once()
        mock_listdir.assert_called_once()

        # 检查返回结果包含预期元素
        self.assertIn("📄 file1.txt", result)
        self.assertIn("📁 subdir", result)

    @patch('mortyclaw.core.tools.sandbox_tools.os.path.exists', return_value=False)
    def test_list_office_files_nonexistent_dir(self, mock_exists):
        """测试列出不存在目录的文件"""
        result = list_office_files.invoke({"sub_dir": "nonexistent"})
        self.assertIn("目录不存在", result)

    @patch('mortyclaw.core.tools.sandbox_tools.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data="file content")
    def test_read_office_file_success(self, mock_file, mock_exists):
        """测试成功读取办公文件"""
        result = read_office_file.invoke({"filepath": "test.txt"})
        self.assertEqual(result, "file content")
        mock_file.assert_called_once()

    @patch('mortyclaw.core.tools.sandbox_tools.os.path.exists', return_value=False)
    def test_read_office_file_nonexistent(self, mock_exists):
        """测试读取不存在的办公文件"""
        result = read_office_file.invoke({"filepath": "nonexistent.txt"})
        self.assertIn("文件不存在", result)

    def test_list_office_files_absolute_dir_readonly(self):
        """测试可只读列出 office 外的绝对路径目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            open(os.path.join(temp_dir, "a.txt"), "w", encoding="utf-8").close()
            os.makedirs(os.path.join(temp_dir, "nested"), exist_ok=True)

            result = list_office_files.invoke({"sub_dir": temp_dir})

            self.assertIn("📄 a.txt", result)
            self.assertIn("📁 nested", result)

    def test_list_office_files_relative_traversal_readonly(self):
        """测试可只读列出相对路径解析到 office 外的目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "visible"), exist_ok=True)
            relative_path = os.path.relpath(temp_dir, OFFICE_DIR)

            result = list_office_files.invoke({"sub_dir": relative_path})

            self.assertIn("📁 visible", result)

    def test_read_office_file_absolute_path_readonly(self):
        """测试可只读读取 office 外的绝对路径文件"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as temp_file:
            temp_file.write("outside office content")
            temp_path = temp_file.name

        try:
            result = read_office_file.invoke({"filepath": temp_path})
            self.assertEqual(result, "outside office content")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_read_office_file_relative_traversal_readonly(self):
        """测试可只读读取相对路径解析到 office 外的文件"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as temp_file:
            temp_file.write("outside office relative content")
            temp_path = temp_file.name

        try:
            relative_path = os.path.relpath(temp_path, OFFICE_DIR)
            result = read_office_file.invoke({"filepath": relative_path})
            self.assertEqual(result, "outside office relative content")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_write_office_file_success(self, mock_makedirs, mock_file):
        """测试成功写入办公文件"""
        result = write_office_file.invoke({"filepath": "test.txt", "content": "test content", "mode": "w"})
        self.assertIn("成功以 覆盖/新建 模式写入文件", result)
        mock_file.assert_called_once()
        mock_makedirs.assert_called_once()

    def test_write_office_file_invalid_mode(self):
        """测试写入办公文件 - 无效模式"""
        result = write_office_file.invoke({"filepath": "test.txt", "content": "test content", "mode": "x"})
        self.assertIn("❌ 错误：mode 参数必须是", result)

    def test_write_office_file_absolute_path_forbidden(self):
        """测试禁止写入 office 外的绝对路径"""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            result = write_office_file.invoke({"filepath": temp_path, "content": "blocked", "mode": "w"})
            self.assertIn("越权拦截", result)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch('mortyclaw.core.tools.sandbox_tools.subprocess.run')
    def test_execute_office_shell_safe_command(self, mock_subprocess):
        """测试执行安全的 shell 命令"""
        # Mock subprocess 结果
        mock_result = mock_subprocess.return_value
        mock_result.returncode = 0
        mock_result.stdout = "command output"
        mock_result.stderr = ""

        result = execute_office_shell.invoke({"command": "ls"})
        # 输出格式包含前缀空格和中文冒号 - 使用更宽松的匹配
        self.assertIn("ls", result)
        self.assertIn("command output", result)

    def test_execute_office_shell_dangerous_commands(self):
        """测试执行危险命令会被拦截"""
        dangerous_commands = [
            "cd ../",
            "cat /etc/passwd",
            "ls ~",
            "dir \\",
            "type C:\\windows\\system32\\config\\sam"
        ]

        for cmd in dangerous_commands:
            with self.subTest(cmd=cmd):
                result = execute_office_shell.invoke({"command": cmd})
                self.assertIn("❌ 权限拒绝", result)


if __name__ == '__main__':
    unittest.main()
