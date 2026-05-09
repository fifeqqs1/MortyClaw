import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mortyclaw.core.tools.builtins import BUILTIN_TOOLS
from mortyclaw.core.tools.summarize_tool import _find_summarize_binary, summarize_content


def completed(stdout="extracted text", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class SummarizeContentToolTests(unittest.TestCase):
    def test_prefers_local_summarize_binary(self):
        with patch("mortyclaw.core.tools.summarize_tool._find_summarize_binary", return_value="/usr/bin/summarize"), patch(
            "mortyclaw.core.tools.summarize_tool.subprocess.run",
            return_value=completed("local extracted content"),
        ) as mock_run:
            result = summarize_content.invoke({
                "source": "https://example.com/report",
                "length": "short",
                "force_summary": True,
            })

        self.assertIn("local extracted content", result)
        self.assertIn("摘要长度偏好：short", result)
        self.assertIn("强制摘要：是", result)
        command = mock_run.call_args.args[0]
        self.assertEqual(command, [
            "/usr/bin/summarize",
            "https://example.com/report",
            "--extract-only",
        ])
        self.assertNotIn("--length", command)
        self.assertNotIn("--force-summary", command)
        self.assertNotIn("shell", mock_run.call_args.kwargs)

    def test_falls_back_to_npx_when_summarize_is_missing(self):
        def which(command):
            return {"npx": "/usr/bin/npx"}.get(command)

        with patch("mortyclaw.core.tools.summarize_tool._find_summarize_binary", return_value=None), patch(
            "mortyclaw.core.tools.summarize_tool.shutil.which",
            side_effect=which,
        ), patch(
            "mortyclaw.core.tools.summarize_tool.subprocess.run",
            return_value=completed("npx extracted content"),
        ) as mock_run:
            result = summarize_content.invoke({
                "source": "https://example.com/article",
                "length": "medium",
            })

        self.assertIn("npx extracted content", result)
        self.assertEqual(mock_run.call_args.args[0], [
            "/usr/bin/npx",
            "-y",
            "@steipete/summarize",
            "https://example.com/article",
            "--extract-only",
        ])

    def test_pdf_sources_use_local_text_extraction_without_summarize_cli(self):
        with patch(
            "mortyclaw.core.tools.summarize_tool._extract_pdf_text",
            return_value="PDF extracted text",
        ) as mock_extract, patch(
            "mortyclaw.core.tools.summarize_tool.subprocess.run",
        ) as mock_run:
            result = summarize_content.invoke({
                "source": "https://example.com/article.pdf",
                "length": "long",
            })

        self.assertIn("PDF extracted text", result)
        self.assertIn("摘要长度偏好：long", result)
        mock_extract.assert_called_once_with("https://example.com/article.pdf")
        mock_run.assert_not_called()

    def test_returns_install_hint_when_no_runner_exists(self):
        with patch("mortyclaw.core.tools.summarize_tool._find_summarize_binary", return_value=None), patch(
            "mortyclaw.core.tools.summarize_tool.shutil.which",
            return_value=None,
        ):
            result = summarize_content.invoke({"source": "https://example.com"})

        self.assertIn("未找到 summarize 或 npx", result)
        self.assertIn("npm i -g @steipete/summarize", result)

    def test_finds_summarize_inside_current_python_env_when_path_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = os.path.join(temp_dir, "Scripts" if os.name == "nt" else "bin")
            os.makedirs(bin_dir, exist_ok=True)
            executable_name = "summarize.cmd" if os.name == "nt" else "summarize"
            executable = os.path.join(bin_dir, executable_name)
            with open(executable, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\n")
            os.chmod(executable, 0o755)

            with patch("mortyclaw.core.tools.summarize_tool.shutil.which", return_value=None), patch(
                "mortyclaw.core.tools.summarize_tool.sys.prefix",
                temp_dir,
            ), patch(
                "mortyclaw.core.tools.summarize_tool.sys.executable",
                os.path.join(bin_dir, "python"),
            ):
                result = _find_summarize_binary()

        self.assertEqual(result, executable)

    def test_rejects_code_and_project_config_files(self):
        blocked_sources = [
            "https://example.com/main.py",
            "agent.py",
            "package.json",
            "config.yaml",
            ".env",
            "pnpm-lock.yaml",
        ]

        with patch("mortyclaw.core.tools.summarize_tool.subprocess.run") as mock_run:
            for source in blocked_sources:
                with self.subTest(source=source):
                    result = summarize_content.invoke({"source": source})
                    self.assertIn("不处理代码或项目配置文件", result)

        mock_run.assert_not_called()

    def test_resolves_relative_non_code_file_inside_office(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "notes.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# Notes\ncontent")

            with patch("mortyclaw.core.tools.summarize_tool.OFFICE_DIR", temp_dir), patch(
                "mortyclaw.core.tools.summarize_tool._find_summarize_binary",
                return_value="/usr/bin/summarize",
            ), patch(
                "mortyclaw.core.tools.summarize_tool.subprocess.run",
                return_value=completed("notes summary"),
            ) as mock_run:
                result = summarize_content.invoke({"source": "notes.md"})

        self.assertIn("# Notes\ncontent", result)
        mock_run.assert_not_called()

    def test_rejects_local_binary_media_without_text_extraction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "image.png")
            with open(file_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")

            result = summarize_content.invoke({"source": file_path})

        self.assertIn("本地图片/音频/视频暂不支持", result)

    def test_rejects_invalid_length_before_running_subprocess(self):
        with patch("mortyclaw.core.tools.summarize_tool.subprocess.run") as mock_run:
            result = summarize_content.invoke({
                "source": "https://example.com",
                "length": "tiny",
            })

        self.assertIn("length 只能是", result)
        mock_run.assert_not_called()

    def test_rejects_relative_path_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "mortyclaw.core.tools.summarize_tool.OFFICE_DIR",
            temp_dir,
        ), patch("mortyclaw.core.tools.summarize_tool.subprocess.run") as mock_run:
            result = summarize_content.invoke({"source": "../notes.md"})

        self.assertIn("越权拦截", result)
        mock_run.assert_not_called()

    def test_reports_subprocess_timeout(self):
        with patch("mortyclaw.core.tools.summarize_tool._find_summarize_binary", return_value="/usr/bin/summarize"), patch(
            "mortyclaw.core.tools.summarize_tool.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["summarize"], timeout=300),
        ):
            result = summarize_content.invoke({"source": "https://example.com"})

        self.assertIn("执行超时", result)

    def test_reports_nonzero_exit(self):
        with patch("mortyclaw.core.tools.summarize_tool._find_summarize_binary", return_value="/usr/bin/summarize"), patch(
            "mortyclaw.core.tools.summarize_tool.subprocess.run",
            return_value=completed(stdout="", stderr="bad input", returncode=2),
        ):
            result = summarize_content.invoke({"source": "https://example.com"})

        self.assertIn("执行失败", result)
        self.assertIn("bad input", result)

    def test_registered_in_builtin_tools(self):
        self.assertIn(summarize_content, BUILTIN_TOOLS)


if __name__ == "__main__":
    unittest.main()
