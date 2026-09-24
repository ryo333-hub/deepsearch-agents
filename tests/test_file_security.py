import builtins
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile

from app.agent import main_agent as main_agent_module
from app.api import server
from app.api.context import reset_session_context, set_session_context
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.tools.upload_file_read_tool import read_file_content
from app.utils.path_utils import (
    PATH_ACCESS_DENIED_MESSAGE,
    resolve_path,
    resolve_session_directory,
    validate_thread_id,
    validate_upload_filename,
)


def _create_directory_link_or_junction(link: Path, target: Path) -> None:
    """Create a temporary reparse-point escape using a Junction when needed."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as symlink_error:
        environment = os.environ.copy()
        environment["FILE_SECURITY_LINK_PATH"] = str(link)
        environment["FILE_SECURITY_TARGET_PATH"] = str(target)
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "New-Item -ItemType Junction "
                "-Path $env:FILE_SECURITY_LINK_PATH "
                "-Target $env:FILE_SECURITY_TARGET_PATH "
                "-ErrorAction Stop | Out-Null",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            raise unittest.SkipTest(
                "当前 Windows 环境无法创建测试符号链接或 Junction: "
                f"{symlink_error}; {completed.stderr.strip()}"
            )


def _remove_directory_link_or_junction(link: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        link.rmdir()


class TemporarySessionTestCase(unittest.TestCase):
    def setUp(self):
        project_tmp = Path.cwd() / ".tmp"
        project_tmp.mkdir(exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            dir=project_tmp,
            prefix="file-security-",
        )
        self.root = Path(self._temporary_directory.name)
        self.session_dir = self.root / "session-A"
        self.session_dir.mkdir()

    def tearDown(self):
        self._temporary_directory.cleanup()


class PathResolverSecurityTests(TemporarySessionTestCase):
    def test_allows_file_in_session_root(self):
        result = Path(resolve_path("input.txt", self.session_dir))
        self.assertEqual(result, (self.session_dir / "input.txt").resolve())

    def test_allows_file_in_session_subdirectory(self):
        result = Path(resolve_path("subdir/file.txt", self.session_dir))
        self.assertEqual(result, (self.session_dir / "subdir" / "file.txt").resolve())

    def test_rejects_forward_slash_traversal(self):
        for value in ("../secret.txt", "../../.env"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_path(value, self.session_dir)

    def test_rejects_backslash_traversal(self):
        for value in (r"..\secret.txt", r"..\..\secret.txt"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_path(value, self.session_dir)

    def test_rejects_mixed_separator_traversal(self):
        for value in (r"..\../secret.txt", r"../..\secret.txt", "sub/../file.txt"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_path(value, self.session_dir)

    def test_rejects_windows_drive_absolute_paths(self):
        for value in (r"C:\Windows\System32\test.txt", r"E:\other\secret.txt"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_path(value, self.session_dir)

    def test_rejects_forward_slash_drive_absolute_path(self):
        with self.assertRaises(ValueError):
            resolve_path("C:/Windows/test.txt", self.session_dir)

    def test_rejects_unc_device_posix_and_drive_relative_paths(self):
        rejected = (
            r"\\server\share\file.txt",
            r"\\?\C:\Windows\test.txt",
            "/absolute/path",
            "C:relative-path",
        )
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_path(value, self.session_dir)

    def test_rejects_windows_alternate_data_stream(self):
        with self.assertRaises(ValueError):
            resolve_path("filename.txt:secret", self.session_dir)

    def test_rejects_windows_reserved_and_ambiguous_components(self):
        for value in ("CON.txt", "subdir/NUL", "trailing-dot.", "trailing-space "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_path(value, self.session_dir)

    def test_allows_uuid_thread_id(self):
        value = "550e8400-e29b-41d4-a716-446655440000"
        self.assertEqual(validate_thread_id(value), value)

    def test_allows_project_thread_id_character_set(self):
        for value in ("abc", "ABC_123", "session-01"):
            with self.subTest(value=value):
                self.assertEqual(validate_thread_id(value), value)

    def test_rejects_dangerous_thread_ids(self):
        rejected = (
            "../abc",
            r"..\abc",
            r"C:\abc",
            "abc/def",
            r"abc\def",
            ".",
            "..",
            "",
            None,
            "a" * 129,
        )
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_thread_id(value)

    def test_resolve_session_directory_uses_validated_thread_id(self):
        sessions_root = self.root / "sessions"
        result = resolve_session_directory(sessions_root, "safe-thread_01")
        self.assertEqual(result, (sessions_root / "session_safe-thread_01").resolve())
        with self.assertRaises(ValueError):
            resolve_session_directory(sessions_root, "../unsafe")

    def test_allows_plain_upload_filename(self):
        for value in ("upload.pdf", "报告 2026.md", "data_01.xlsx"):
            with self.subTest(value=value):
                self.assertEqual(validate_upload_filename(value), value)

    def test_rejects_dangerous_upload_filenames(self):
        rejected = (
            "../../.env",
            r"..\..\secret.txt",
            "folder/file.pdf",
            r"folder\file.pdf",
            r"C:\x.pdf",
            r"\\server\share\x.pdf",
            "file.txt:stream",
            "CON.txt",
            "",
            None,
        )
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_upload_filename(value)

    def test_sessions_cannot_access_each_other(self):
        session_b = self.root / "session-B"
        session_b.mkdir()
        (self.session_dir / "a.txt").write_text("A", encoding="utf-8")
        (session_b / "b.txt").write_text("B", encoding="utf-8")

        self.assertEqual(
            Path(resolve_path("a.txt", self.session_dir)).read_text(encoding="utf-8"),
            "A",
        )
        self.assertEqual(
            Path(resolve_path("b.txt", session_b)).read_text(encoding="utf-8"),
            "B",
        )
        with self.assertRaises(ValueError):
            resolve_path("../session-B/b.txt", self.session_dir)
        with self.assertRaises(ValueError):
            resolve_path("../session-A/a.txt", session_b)

    def test_rejects_symlink_or_junction_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret", encoding="utf-8")
        link = self.session_dir / "outside-link"
        _create_directory_link_or_junction(link, outside)

        try:
            with self.assertRaises(ValueError):
                resolve_path("outside-link/secret.txt", self.session_dir)
        finally:
            _remove_directory_link_or_junction(link)


class FileToolGuardTests(TemporarySessionTestCase):
    def setUp(self):
        super().setUp()
        self._session_token = set_session_context(str(self.session_dir))

    def tearDown(self):
        reset_session_context(self._session_token)
        super().tearDown()

    def test_read_rejects_before_file_io(self):
        with (
            patch.object(Path, "exists") as exists_mock,
            patch.object(Path, "read_text") as read_text_mock,
            patch.object(builtins, "open") as open_mock,
        ):
            result = read_file_content.invoke({"filename": "../secret.txt"})

        self.assertEqual(result, PATH_ACCESS_DENIED_MESSAGE)
        exists_mock.assert_not_called()
        read_text_mock.assert_not_called()
        open_mock.assert_not_called()

    def test_markdown_rejects_before_directory_or_write_io(self):
        with (
            patch.object(Path, "exists") as exists_mock,
            patch.object(Path, "mkdir") as mkdir_mock,
            patch.object(Path, "write_text") as write_text_mock,
        ):
            result = generate_markdown.invoke(
                {"content": "blocked", "filename": "../secret"}
            )

        self.assertEqual(result, PATH_ACCESS_DENIED_MESSAGE)
        exists_mock.assert_not_called()
        mkdir_mock.assert_not_called()
        write_text_mock.assert_not_called()

    def test_pdf_input_rejects_before_exists_or_conversion(self):
        with (
            patch.object(Path, "exists") as exists_mock,
            patch("app.tools.pdf_tools.convert_md_to_pdf_via_word") as converter_mock,
        ):
            result = convert_md_to_pdf.invoke({"md_filename": "../secret.md"})

        self.assertEqual(result, PATH_ACCESS_DENIED_MESSAGE)
        exists_mock.assert_not_called()
        converter_mock.assert_not_called()

    def test_pdf_output_rejects_before_input_io_or_conversion(self):
        (self.session_dir / "safe.md").write_text("# safe", encoding="utf-8")
        with (
            patch.object(Path, "exists") as exists_mock,
            patch("app.tools.pdf_tools.convert_md_to_pdf_via_word") as converter_mock,
        ):
            result = convert_md_to_pdf.invoke(
                {"md_filename": "safe.md", "pdf_filename": "../outside.pdf"}
            )

        self.assertEqual(result, PATH_ACCESS_DENIED_MESSAGE)
        exists_mock.assert_not_called()
        converter_mock.assert_not_called()

    def test_ads_is_rejected_before_suffix_completion_and_io(self):
        with (
            patch.object(Path, "write_text") as write_text_mock,
            patch("app.tools.pdf_tools.convert_md_to_pdf_via_word") as converter_mock,
        ):
            markdown_result = generate_markdown.invoke(
                {"content": "blocked", "filename": "report.txt:stream"}
            )
            pdf_result = convert_md_to_pdf.invoke(
                {"md_filename": "report.txt:stream"}
            )

        self.assertEqual(markdown_result, PATH_ACCESS_DENIED_MESSAGE)
        self.assertEqual(pdf_result, PATH_ACCESS_DENIED_MESSAGE)
        write_text_mock.assert_not_called()
        converter_mock.assert_not_called()

    def test_file_tool_failures_do_not_expose_host_session_path(self):
        readable = self.session_dir / "readable.txt"
        source = self.session_dir / "source.md"
        readable.write_text("content", encoding="utf-8")
        source.write_text("# source", encoding="utf-8")
        absolute_error = PermissionError(str(self.session_dir / "secret"))

        with patch.object(Path, "read_text", side_effect=absolute_error):
            read_result = read_file_content.invoke({"filename": "readable.txt"})
        with patch.object(Path, "write_text", side_effect=absolute_error):
            markdown_result = generate_markdown.invoke(
                {"content": "content", "filename": "report.md"}
            )
        with patch(
            "app.tools.pdf_tools.convert_md_to_pdf_via_word",
            side_effect=absolute_error,
        ):
            pdf_result = convert_md_to_pdf.invoke({"md_filename": "source.md"})

        for result in (read_result, markdown_result, pdf_result):
            with self.subTest(result=result):
                self.assertNotIn(str(self.session_dir), result)

    def test_real_markdown_write_and_read_regression(self):
        write_result = generate_markdown.invoke(
            {"content": "# 安全回归\n内容正常。", "filename": "report", "path": "nested"}
        )
        output_file = self.session_dir / "nested" / "report.md"

        self.assertIn("已成功生成", write_result)
        self.assertIn("nested/report.md", write_result)
        self.assertNotIn(str(self.session_dir), write_result)
        self.assertTrue(output_file.is_file())
        self.assertEqual(
            read_file_content.invoke({"filename": "nested/report.md"}),
            "# 安全回归\n内容正常。",
        )

    def test_real_session_local_markdown_to_pdf_regression(self):
        source = self.session_dir / "pdf-source.md"
        source.write_text("# PDF 安全回归\n\n仅位于临时会话目录。", encoding="utf-8")

        result = convert_md_to_pdf.invoke(
            {"md_filename": "pdf-source.md", "pdf_filename": "nested/result.pdf"}
        )

        self.assertEqual(result, "成功转换: nested/result.pdf")
        self.assertNotIn(str(self.session_dir), result)
        self.assertTrue((self.session_dir / "nested" / "result.pdf").is_file())

    def test_markdown_accepts_session_relative_filename_with_subdirectory(self):
        result = generate_markdown.invoke(
            {"content": "# 逻辑路径", "filename": "reports/direct.md"}
        )

        self.assertEqual(
            result,
            "Markdown文件 'reports/direct.md' 已成功生成并保存。",
        )
        self.assertTrue((self.session_dir / "reports" / "direct.md").is_file())


class FilePathSemanticConsistencyTests(unittest.TestCase):
    def test_main_prompt_uses_only_session_relative_file_semantics(self):
        prompt = main_agent_module.main_agent_content["system_prompt"]

        self.assertIn("当前 session 内相对路径", prompt)
        self.assertIn("read_file_content", prompt)
        self.assertIn("generate_markdown", prompt)
        self.assertIn("convert_md_to_pdf", prompt)
        self.assertIn("不要要求用户提供服务器绝对路径", prompt)
        self.assertIn("不得尝试读取 .env", prompt)
        self.assertNotIn("文件生成助手", prompt)
        self.assertNotIn("指定的绝对路径作为工作目录", prompt)

    def test_runtime_file_instruction_contains_only_logical_upload_names(self):
        instruction = main_agent_module._build_session_file_instruction(
            ["input.pdf", "数据.xlsx"]
        )

        self.assertIn("当前 session 内相对路径", instruction)
        self.assertIn("- input.pdf", instruction)
        self.assertIn("- 数据.xlsx", instruction)
        self.assertNotIn(str(main_agent_module.project_root_path), instruction)
        self.assertNotIn(r"E:\Code\deepsearch-agents", instruction)
        self.assertNotIn(r"C:\Users", instruction)

    def test_read_file_schema_requires_session_relative_path(self):
        description = read_file_content.args_schema.model_json_schema()["properties"][
            "filename"
        ]["description"]

        self.assertIn("当前 session 内相对路径", description)
        self.assertIn("禁止绝对路径", description)
        self.assertIn("uploads/data.pdf", description)

    def test_markdown_schema_requires_relative_filename_and_path(self):
        properties = generate_markdown.args_schema.model_json_schema()["properties"]

        self.assertIn("当前 session 内", properties["filename"]["description"])
        self.assertIn("reports/report.md", properties["filename"]["description"])
        self.assertIn("当前 session 内相对子目录", properties["path"]["description"])
        self.assertNotIn("文件保存的绝对路径", properties["path"]["description"])

    def test_pdf_schema_requires_relative_input_and_output(self):
        properties = convert_md_to_pdf.args_schema.model_json_schema()["properties"]

        self.assertIn("当前 session 内", properties["md_filename"]["description"])
        self.assertIn("reports/report.md", properties["md_filename"]["description"])
        self.assertIn("当前 session 内", properties["pdf_filename"]["description"])
        self.assertIn("reports/report.pdf", properties["pdf_filename"]["description"])


class ApiBoundaryTests(TemporarySessionTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_upload_rejects_dangerous_name_before_io(self):
        upload = UploadFile(filename="../../.env", file=io.BytesIO(b"blocked"))
        with (
            patch.object(server, "updated_dir", self.root / "updated"),
            patch.object(Path, "mkdir") as mkdir_mock,
            patch.object(Path, "open") as open_mock,
            patch.object(server.shutil, "copyfileobj") as copy_mock,
        ):
            with self.assertRaises(HTTPException) as raised:
                await server.upload_files([upload], "valid-thread")

        self.assertEqual(raised.exception.status_code, 400)
        mkdir_mock.assert_not_called()
        open_mock.assert_not_called()
        copy_mock.assert_not_called()

    async def test_plain_upload_stays_in_session_upload_directory(self):
        updated_root = self.root / "updated"
        upload = UploadFile(filename="upload.pdf", file=io.BytesIO(b"pdf-data"))
        with patch.object(server, "updated_dir", updated_root):
            result = await server.upload_files([upload], "safe-thread")

        uploaded = updated_root / "session_safe-thread" / "upload.pdf"
        self.assertEqual(result, {"status": "uploaded", "files": ["upload.pdf"]})
        self.assertEqual(uploaded.read_bytes(), b"pdf-data")

    async def test_download_and_list_are_scoped_to_requested_session(self):
        output_root = self.root / "output"
        session_a = resolve_session_directory(output_root, "A")
        session_b = resolve_session_directory(output_root, "B")
        session_a.mkdir(parents=True)
        session_b.mkdir(parents=True)
        (session_a / "a.txt").write_text("A", encoding="utf-8")
        (session_b / "b.txt").write_text("B", encoding="utf-8")

        with patch.object(server, "output_dir", output_root):
            listed = await server.list_files("A")
            downloaded = await server.download_file("A", "a.txt")
            cross_session = await server.download_file("A", "../session_B/b.txt")

        self.assertEqual([item["path"] for item in listed["files"]], ["a.txt"])
        self.assertEqual(Path(downloaded.path), session_a / "a.txt")
        self.assertEqual(cross_session, {"error": PATH_ACCESS_DENIED_MESSAGE})

    async def test_file_list_does_not_follow_junction_outside_session(self):
        output_root = self.root / "output"
        session = resolve_session_directory(output_root, "safe-thread")
        outside = self.root / "outside"
        session.mkdir(parents=True)
        outside.mkdir()
        (session / "inside.txt").write_text("inside", encoding="utf-8")
        (outside / "secret.txt").write_text("outside", encoding="utf-8")
        link = session / "outside-link"
        _create_directory_link_or_junction(link, outside)

        try:
            with patch.object(server, "output_dir", output_root):
                result = await server.list_files("safe-thread")
        finally:
            _remove_directory_link_or_junction(link)

        self.assertEqual([item["path"] for item in result["files"]], ["inside.txt"])

    async def test_download_traversal_rejects_before_file_check(self):
        with (
            patch.object(server, "output_dir", self.root / "output"),
            patch.object(Path, "is_file") as is_file_mock,
        ):
            result = await server.download_file("safe-thread", "../secret.txt")

        self.assertEqual(result, {"error": PATH_ACCESS_DENIED_MESSAGE})
        is_file_mock.assert_not_called()

    async def test_list_traversal_rejects_before_directory_check(self):
        with (
            patch.object(server, "output_dir", self.root / "output"),
            patch.object(Path, "is_dir") as is_dir_mock,
        ):
            result = await server.list_files("safe-thread", "../other-session")

        self.assertEqual(result, {"error": PATH_ACCESS_DENIED_MESSAGE})
        is_dir_mock.assert_not_called()

    async def test_invalid_task_thread_id_rejects_before_agent_task_creation(self):
        with patch.object(server.asyncio, "create_task") as create_task_mock:
            with self.assertRaises(HTTPException) as raised:
                await server.run_task(server.TaskRequest(query="test", thread_id="../bad"))

        self.assertEqual(raised.exception.status_code, 400)
        create_task_mock.assert_not_called()

    async def test_invalid_websocket_thread_id_rejects_before_registration(self):
        websocket = unittest.mock.MagicMock()
        websocket.close = AsyncMock()
        with patch.object(server.manager, "connect", new_callable=AsyncMock) as connect_mock:
            await server.websocket_endpoint(websocket, "../bad")

        websocket.close.assert_awaited_once()
        connect_mock.assert_not_awaited()

    async def test_direct_agent_call_rejects_bad_session_before_mkdir(self):
        with patch.object(Path, "mkdir") as mkdir_mock:
            with self.assertRaises(ValueError):
                await main_agent_module.run_deep_agent("test", "../bad")

        mkdir_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
