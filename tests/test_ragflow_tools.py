"""RAGFlow 0.25.2 离线契约测试：保留真实 SDK，替换 HTTP 传输。"""

import importlib
import json
import unittest
from urllib.parse import urlsplit
from unittest.mock import patch

import requests
from ragflow_sdk.modules.session import Session

from app.tools import ragflow_tools


class FakeResponse(requests.Response):
    def __init__(self, data, status=200, invalid_json=False):
        super().__init__()
        self.status_code = status
        self.url = "https://offline.invalid/api/v1/test"
        self.encoding = "utf-8"
        self._content = b"not-json" if invalid_json else json.dumps(data).encode("utf-8")
        self._content_consumed = True
        self.status_checks = 0

    def raise_for_status(self):
        self.status_checks += 1
        return super().raise_for_status()


class RagflowToolsTests(unittest.TestCase):
    def setUp(self):
        self.chat = {"id": "chat-1", "name": "助手", "dataset_ids": ["dataset-1"]}
        self.chats = [self.chat]
        self.answer = "测试答案"
        self.calls = []
        self.responses = []
        self.ask_error = None
        self.delete_error = None
        self.ask_status = 200
        self.list_status = 200
        self.ask_code = 0
        self.invalid_json = False
        self.invalid_shape = False
        self.empty_session_id = False
        self.reference = {"chunks": [{"document_name": "sample.pdf", "content": "引用"}]}
        self.config = patch(
            "app.tools.ragflow_tools._load_ragflow_env",
            return_value=("offline-secret-key", "https://offline.invalid"),
        ).start()
        self.addCleanup(patch.stopall)
        patch.object(ragflow_tools.monitor, "report_tool").start()
        self.http_mock = patch("requests.request", side_effect=self.http).start()
        # 即便项目意外绕过 fake HTTP，也不能建立真实连接。
        self.socket_mocks = [
            patch(target, side_effect=AssertionError("Real network forbidden")).start()
            for target in (
                "socket.socket.connect", "socket.socket.connect_ex",
                "socket.getaddrinfo", "requests.sessions.Session.send",
            )
        ]
        self.addCleanup(self.assert_no_network)

    def assert_no_network(self):
        for mock in self.socket_mocks:
            mock.assert_not_called()

    def response(self, data, status=200, invalid_json=False):
        response = FakeResponse(data, status, invalid_json)
        self.responses.append(response)
        return response

    def http(self, method, url, **kwargs):
        self.assertEqual(urlsplit(url).hostname, "offline.invalid")
        path = urlsplit(url).path
        self.calls.append((method, path, kwargs))
        if method == "GET":
            if self.list_status != 200:
                return self.response({"code": self.list_status, "message": "offline-secret-key"}, self.list_status)
            return self.response({"code": 0, "data": {"chats": self.chats}})
        if method == "POST" and path.endswith("/sessions"):
            return self.response({"code": 0, "data": {
                "id": None if self.empty_session_id else "session-1",
                "chat_id": "chat-1",
            }})
        if method == "POST" and path.endswith("/completions"):
            if self.ask_error:
                raise self.ask_error
            if self.invalid_shape:
                return self.response({"code": 0, "data": None})
            return self.response(
                {"code": self.ask_code, "message": "offline-secret-key",
                 "data": {"answer": self.answer, "reference": self.reference}},
                self.ask_status, self.invalid_json,
            )
        if method == "DELETE":
            if self.delete_error:
                raise self.delete_error
            return self.response({"code": 0})
        self.fail("Unexpected HTTP operation")

    def ask(self):
        return ragflow_tools.create_ask_delete.invoke({"chat_name": "助手", "question": "测试问题"})

    def expected_answer(self, answer="测试答案"):
        return f"答案：\n{answer}\n\n来源：\n1. 文档：sample.pdf"

    def assert_deleted(self):
        deletes = [c for c in self.calls if c[0] == "DELETE"]
        self.assertEqual(len(deletes), 1)
        self.assertEqual(deletes[0][2]["json"], {"ids": ["session-1"], "delete_all": False})

    def test_missing_url_never_requests(self):
        for url in (None, "", "  "):
            with self.subTest(url=url):
                self.config.return_value = ("offline-secret-key", url)
                for result in (ragflow_tools.get_assistant_list.invoke({}), self.ask()):
                    self.assertIn("缺少 RAGFLOW_API_URL", result)
        self.http_mock.assert_not_called()

    def test_missing_key_never_requests(self):
        for key in (None, "", "  "):
            with self.subTest(key=key):
                self.config.return_value = (key, "https://offline.invalid")
                self.assertIn("缺少 RAGFLOW_API_KEY", self.ask())
                self.assertIn("缺少 RAGFLOW_API_KEY", ragflow_tools.get_assistant_list.invoke({}))
        self.http_mock.assert_not_called()

    def test_invalid_urls_never_request_or_echo_input(self):
        for url in ("file:///secret", "offline.invalid", "https://", "https://host:bad",
                    "https://user:offline-secret-key@host", "https://bad host"):
            with self.subTest(url=url):
                self.config.return_value = ("offline-secret-key", url)
                result = self.ask()
                self.assertIn("合法 HTTP/HTTPS", result)
                self.assertNotIn("offline-secret-key", result)
        self.http_mock.assert_not_called()

    def test_minimal_sdk_fields_include_ids_and_missing_description(self):
        result = ragflow_tools.get_assistant_list.invoke({})
        for part in ("助手", "Chat ID：chat-1", "无描述", "关联 Dataset ID：dataset-1"):
            self.assertIn(part, result)
        self.assertNotIn("知识库名称", result)

    def test_optional_description_is_preserved(self):
        self.chat["description"] = "真实描述"
        self.assertIn("真实描述", ragflow_tools.get_assistant_list.invoke({}))

    def test_no_dataset_is_reported_as_unprovided(self):
        self.chat.pop("dataset_ids")
        self.assertIn("关联 Dataset ID：未提供", ragflow_tools.get_assistant_list.invoke({}))

    def test_empty_assistant_list(self):
        self.chats = []
        self.assertEqual(ragflow_tools.get_assistant_list.invoke({}), "没有任何可用助手")

    def test_missing_assistant_never_creates_session(self):
        self.chats = []
        self.assertEqual(self.ask(), "RAGFlow 未找到指定助手。")
        self.assertEqual(len(self.calls), 1)

    def test_nonexact_name_match_is_rejected(self):
        self.chat["name"] = "另一个助手"
        self.assertEqual(self.ask(), "RAGFlow 未找到指定助手。")
        self.assertEqual(len(self.calls), 1)

    def test_duplicate_assistants_are_rejected(self):
        self.chats.append({**self.chat, "id": "chat-2"})
        self.assertIn("存在多个同名助手", self.ask())
        self.assertEqual(len(self.calls), 1)

    def test_normal_sdk_lifecycle_and_payload(self):
        self.assertEqual(self.ask(), self.expected_answer())
        self.assertEqual([(m, p) for m, p, _ in self.calls], [
            ("GET", "/api/v1/chats"),
            ("POST", "/api/v1/chats/chat-1/sessions"),
            ("POST", "/api/v1/chats/chat-1/completions"),
            ("DELETE", "/api/v1/chats/chat-1/sessions"),
        ])
        self.assertEqual(self.calls[0][2]["params"]["name"], "助手")
        self.assertEqual(self.calls[1][2]["json"], {"name": "temp_session_ask"})
        self.assertEqual(self.calls[2][2]["json"], {
            "question": "测试问题", "stream": False, "session_id": "session-1",
        })
        self.assertFalse(self.calls[2][2]["stream"])
        self.assert_deleted()
        self.assertTrue(all(r.status_checks == 1 for r in self.responses))

    def test_public_sdk_ask_is_used(self):
        original = Session.ask
        with patch.object(Session, "ask", autospec=True, side_effect=original) as ask:
            self.assertEqual(self.ask(), self.expected_answer())
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(ask.call_args.kwargs, {"question": "测试问题", "stream": False})

    def test_ask_exception_still_deletes_without_leaking_exception(self):
        self.ask_error = RuntimeError("offline-secret-key")
        result = self.ask()
        self.assertIn("RAGFlow 提问失败", result)
        self.assertNotIn("offline-secret-key", result)
        self.assert_deleted()

    def test_json_parsing_failure_still_deletes(self):
        self.invalid_json = True
        self.assertIn("无效 JSON", self.ask())
        self.assert_deleted()

    def test_sdk_response_shape_failure_still_deletes(self):
        self.invalid_shape = True
        self.assertIn("响应处理失败", self.ask())
        self.assert_deleted()

    def test_cleanup_failure_preserves_answer(self):
        self.delete_error = requests.ConnectionError("offline-secret-key")
        result = self.ask()
        self.assertTrue(result.startswith(self.expected_answer() + "\n"))
        self.assertIn("临时会话清理失败", result)
        self.assertNotIn("提问失败", result)
        self.assertNotIn("offline-secret-key", result)
        self.assert_deleted()

    def test_ask_and_cleanup_errors_are_both_reported(self):
        self.ask_error = requests.Timeout("offline-secret-key")
        self.delete_error = requests.ConnectionError("offline-secret-key")
        result = self.ask()
        self.assertIn("提问失败：请求超时", result)
        self.assertIn("临时会话清理失败：无法连接服务", result)
        self.assert_deleted()

    def test_timeout_still_deletes(self):
        self.ask_error = requests.Timeout("offline-secret-key")
        self.assertIn("请求超时", self.ask())
        self.assert_deleted()

    def test_connection_error_still_deletes(self):
        self.ask_error = requests.ConnectionError("offline-secret-key")
        self.assertIn("无法连接服务", self.ask())
        self.assert_deleted()

    def check_http_failure(self, status):
        self.ask_status = status
        self.invalid_json = True  # 必须先判 HTTP 状态，不能先解析 JSON。
        result = self.ask()
        self.assertIn(f"HTTP {status}", result)
        self.assertNotIn("无效 JSON", result)
        self.assertNotIn("offline-secret-key", result)
        self.assert_deleted()

    def test_http_401(self):
        self.check_http_failure(401)

    def test_http_403(self):
        self.check_http_failure(403)

    def test_http_429(self):
        self.check_http_failure(429)

    def test_http_500(self):
        self.check_http_failure(500)

    def test_list_http_error_is_not_empty_list(self):
        self.list_status = 403
        self.assertIn("HTTP 403", ragflow_tools.get_assistant_list.invoke({}))
        self.assertEqual(len(self.calls), 1)

    def test_business_error_is_not_empty_answer(self):
        self.ask_code = 102
        result = self.ask()
        self.assertIn("业务错误（code=102）", result)
        self.assertNotIn("offline-secret-key", result)
        self.assert_deleted()

    def test_empty_answer_still_deletes(self):
        for answer in ("", "  "):
            with self.subTest(answer=answer):
                self.calls.clear()
                self.answer = answer
                self.assertEqual(self.ask(), "RAGFlow 未返回有效答案。")
                self.assert_deleted()

    def test_complete_answer_keeps_repeated_characters(self):
        self.answer = "aa"
        self.assertEqual(self.ask(), self.expected_answer("aa"))
        self.assertFalse(self.calls[2][2]["json"]["stream"])

    def test_answer_with_single_document_reference(self):
        self.reference = {"chunks": [{
            "document_id": "doc-1", "document_name": "制度.pdf",
            "dataset_id": "dataset-1", "similarity": 0.91,
            "content": "完整私有正文不得进入工具返回",
        }]}
        result = self.ask()
        for value in ("答案：\n测试答案", "文档：制度.pdf", "Dataset ID：dataset-1", "相关度：0.91"):
            self.assertIn(value, result)
        self.assertNotIn("完整私有正文", result)

    def test_multiple_document_references(self):
        self.reference = {"chunks": [
            {"document_id": "doc-1", "document_name": "A.pdf", "dataset_id": "ds-1"},
            {"document_id": "doc-2", "document_name": "B.docx", "dataset_id": "ds-2"},
        ]}
        result = self.ask()
        self.assertIn("1. 文档：A.pdf", result)
        self.assertIn("2. 文档：B.docx", result)

    def test_duplicate_document_keeps_first_order_and_highest_score(self):
        self.reference = {"chunks": [
            {"document_id": "doc-1", "document_name": "A.pdf", "similarity": 0.4},
            {"document_id": "doc-2", "document_name": "B.pdf", "similarity": 0.8},
            {"document_id": "doc-1", "document_name": "A.pdf", "similarity": 0.95},
        ]}
        result = self.ask()
        self.assertEqual(result.count("文档：A.pdf"), 1)
        self.assertLess(result.index("文档：A.pdf"), result.index("文档：B.pdf"))
        self.assertIn("相关度：0.95", result)
        self.assertNotIn("相关度：0.4", result)

    def test_citations_are_limited_to_five_in_sdk_order(self):
        self.reference = {"chunks": [
            {"document_id": f"doc-{index}", "document_name": f"{index}.pdf"}
            for index in range(1, 8)
        ]}
        result = self.ask()
        for index in range(1, 6):
            self.assertIn(f"文档：{index}.pdf", result)
        self.assertNotIn("文档：6.pdf", result)
        self.assertNotIn("文档：7.pdf", result)

    def test_reference_none_preserves_answer(self):
        self.reference = None
        result = self.ask()
        self.assertIn("答案：\n测试答案", result)
        self.assertIn("RAGFlow 本次未返回可用文档引用", result)

    def test_empty_chunks_preserve_answer(self):
        self.reference = {"chunks": []}
        result = self.ask()
        self.assertIn("答案：\n测试答案", result)
        self.assertIn("RAGFlow 本次未返回可用文档引用", result)

    def test_missing_document_name_falls_back_to_document_id(self):
        self.reference = {"chunks": [{"document_id": "doc-1", "dataset_id": "ds-1"}]}
        result = self.ask()
        self.assertIn("Document ID：doc-1", result)
        self.assertNotIn("文档：", result)

    def test_missing_document_id_uses_document_name(self):
        self.reference = {"chunks": [{"document_name": "A.pdf", "dataset_id": "ds-1"}]}
        self.assertIn("文档：A.pdf", self.ask())

    def test_missing_dataset_id_omits_dataset_line(self):
        self.reference = {"chunks": [{"document_id": "doc-1", "document_name": "A.pdf"}]}
        result = self.ask()
        self.assertIn("文档：A.pdf", result)
        self.assertNotIn("Dataset ID", result)

    def test_missing_score_omits_relevance_line(self):
        self.reference = {"chunks": [{"document_id": "doc-1", "document_name": "A.pdf"}]}
        result = self.ask()
        self.assertIn("文档：A.pdf", result)
        self.assertNotIn("相关度", result)

    def test_partially_malformed_chunks_do_not_drop_valid_source(self):
        self.reference = {"chunks": [
            123,
            {"metadata": {"secret": "不得输出"}},
            {"document_id": "doc-1", "document_name": "A.pdf", "similarity": "bad"},
        ]}
        result = self.ask()
        self.assertIn("来源信息不完整", result)
        self.assertIn("文档：A.pdf", result)
        self.assertNotIn("不得输出", result)
        self.assertNotIn("相关度", result)

    def test_citation_formatter_failure_preserves_answer_and_cleanup(self):
        with patch(
            "app.tools.ragflow_tools._format_answer_with_citations",
            side_effect=RuntimeError("offline-secret-key"),
        ):
            result = self.ask()
        self.assertIn("答案：\n测试答案", result)
        self.assertIn("引用信息解析不完整", result)
        self.assertNotIn("offline-secret-key", result)
        self.assert_deleted()

    def test_cleanup_failure_preserves_answer_and_citation(self):
        self.reference = {"chunks": [{
            "document_id": "doc-1", "document_name": "A.pdf",
            "dataset_id": "ds-1", "score": 0.88,
        }]}
        self.delete_error = requests.ConnectionError("offline-secret-key")
        result = self.ask()
        for value in ("答案：\n测试答案", "文档：A.pdf", "Dataset ID：ds-1",
                      "相关度：0.88", "临时会话清理失败"):
            self.assertIn(value, result)
        self.assert_deleted()

    def test_answer_citation_and_finally_delete(self):
        result = self.ask()
        self.assertIn("答案：\n测试答案", result)
        self.assertIn("来源：\n1. 文档：sample.pdf", result)
        self.assert_deleted()

    def test_tool_message_contains_citation_without_raw_reference(self):
        self.reference = {"chunks": [{
            "document_id": "doc-1", "document_name": "A.pdf",
            "dataset_id": "ds-1", "content": "超长正文标记" * 500,
            "metadata": {"internal": "内部元数据标记"},
        }]}
        message = ragflow_tools.create_ask_delete.invoke({
            "type": "tool_call", "id": "call-offline", "name": "create_ask_delete",
            "args": {"chat_name": "助手", "question": "测试问题"},
        })
        self.assertEqual(message.type, "tool")
        self.assertIn("答案：\n测试答案", message.content)
        self.assertIn("文档：A.pdf", message.content)
        self.assertIn("Dataset ID：ds-1", message.content)
        self.assertNotIn("超长正文标记", message.content)
        self.assertNotIn("内部元数据标记", message.content)
        self.assertNotIn("session-1", message.content)

    def test_all_http_requests_have_timeout(self):
        self.ask()
        for _, _, kwargs in self.calls:
            self.assertEqual(kwargs["timeout"], (10, 30))

    def test_missing_session_id_never_deletes_unspecified_sessions(self):
        self.empty_session_id = True
        result = self.ask()
        self.assertIn("Session ID", result)
        self.assertIn("清理失败", result)
        self.assertEqual([m for m, _, _ in self.calls], ["GET", "POST"])

    def test_import_does_not_read_config_or_construct_client(self):
        with patch("app.ragflow.rag_config._load_ragflow_env") as loader:
            with patch("ragflow_sdk.RAGFlow.__init__", return_value=None) as constructor:
                importlib.reload(ragflow_tools)
            loader.assert_not_called()
            constructor.assert_not_called()
        self.http_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
