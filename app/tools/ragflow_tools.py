"""
RAGFlow 工具：通过 SDK 查询助手和临时 Session；配置在调用时校验。
未配置 RAGFlow 不影响其他 Agent 模块导入。
"""

import math
from collections.abc import Mapping
from urllib.parse import urlsplit

import requests
from langchain_core.tools import tool
from ragflow_sdk import RAGFlow

from app.api.monitor import monitor
from app.ragflow.rag_config import _load_ragflow_env


_REQUEST_TIMEOUT = (10, 30)  # 连接/读取超时（秒），不是任务总时限。
_MAX_CITATIONS = 5


class _RAGFlowError(Exception):
    """可安全返回的本地错误，不包含服务端原文或凭据。"""


class _RAGFlowConfigError(_RAGFlowError):
    pass


class _RAGFlowClient(RAGFlow):
    """保留 SDK Chat/Session 协议，仅补充本线路的 HTTP 传输包装。"""

    def _request(self, method, path, **kwargs):
        with requests.request(
            method, self.api_url + path,
            headers=self.authorization_header,
            timeout=_REQUEST_TIMEOUT, **kwargs,
        ) as response:
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise _RAGFlowError("服务返回无效 JSON。") from exc
            if not isinstance(payload, dict) or "code" not in payload:
                raise _RAGFlowError("服务响应缺少业务状态码。")
            if payload["code"] != 0:
                code = payload["code"]
                detail = f"（code={code}）" if type(code) is int else ""
                raise _RAGFlowError(f"服务返回业务错误{detail}。")
            # 非流式响应已读入内存，关闭连接后 SDK 仍可 json()。
            return response

    def get(self, path, params=None, json=None):
        return self._request("GET", path, params=params, json=json)

    def post(self, path, json=None, stream=False, files=None):
        if stream:
            raise _RAGFlowError("当前工具仅支持非流式问答。")
        return self._request("POST", path, json=json, stream=False, files=files)

    def delete(self, path, json):
        return self._request("DELETE", path, json=json)


def _get_ragflow_client() -> RAGFlow:
    """校验配置并延迟创建 Client，不缓存无效配置。"""
    api_key, base_url = _load_ragflow_env()
    if not isinstance(base_url, str) or not base_url.strip():
        raise _RAGFlowConfigError("RAGFlow 未配置：缺少 RAGFLOW_API_URL。")
    if not isinstance(api_key, str) or not api_key.strip():
        raise _RAGFlowConfigError("RAGFlow 未配置：缺少 RAGFLOW_API_KEY。")
    base_url = base_url.strip().rstrip("/")
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme in {"http", "https"} and parsed.hostname
            and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment
            and not any(char.isspace() for char in base_url)
        )
        parsed.port  # 验证端口；不在错误中返回原始地址。
    except ValueError:
        valid = False
    if not valid:
        raise _RAGFlowConfigError(
            "RAGFlow 配置错误：RAGFLOW_API_URL 必须是合法 HTTP/HTTPS 服务地址。"
        )
    if "\r" in api_key or "\n" in api_key:
        raise _RAGFlowConfigError("RAGFlow 配置错误：RAGFLOW_API_KEY 格式无效。")
    return _RAGFlowClient(api_key=api_key.strip(), base_url=base_url)


def _error_detail(exc: Exception) -> str:
    """不返回 SDK/Requests 异常原文，避免 URL、Key 或响应内容外泄。"""
    if isinstance(exc, _RAGFlowError):
        return str(exc)
    if isinstance(exc, requests.Timeout):
        return "请求超时。"
    if isinstance(exc, requests.ConnectionError):
        return "无法连接服务。"
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return f"HTTP {status}。" if status is not None else "HTTP 请求失败。"
    if isinstance(exc, requests.RequestException):
        return "网络请求失败。"
    return "SDK 调用或响应处理失败。"


def _citation_value(chunk, *names):
    """兼容 SDK 保留的字典 chunk 和对象 chunk；字段异常按缺失处理。"""
    for name in names:
        try:
            value = chunk.get(name) if isinstance(chunk, Mapping) else getattr(chunk, name, None)
        except Exception:
            continue
        if value is not None and value != "":
            return value
    return None


def _citation_text(value, max_length=200):
    """把来源标识压缩为单行，避免输出对象 repr 或无限增长的字段。"""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = " ".join(str(value).split())
    return text[:max_length] if text else None


def _citation_score(chunk):
    value = _citation_value(chunk, "score", "similarity")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _format_citations(reference) -> str:
    """按 SDK 顺序输出至多五个唯一文档来源，不包含 chunk 正文或 metadata。"""
    if reference is None:
        return "RAGFlow 本次未返回可用文档引用。"
    if isinstance(reference, Mapping):
        chunks = reference.get("chunks")
    elif isinstance(reference, (list, tuple)):
        chunks = reference
    else:
        try:
            chunks = getattr(reference, "chunks", None)
        except Exception:
            return "引用信息解析不完整。"
    if not chunks:
        return "RAGFlow 本次未返回可用文档引用。"
    if not isinstance(chunks, (list, tuple)):
        return "引用信息解析不完整。"

    citations = []
    citation_indexes = {}
    incomplete_seen = False
    for chunk in chunks:
        document_name = _citation_text(_citation_value(chunk, "document_name", "doc_name"))
        document_id = _citation_text(_citation_value(chunk, "document_id"))
        dataset_id = _citation_text(_citation_value(chunk, "dataset_id"))
        score = _citation_score(chunk)
        if document_id:
            key = ("document_id", document_id)
        elif document_name:
            key = ("document_name", document_name.casefold())
        else:
            if incomplete_seen:
                continue
            incomplete_seen = True
            key = ("incomplete",)
        citation = {
            "document_name": document_name,
            "document_id": document_id,
            "dataset_id": dataset_id,
            "score": score,
        }
        if key in citation_indexes:
            index = citation_indexes[key]
            previous_score = citations[index]["score"]
            if score is not None and (previous_score is None or score > previous_score):
                citations[index] = citation
            continue
        if len(citations) >= _MAX_CITATIONS:
            continue
        citation_indexes[key] = len(citations)
        citations.append(citation)

    if not citations:
        return "引用信息解析不完整。"
    lines = []
    for index, citation in enumerate(citations, 1):
        if citation["document_name"]:
            lines.append(f"{index}. 文档：{citation['document_name']}")
        elif citation["document_id"]:
            lines.append(f"{index}. Document ID：{citation['document_id']}")
        else:
            lines.append(f"{index}. 来源信息不完整")
        if citation["dataset_id"]:
            lines.append(f"   Dataset ID：{citation['dataset_id']}")
        if citation["score"] is not None:
            lines.append(f"   相关度：{citation['score']:.6g}")
    return "\n".join(lines)


def _format_answer_with_citations(answer: str, reference) -> str:
    return f"答案：\n{answer}\n\n来源：\n{_format_citations(reference)}"


@tool
def get_assistant_list() -> str:
    """
    查询 RAGFlow 助手，返回名称、Chat ID、描述及关联 Dataset ID。

    先通过此工具确认助手，再调用 create_ask_delete。
    Dataset ID 是标识符，不等于知识库名称。
    :return: 助手列表文本；无助手或失败时返回明确说明。
    """
    try:
        client = _get_ragflow_client()
        monitor.report_tool(tool_name="ragflow聊天助手列表查询工具：get_assistant_list")
        chats = client.list_chats()
        if not chats:
            return "没有任何可用助手"
        rows = []
        for chat in chats:
            dataset_ids = getattr(chat, "dataset_ids", None) or []
            rows.append(
                f"助手名称：{getattr(chat, 'name', None) or '未命名'}; "
                f"Chat ID：{getattr(chat, 'id', None) or '未提供'}; "
                f"描述：{getattr(chat, 'description', None) or '无描述'}; "
                f"关联 Dataset ID：{', '.join(map(str, dataset_ids)) or '未提供'}"
            )
        return "\n".join(rows)
    except _RAGFlowConfigError as exc:
        return str(exc)
    except Exception as exc:
        return f"RAGFlow 助手列表查询失败：{_error_detail(exc)}"


@tool
def create_ask_delete(chat_name, question) -> str:
    """
    向唯一匹配的 RAGFlow 助手创建临时 Session、提问，并尝试清理 Session。

    调用前先通过 get_assistant_list 确认名称；同名多个助手时拒绝提问。
    :param chat_name: 来自助手列表的名称。
    :param question: 围绕用户需求的问题。
    :return: 回答或失败说明；清理失败附加警告，不覆盖已取得的答案。
    """
    session = None
    use_chat = None
    result = ""
    reference = None
    cleanup_warning = ""
    try:
        client = _get_ragflow_client()
        monitor.report_tool(
            tool_name="ragflow提问助手工具：create_ask_delete",
            args={"chat_name": chat_name, "question": question},
        )
        chats = [
            chat for chat in (client.list_chats(name=chat_name) or [])
            if getattr(chat, "name", None) == chat_name
        ]
        if not chats:
            return "RAGFlow 未找到指定助手。"
        if len(chats) > 1:
            return "存在多个同名助手，无法唯一确定目标。"
        use_chat = chats[0]
        session = use_chat.create_session(name="temp_session_ask")
        if not getattr(session, "id", None):
            raise _RAGFlowError("服务未返回有效 Session ID。")
        # 0.25.2 的 ask 即使 stream=False 也返回生成器。
        # 只取非流式完整回答，移除不可靠的 delta/cumulative 前缀拼接。
        for message in session.ask(question=question, stream=False):
            if not isinstance(message.content, str):
                raise _RAGFlowError("服务返回的答案格式无效。")
            result = message.content
            reference = getattr(message, "reference", None)
        if not result.strip():
            result = "RAGFlow 未返回有效答案。"
        else:
            try:
                result = _format_answer_with_citations(result, reference)
            except Exception:
                # 引用异常不影响已经取得的答案，也不暴露 traceback 或 SDK 对象。
                result = f"答案：\n{result}\n\n来源：\n引用信息解析不完整。"
    except _RAGFlowConfigError as exc:
        result = str(exc)
    except Exception as exc:
        result = f"RAGFlow 提问失败：{_error_detail(exc)}"
    finally:
        if session is not None and use_chat is not None:
            try:
                session_id = getattr(session, "id", None)
                if not session_id:
                    raise _RAGFlowError("缺少 Session ID，无法安全指定清理目标。")
                use_chat.delete_sessions(ids=[session_id])
            except Exception as exc:
                cleanup_warning = f"RAGFlow 临时会话清理失败：{_error_detail(exc)}"
    return f"{result}\n{cleanup_warning}" if cleanup_warning else result
