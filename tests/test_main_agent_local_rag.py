"""Offline Main-to-Knowledge routing and selected-KB boundary checks."""

import asyncio
import contextlib
from io import StringIO
import inspect
import json
import socket
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from deepagents import create_deep_agent
from fastapi import HTTPException
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.api.context import (
    get_selected_knowledge_base_context,
    get_thread_context,
    has_selected_knowledge_base_context,
    reset_selected_knowledge_base_context,
    set_selected_knowledge_base_context,
)
from app.local_rag.embeddings import LocalEmbeddingAdapter, get_embedding_adapter
from app.local_rag.ingestion import ingest_document
from app.local_rag.retrieval import LocalRAGRetriever
from app.local_rag.storage import KnowledgeBaseAccessError
from app.local_rag.vector_store import LocalVectorStore
from app.tools import local_knowledge_base_tool as tool_module
from app.tools.local_knowledge_base_tool import search_local_knowledge_base
from local_rag_test_support import OfflineDocumentTestCase
from test_local_rag_embeddings import FakeModel


_original_socket_connect = socket.socket.connect


def run_fixture_coroutine(coroutine):
    """Allow only asyncio's Windows-local socketpair under the offline fixture."""
    def connect(sock, address):
        if (isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}
                and any(frame.function == "_fallback_socketpair" for frame in inspect.stack())):
            return _original_socket_connect(sock, address)
        raise AssertionError("External network prohibited")

    with patch("socket.socket.connect", connect):
        return asyncio.run(coroutine)


with contextlib.redirect_stdout(StringIO()):
    from app.agent import main_agent as main_module
    from app.agent.prompts import main_agent_content
    from app.agent.subagents.database_query_agent import database_query_agent
    from app.agent.subagents.knowledge_base_agent import (
        build_knowledge_agent_task, knowledge_base_agent,
    )
    from app.agent.subagents.network_search_agent import network_search_agent
    from app.api import server


class ScriptedToolModel(FakeMessagesListChatModel):
    """Predetermined messages test graph wiring, not actual LLM decisions."""

    def bind_tools(self, tools, **kwargs):
        return self


def task_call(agent_name: str, description: str, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "task",
        "args": {"subagent_type": agent_name, "description": description},
        "id": call_id,
    }])


class MainRegistryTests(TestCase):
    def test_main_registers_three_business_agents_but_not_local_tool(self):
        names = [database_query_agent["name"], network_search_agent["name"],
                 knowledge_base_agent["name"]]
        self.assertEqual(len(set(names)), 3)
        tools = main_module.main_agent.nodes["tools"].bound.tools_by_name
        self.assertIn("task", tools)
        self.assertNotIn("search_local_knowledge_base", tools)
        description = tools["task"].description
        for name in names:
            self.assertIn(name, description)
        available = description.split("Available agent types and the tools they have access to:", 1)[1]
        available = available.split("When using the Task tool", 1)[0]
        self.assertNotIn("general-purpose", available)
        self.assertNotIn("RAGFlow", description)
        self.assertEqual([tool.name for tool in knowledge_base_agent["tools"]],
                         ["search_local_knowledge_base"])

    def test_main_prompt_routes_sources_and_preserves_real_citations(self):
        prompt = main_agent_content["system_prompt"]
        self.assertNotIn("RAGFlow", prompt)
        for phrase in ("企业知识助手", "数据库查询助手", "网络搜索助手",
                       "knowledge_base_id", "task.description", "[C1]", "[C2]",
                       "不得改号", "不存在的 Citation", "没有足够证据", "综合"):
            self.assertIn(phrase, prompt)
        self.assertIn("当前任务未指定可查询的企业知识库", prompt)
        self.assertIn("把网络搜索和数据库结果冒充企业知识库引用", prompt)

    def test_task_request_accepts_optional_selected_kb(self):
        self.assertIsNone(server.TaskRequest(query="x").knowledge_base_id)
        self.assertEqual(server.TaskRequest(query="x", knowledge_base_id="kb_test").knowledge_base_id,
                         "kb_test")


class SelectedKnowledgeBaseTests(OfflineDocumentTestCase):
    def test_valid_selection_is_session_authorized_and_context_restored(self):
        kb = self.storage.create_knowledge_base("活动资料")
        with patch.object(main_module, "LocalRAGStorage", return_value=self.storage):
            self.assertEqual(main_module.validate_selected_knowledge_base(
                "session-A", kb.knowledge_base_id), kb.knowledge_base_id)
        self.assertEqual(get_thread_context(), "session-A")
        self.assertFalse(has_selected_knowledge_base_context())

    def test_absent_foreign_and_invalid_kb_are_never_selected(self):
        kb = self.storage.create_knowledge_base("A 资料")
        with patch.object(main_module, "LocalRAGStorage", return_value=self.storage):
            self.assertIsNone(main_module.validate_selected_knowledge_base("session-A", None))
            for value in ("kb_default", "kb_" + "a" * 32, kb.knowledge_base_id):
                session = "session-B" if value == kb.knowledge_base_id else "session-A"
                with self.subTest(value=value), self.assertRaises(KnowledgeBaseAccessError):
                    main_module.validate_selected_knowledge_base(session, value)
        self.assertEqual(get_thread_context(), "session-A")

    def test_multiple_kbs_require_explicit_selection_without_filesystem_path(self):
        first = self.storage.create_knowledge_base("第一份")
        second = self.storage.create_knowledge_base("第二份")
        missing = main_module._build_knowledge_base_instruction(None, "查内部文档")
        self.assertIn("未选择企业知识库", missing)
        self.assertNotIn(first.knowledge_base_id, missing)
        self.assertNotIn(second.knowledge_base_id, missing)
        with patch.object(main_module, "LocalRAGStorage", return_value=self.storage):
            selected = main_module._build_knowledge_base_instruction(second.knowledge_base_id,
                                                                       "查内部文档")
        self.assertIn(second.knowledge_base_id, selected)
        self.assertNotIn(first.knowledge_base_id, selected)
        self.assertNotIn(str(self.base), selected)
        delegated = selected.split("企业内部知识检索任务：", 1)[1].strip()
        self.assertEqual(json.loads(delegated)["knowledge_base_id"], second.knowledge_base_id)

    def test_selected_context_resets_and_blocks_wrong_or_missing_kb_before_retrieval(self):
        first = self.storage.create_knowledge_base("第一份")
        second = self.storage.create_knowledge_base("第二份")
        for selected, requested in ((None, first.knowledge_base_id),
                                    (first.knowledge_base_id, second.knowledge_base_id)):
            token = set_selected_knowledge_base_context(selected)
            try:
                with patch.object(tool_module, "_get_retriever") as getter:
                    with self.assertRaises(KnowledgeBaseAccessError):
                        search_local_knowledge_base.invoke({
                            "knowledge_base_id": requested, "query": "活动复盘",
                        })
                    getter.assert_not_called()
            finally:
                reset_selected_knowledge_base_context(token)
        self.assertFalse(has_selected_knowledge_base_context())
        self.assertIsNone(get_selected_knowledge_base_context())

    def test_runtime_passes_selection_without_host_path_and_restores_context(self):
        kb = self.storage.create_knowledge_base("活动资料")
        captured = []

        class CaptureGraph:
            async def astream(self, state, config):
                captured.append((state, config, get_selected_knowledge_base_context(),
                                 has_selected_knowledge_base_context()))
                if False:
                    yield {}

        with (patch.object(main_module, "LocalRAGStorage", return_value=self.storage),
              patch.object(main_module, "project_root_path", self.base),
              patch.object(main_module, "main_agent", CaptureGraph()),
              patch.object(main_module.monitor, "report_session_dir")):
            run_fixture_coroutine(main_module.run_deep_agent(
                "查 618 复盘", "session-A", kb.knowledge_base_id))

        self.assertEqual(len(captured), 1)
        state, config, selected, active = captured[0]
        self.assertEqual(selected, kb.knowledge_base_id)
        self.assertTrue(active)
        self.assertEqual(config["configurable"]["thread_id"], "session-A")
        self.assertIn(kb.knowledge_base_id, state["messages"][0]["content"])
        self.assertNotIn(str(self.base), state["messages"][0]["content"])
        self.assertFalse(has_selected_knowledge_base_context())

    def test_runtime_without_selection_is_explicit_and_cannot_retrieve(self):
        captured = []

        class CaptureGraph:
            async def astream(self, state, config):
                captured.append((state["messages"][0]["content"],
                                 get_selected_knowledge_base_context(),
                                 has_selected_knowledge_base_context()))
                if False:
                    yield {}

        with (patch.object(main_module, "project_root_path", self.base),
              patch.object(main_module, "main_agent", CaptureGraph()),
              patch.object(main_module.monitor, "report_session_dir")):
            run_fixture_coroutine(main_module.run_deep_agent("查内部文档", "session-A"))
        self.assertEqual(len(captured), 1)
        self.assertIn("未选择企业知识库", captured[0][0])
        self.assertEqual(captured[0][1:], (None, True))
        self.assertFalse(has_selected_knowledge_base_context())

    def test_invalid_runtime_selection_rejects_before_agent_or_output_creation(self):
        with (patch.object(main_module, "LocalRAGStorage", return_value=self.storage),
              patch.object(main_module, "project_root_path", self.base),
              patch.object(main_module.main_agent, "astream") as stream):
            with self.assertRaises(KnowledgeBaseAccessError):
                run_fixture_coroutine(main_module.run_deep_agent(
                    "查内部文档", "session-A", "kb_default"))
            stream.assert_not_called()
        self.assertFalse((self.base / "output").exists())

    def test_selected_real_tool_reaches_local_rag_evidence(self):
        kb = self.storage.create_knowledge_base("618 复盘")
        result = ingest_document(kb.knowledge_base_id,
                                 self.upload("618活动复盘.md", "美妆活动流量上涨，但结算页转化率下降。"),
                                 storage=self.storage, loader=self.loader)
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        LocalVectorStore(self.storage).build_index(kb.knowledge_base_id,
                                                   adapter.embed_documents(result.chunks),
                                                   adapter.metadata)
        retriever = LocalRAGRetriever(self.storage, adapter)
        token = set_selected_knowledge_base_context(kb.knowledge_base_id)
        try:
            with patch.object(tool_module, "_get_retriever", return_value=retriever):
                answer = search_local_knowledge_base.invoke({
                    "type": "tool_call", "id": "local-1", "name": "search_local_knowledge_base",
                    "args": {"knowledge_base_id": kb.knowledge_base_id,
                             "query": "618 美妆活动为什么转化下降？", "top_k": 1},
                })
        finally:
            reset_selected_knowledge_base_context(token)
        self.assertIsInstance(answer, ToolMessage)
        self.assertIn("[C1]", answer.content)
        self.assertIn("618活动复盘.md", answer.content)
        self.assertEqual(answer.artifact[0].citation.chunk_id, result.chunks[0].chunk_id)
        self.assertFalse(get_embedding_adapter().is_loaded)


class OfflineMainRoutingTests(OfflineDocumentTestCase):
    def graph(self, responses):
        return create_deep_agent(
            model=ScriptedToolModel(responses=responses),
            system_prompt=main_agent_content["system_prompt"],
            tools=[main_module.generate_markdown, main_module.convert_md_to_pdf,
                   main_module.read_file_content],
            subagents=[database_query_agent, network_search_agent, knowledge_base_agent],
            checkpointer=InMemorySaver(),
        )

    def test_internal_knowledge_route_runs_real_local_tool_and_returns_citation(self):
        kb = self.storage.create_knowledge_base("历史活动")
        result = ingest_document(kb.knowledge_base_id,
                                 self.upload("618复盘.md", "活动流量上涨，但结算页转化率下降。"),
                                 storage=self.storage, loader=self.loader)
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        LocalVectorStore(self.storage).build_index(kb.knowledge_base_id,
                                                   adapter.embed_documents(result.chunks),
                                                   adapter.metadata)
        retriever = LocalRAGRetriever(self.storage, adapter)
        question = "查一下去年 618 活动复盘中，为什么流量上涨但转化率下降。"
        description = build_knowledge_agent_task(kb.knowledge_base_id, question,
                                                  storage=self.storage)
        scripted = [
            task_call(knowledge_base_agent["name"], description, "task-knowledge"),
            AIMessage(content="", tool_calls=[{
                "name": "search_local_knowledge_base", "id": "search-1",
                "args": {"knowledge_base_id": kb.knowledge_base_id, "query": question,
                         "top_k": 1},
            }]),
            AIMessage(content="历史复盘显示结算页转化率下降。[C1]"),
            AIMessage(content="据《618复盘.md》，结算页转化率下降。[C1]"),
        ]
        graph = self.graph(scripted)
        token = set_selected_knowledge_base_context(kb.knowledge_base_id)
        try:
            with (patch.object(tool_module, "_get_retriever", return_value=retriever),
                  patch.object(retriever, "retrieve", wraps=retriever.retrieve) as retrieve):
                state = run_fixture_coroutine(graph.ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={"configurable": {"thread_id": "session-A"},
                            "recursion_limit": 16},
                ))
                retrieve.assert_called_once()
        finally:
            reset_selected_knowledge_base_context(token)
        delegated = [message for message in state["messages"] if isinstance(message, ToolMessage)
                     and message.name == "task"]
        self.assertEqual(len(delegated), 1)
        self.assertIn("[C1]", delegated[0].content)
        self.assertIn("[C1]", state["messages"][-1].content)

    def test_database_and_network_route_to_their_own_agents(self):
        for name, question in ((database_query_agent["name"], "查询最近 30 天美妆品类 GMV。"),
                               (network_search_agent["name"], "搜索最近一周美妆行业趋势。")):
            with self.subTest(name=name):
                graph = self.graph([
                    task_call(name, question, "route-1"),
                    AIMessage(content="离线子任务结果"),
                    AIMessage(content="离线汇总"),
                ])
                with patch.object(tool_module, "_get_retriever") as getter:
                    state = graph.invoke({"messages": [HumanMessage(content=question)]},
                                         config={"configurable": {"thread_id": "session-A"},
                                                 "recursion_limit": 16})
                    getter.assert_not_called()
                delegated = [message for message in state["messages"]
                             if isinstance(message, ToolMessage) and message.name == "task"]
                self.assertEqual(len(delegated), 1)
                self.assertEqual(state["messages"][-1].content, "离线汇总")

    def test_mixed_task_allows_three_sequential_business_agents(self):
        question = "结合去年 618 复盘、最近销售数据和竞品活动分析今年促销策略。"
        names = [knowledge_base_agent["name"], database_query_agent["name"],
                 network_search_agent["name"]]
        scripted = []
        for index, name in enumerate(names, 1):
            scripted.extend([task_call(name, f"子任务 {index}", f"task-{index}"),
                             AIMessage(content=f"来源 {index} 结果")])
        scripted.append(AIMessage(content="综合三类来源。"))
        graph = self.graph(scripted)
        state = graph.invoke({"messages": [HumanMessage(content=question)]},
                             config={"configurable": {"thread_id": "session-A"},
                                     "recursion_limit": 24})
        delegated = [message for message in state["messages"]
                     if isinstance(message, ToolMessage) and message.name == "task"]
        self.assertEqual(len(delegated), 3)
        self.assertEqual(state["messages"][-1].content, "综合三类来源。")


class TaskApiSelectionTests(TestCase):
    def test_api_passes_selected_kb_and_preserves_old_two_arg_call(self):
        async def check(request):
            return await server.run_task(request)

        for selected in (None, "kb_selected"):
            with self.subTest(selected=selected), patch.dict(server.active_tasks, {}, clear=True):
                with (patch.object(server, "validate_selected_knowledge_base") as validate,
                      patch.object(server, "run_deep_agent", new_callable=AsyncMock) as run):
                    request = server.TaskRequest(query="查资料", thread_id="session-A",
                                                 knowledge_base_id=selected)
                    result = asyncio.run(check(request))
                    self.assertEqual(result["thread_id"], "session-A")
                    if selected is None:
                        validate.assert_not_called()
                        run.assert_awaited_once_with("查资料", "session-A")
                    else:
                        validate.assert_called_once_with("session-A", selected)
                        run.assert_awaited_once_with("查资料", "session-A",
                                                     knowledge_base_id=selected)

    def test_invalid_kb_rejected_before_background_task(self):
        async def check():
            return await server.run_task(server.TaskRequest(
                query="查资料", thread_id="session-A", knowledge_base_id="kb_bad"))

        with (patch.object(server, "validate_selected_knowledge_base",
                           side_effect=KnowledgeBaseAccessError("denied")),
              patch.object(server.asyncio, "create_task") as create):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(check())
            self.assertEqual(raised.exception.status_code, 400)
            create.assert_not_called()
