"""Offline Knowledge Agent registration, KB handoff and tool boundary checks."""

import contextlib
from io import StringIO
import json
from unittest import TestCase
from unittest.mock import patch

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel, FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.context import reset_session_context, set_session_context, set_thread_context
from app.local_rag.embeddings import get_embedding_adapter
from app.local_rag.embeddings import LocalEmbeddingAdapter
from app.local_rag.ingestion import ingest_document
from app.local_rag.retrieval import LocalRAGRetriever
from app.local_rag.storage import KnowledgeBaseAccessError
from app.local_rag.vector_store import LocalVectorStore
from app.tools import local_knowledge_base_tool as tool_module
from app.tools.local_knowledge_base_tool import search_local_knowledge_base
from local_rag_test_support import OfflineDocumentTestCase
from test_local_rag_embeddings import FakeModel


with contextlib.redirect_stdout(StringIO()):
    from app.agent.prompts import load_yaml, sub_agents_content
    from app.agent.subagents.knowledge_base_agent import (
        build_knowledge_agent_task, knowledge_base_agent,
    )


class ScriptedToolModel(FakeMessagesListChatModel):
    """Scripted graph transport; it does not emulate LLM decision quality."""

    def bind_tools(self, tools, **kwargs):
        return self


class KnowledgeAgentContractTests(TestCase):
    def test_yaml_safe_load_and_agent_description(self):
        loaded = load_yaml(__import__("pathlib").Path("app/prompt/prompts.yml"))
        self.assertIn("local_knowledge", loaded["sub_agents"])
        self.assertEqual(knowledge_base_agent["name"], "企业知识助手")
        for phrase in ("当前会话", "运营 SOP", "历史营销活动复盘", "互联网", "MySQL"):
            self.assertIn(phrase, knowledge_base_agent["description"])

    def test_only_business_tool_is_local_rag(self):
        self.assertEqual(knowledge_base_agent["tools"], [search_local_knowledge_base])
        self.assertEqual([tool.name for tool in knowledge_base_agent["tools"]],
                         ["search_local_knowledge_base"])
        self.assertNotIn("get_assistant_list", str([tool.name for tool in knowledge_base_agent["tools"]]))
        self.assertNotIn("create_ask_delete", str([tool.name for tool in knowledge_base_agent["tools"]]))

    def test_tool_schema_exposes_no_session_or_paths(self):
        self.assertEqual(set(search_local_knowledge_base.args),
                         {"knowledge_base_id", "query", "top_k"})
        for name in ("thread_id", "session_dir", "filesystem_path", "index_path", "embedding_model"):
            self.assertNotIn(name, search_local_knowledge_base.args)

    def test_prompt_requires_tool_evidence_and_preserves_citation(self):
        prompt = knowledge_base_agent["system_prompt"]
        for phrase in ("search_local_knowledge_base", "Evidence", "[C1]", "[C2]",
                       "不得凭模型记忆", "不得改号", "编造不存在的 Citation"):
            self.assertIn(phrase, prompt)

    def test_prompt_missing_evidence_and_kb_behavior(self):
        prompt = knowledge_base_agent["system_prompt"]
        for phrase in ("knowledge_base_id", "没有 ID", "不得猜测", "没有检索到足够证据",
                       "不得用模型记忆补出企业内部结论"):
            self.assertIn(phrase, prompt)

    def test_prompt_distinguishes_network_and_database(self):
        prompt = knowledge_base_agent["system_prompt"]
        for phrase in ("互联网最新趋势", "网络搜索助手", "GMV", "SKU", "数据库查询助手"):
            self.assertIn(phrase, prompt)

    def test_ragflow_definition_retained_but_not_registered(self):
        self.assertIn("ragflow", sub_agents_content)
        self.assertIn("get_assistant_list", sub_agents_content["ragflow"]["system_prompt"])
        self.assertEqual(knowledge_base_agent["tools"], [search_local_knowledge_base])

    def test_offline_agent_graph_contains_local_tool_without_loading_model(self):
        adapter = get_embedding_adapter()
        self.assertFalse(adapter.is_loaded)
        fake = FakeListChatModel(responses=["离线占位"])
        graph = create_deep_agent(model=fake, tools=knowledge_base_agent["tools"],
                                  subagents=[], system_prompt=knowledge_base_agent["system_prompt"])
        names = set(graph.nodes["tools"].bound.tools_by_name)
        self.assertIn("search_local_knowledge_base", names)
        self.assertNotIn("get_assistant_list", names)
        self.assertNotIn("create_ask_delete", names)
        self.assertFalse(adapter.is_loaded)
        self.assertEqual(fake.i, 0)

    def test_agent_construction_has_no_external_network(self):
        with patch("socket.socket.connect", side_effect=AssertionError("network")) as connect:
            create_deep_agent(model=FakeListChatModel(responses=["测试"]),
                              tools=knowledge_base_agent["tools"], subagents=[],
                              system_prompt=knowledge_base_agent["system_prompt"])
            connect.assert_not_called()


class KnowledgeTaskContextTests(OfflineDocumentTestCase):
    def test_explicit_kb_context_is_authorized_and_escaped(self):
        kb = self.storage.create_knowledge_base("企业内部资料")
        question = "请查询历史 618 活动复盘，\n看看流量上涨但销售没有同步上涨的原因。"
        task = build_knowledge_agent_task(kb.knowledge_base_id, question,
                                          storage=self.storage)
        self.assertTrue(task.startswith("企业内部知识检索任务："))
        self.assertEqual(json.loads(task.split("：", 1)[1]),
                         {"knowledge_base_id": kb.knowledge_base_id, "query": question})
        self.assertNotIn(str(self.session_dir), task)
        self.assertNotIn("session-A", task)

    def test_missing_or_invalid_kb_is_not_guessed(self):
        for value in ("", "kb_default", "kb_001", "../kb", "kb_" + "a" * 32):
            with self.subTest(value=value), self.assertRaises(KnowledgeBaseAccessError):
                build_knowledge_agent_task(value, "请查内部资料", storage=self.storage)

    def test_empty_question_rejected(self):
        kb = self.storage.create_knowledge_base("企业内部资料")
        for value in ("", "  ", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_knowledge_agent_task(kb.knowledge_base_id, value,
                                           storage=self.storage)

    def test_kb_from_other_session_rejected(self):
        kb = self.storage.create_knowledge_base("A 会话资料")
        other = self.upload_root / "session_session-B"
        other.mkdir()
        session = set_session_context(str(other))
        thread = set_thread_context("session-B")
        try:
            with self.assertRaises(KnowledgeBaseAccessError):
                build_knowledge_agent_task(kb.knowledge_base_id, "请查资料",
                                           storage=self.storage)
        finally:
            reset_session_context(session, thread)

    def test_context_build_does_not_load_embedding_or_call_external_service(self):
        kb = self.storage.create_knowledge_base("企业资料")
        adapter = get_embedding_adapter()
        self.assertFalse(adapter.is_loaded)
        with patch("socket.socket.connect", side_effect=AssertionError("network")) as connect:
            build_knowledge_agent_task(kb.knowledge_base_id, "历史复盘",
                                       storage=self.storage)
            connect.assert_not_called()
        self.assertFalse(adapter.is_loaded)


class ScriptedKnowledgeGraphTests(OfflineDocumentTestCase):
    def graph(self, responses):
        model = ScriptedToolModel(responses=responses)
        return create_deep_agent(model=model, tools=knowledge_base_agent["tools"],
                                 subagents=[], system_prompt=knowledge_base_agent["system_prompt"])

    def test_internal_question_reaches_real_local_tool_and_returns_citation(self):
        kb = self.storage.create_knowledge_base("历史活动")
        result = ingest_document(
            kb.knowledge_base_id,
            self.upload("618活动复盘.md", "618 美妆活动流量上涨，但结算页转化率下降。"),
            storage=self.storage, loader=self.loader,
        )
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        LocalVectorStore(self.storage).build_index(
            kb.knowledge_base_id, adapter.embed_documents(result.chunks), adapter.metadata,
        )
        delegated = build_knowledge_agent_task(
            kb.knowledge_base_id,
            "请查询历史 618 活动复盘，看看流量上涨但销售没有同步上涨的原因。",
            storage=self.storage,
        )
        scripted = [
            AIMessage(content="", tool_calls=[{
                "name": "search_local_knowledge_base",
                "args": {"knowledge_base_id": kb.knowledge_base_id,
                         "query": "618 活动流量上涨但销售没同步上涨的原因", "top_k": 1},
                "id": "local-call-1",
            }]),
            AIMessage(content="活动流量上涨，但结算页转化率下降。[C1]"),
        ]
        graph = self.graph(scripted)
        retriever = LocalRAGRetriever(self.storage, adapter)
        with patch.object(tool_module, "_get_retriever", return_value=retriever):
            state = graph.invoke({"messages": [HumanMessage(content=delegated)]},
                                 config={"recursion_limit": 10})
        messages = state["messages"]
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)
                         and message.name == "search_local_knowledge_base"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, "local-call-1")
        self.assertIn("[C1]", tool_messages[0].content)
        self.assertIn("618活动复盘.md", tool_messages[0].content)
        self.assertEqual(tool_messages[0].artifact[0].citation.chunk_id,
                         result.chunks[0].chunk_id)
        self.assertIn("[C1]", messages[-1].content)

    def test_scripted_web_question_does_not_call_local_tool(self):
        graph = self.graph([AIMessage(content="近期互联网趋势属于网络搜索助手。")])
        with patch.object(tool_module, "_get_retriever", side_effect=AssertionError("wrong route")) as getter:
            state = graph.invoke({"messages": [HumanMessage(
                content="搜索最近一周美妆行业的新趋势。")]}, config={"recursion_limit": 10})
        getter.assert_not_called()
        self.assertIn("网络搜索助手", state["messages"][-1].content)

    def test_scripted_structured_data_question_does_not_call_local_tool(self):
        graph = self.graph([AIMessage(content="昨天各 SKU 的 GMV 属于数据库查询助手。")])
        with patch.object(tool_module, "_get_retriever", side_effect=AssertionError("wrong route")) as getter:
            state = graph.invoke({"messages": [HumanMessage(
                content="查询昨天每个 SKU 的 GMV。")]}, config={"recursion_limit": 10})
        getter.assert_not_called()
        self.assertIn("数据库查询助手", state["messages"][-1].content)
