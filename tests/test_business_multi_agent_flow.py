"""Three-agent business flow: local Knowledge, fake MySQL and fake Tavily."""

import contextlib
from io import StringIO
from unittest import TestCase
from unittest.mock import MagicMock, patch

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from mysql.connector import Error as MySQLError

from app.api.context import reset_selected_knowledge_base_context, set_selected_knowledge_base_context
from app.local_rag.embeddings import LocalEmbeddingAdapter
from app.local_rag.ingestion import ingest_document
from app.local_rag.retrieval import LocalRAGRetriever
from app.local_rag.vector_store import LocalVectorStore
from app.tools import db_tools, local_knowledge_base_tool, tavily_tool
from local_rag_test_support import OfflineDocumentTestCase
from test_local_rag_embeddings import FakeModel
from test_main_agent_local_rag import ScriptedToolModel, task_call


with contextlib.redirect_stdout(StringIO()):
    from app.agent import main_agent as main_module
    from app.agent.prompts import main_agent_content
    from app.agent.subagents.database_query_agent import database_query_agent
    from app.agent.subagents.knowledge_base_agent import build_knowledge_agent_task, knowledge_base_agent
    from app.agent.subagents.network_search_agent import network_search_agent


SQL = "SELECT COUNT(*) AS drug_count FROM drugs;"
PUBLIC_URL = "https://example.org/offline-market-example"


def fake_database():
    """Return a connector mock that understands only the teaching schema."""
    connector = MagicMock()
    cursor = MagicMock()
    state = {"sql": None}
    queries = []

    def execute(sql, params=None):
        queries.append((sql, params))
        state["sql"] = sql
        if sql == "SHOW TABLES":
            cursor.description = [("Tables_in_deepsearch_db",)]
        elif sql == db_tools._TABLE_EXISTS_QUERY:
            cursor.description = [("exists",)]
        elif sql == "SELECT * FROM `drugs` LIMIT 100":
            cursor.description = [("drug_id",), ("brand_name",)]
        elif sql == SQL:
            cursor.description = [("drug_count",)]
        else:
            raise AssertionError("unexpected SQL")

    def fetchall():
        if state["sql"] == "SHOW TABLES":
            return [("drugs",), ("inventory",), ("sales_records",)]
        if state["sql"] == "SELECT * FROM `drugs` LIMIT 100":
            return [(1, "教学药品")]
        if state["sql"] == SQL:
            return [(50,)]
        raise AssertionError("unexpected fetchall")

    cursor.execute.side_effect = execute
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.side_effect = fetchall
    connector.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cursor
    return connector, queries


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class BusinessFlowPromptTests(TestCase):
    def test_source_contract_and_conflict_handling(self):
        prompt = main_agent_content["system_prompt"]
        for phrase in ("多来源经营分析", "内部经营数据发现", "历史内部资料发现",
                       "外部市场发现", "综合原因", "运营建议", "时间范围", "统计口径",
                       "缺少某一来源时标明缺口", "不强行得出统一结论",
                       "不得把失败结果写成已确认事实"):
            self.assertIn(phrase, prompt)
        self.assertIn("不得把三种来源混为一种 Citation", prompt)
        self.assertIn("简单问题无需套用固定报告模板", prompt)

    def test_main_and_workers_keep_distinct_tool_registries(self):
        self.assertEqual([t.name for t in knowledge_base_agent["tools"]],
                         ["search_local_knowledge_base"])
        self.assertEqual([t.name for t in database_query_agent["tools"]],
                         ["list_sql_tables", "get_table_data", "execute_sql_query"])
        self.assertEqual([t.name for t in network_search_agent["tools"]], ["internet_search"])
        main_tools = main_module.main_agent.nodes["tools"].bound.tools_by_name
        self.assertIn("task", main_tools)
        for name in ("search_local_knowledge_base", "list_sql_tables", "get_table_data",
                     "execute_sql_query", "internet_search"):
            self.assertNotIn(name, main_tools)


class ScriptedBusinessFlowTests(OfflineDocumentTestCase):
    def run_graph(self, responses, question):
        graph = create_deep_agent(
            model=ScriptedToolModel(responses=responses),
            system_prompt=main_agent_content["system_prompt"],
            tools=[main_module.generate_markdown, main_module.convert_md_to_pdf,
                   main_module.read_file_content],
            subagents=[knowledge_base_agent, database_query_agent, network_search_agent],
            checkpointer=InMemorySaver(),
        )
        state = graph.invoke({"messages": [HumanMessage(content=question)]},
                             config={"configurable": {"thread_id": "session-A"},
                                     "recursion_limit": 32})
        return state["messages"]

    @staticmethod
    def delegated(messages):
        return [m for m in messages if isinstance(m, ToolMessage) and m.name == "task"]

    def test_mixed_task_calls_three_real_registered_tools_with_separate_sources(self):
        kb = self.storage.create_knowledge_base("618 复盘")
        retrospective = ingest_document(
            kb.knowledge_base_id,
            self.upload("618复盘.md", "活动流量上涨，但结算页转化率下降，销售未同步增长。"),
            storage=self.storage, loader=self.loader,
        )
        sop = ingest_document(
            kb.knowledge_base_id,
            self.upload("转化排查SOP.md", "转化下降时先核对结算流程和商品页信息。"),
            storage=self.storage, loader=self.loader,
        )
        adapter = LocalEmbeddingAdapter(model_factory=lambda _: FakeModel())
        LocalVectorStore(self.storage).build_index(
            kb.knowledge_base_id,
            adapter.embed_documents(retrospective.chunks + sop.chunks), adapter.metadata)
        retriever = LocalRAGRetriever(self.storage, adapter)
        connector, queries = fake_database()
        question = "结合历史 618 复盘、内部销售数据和公开竞品活动，分析转化问题。"
        description = build_knowledge_agent_task(kb.knowledge_base_id,
                                                  "查询历史 618 复盘的流量与转化。",
                                                  storage=self.storage)
        responses = [
            task_call(database_query_agent["name"], "核对现有教学库统计。", "task-d"),
            tool_call("list_sql_tables", {}, "tables-d"),
            tool_call("get_table_data", {"table_name": "drugs"}, "sample-d"),
            tool_call("execute_sql_query", {"query": SQL}, "sql-d"),
            AIMessage(content="数据库是药品教学数据：drugs=50；不能代表美妆销售。"),
            task_call(knowledge_base_agent["name"], description, "task-k"),
            tool_call("search_local_knowledge_base",
                      {"knowledge_base_id": kb.knowledge_base_id, "query": "618 流量与转化",
                       "top_k": 2}, "search-k"),
            AIMessage(content="历史复盘与转化排查资料提供两条独立证据。[C1][C2]"),
            task_call(network_search_agent["name"], "查询公开竞品促销。", "task-n"),
            tool_call("internet_search", {"query": "beauty promotion", "topic": "general",
                                          "max_results": 3, "include_raw_content": False}, "web-n"),
            AIMessage(content=f"模拟公开来源：{PUBLIC_URL}"),
            AIMessage(content=(
                "内部经营数据：药品教学库 drugs=50，不能代表美妆。\n"
                "历史内部资料：两条内部文档证据。[C1][C2]\n"
                f"外部市场：仅有模拟公开来源 {PUBLIC_URL}。\n"
                "综合原因：缺少可比的美妆经营数据，不能定论。\n"
                "运营建议：先补齐同口径销售与转化数据。"
            )),
        ]
        token = set_selected_knowledge_base_context(kb.knowledge_base_id)
        try:
            with (patch.object(local_knowledge_base_tool, "_get_retriever", return_value=retriever),
                  patch.object(retriever, "retrieve", wraps=retriever.retrieve) as retrieve,
                  patch.object(local_knowledge_base_tool, "format_evidence_for_context",
                               wraps=local_knowledge_base_tool.format_evidence_for_context) as formatted,
                  patch.object(db_tools, "get_db_config", return_value={
                      "database": "deepsearch_db", "user": "deepsearch_ro"}),
                  patch.object(db_tools, "connect", connector),
                  patch.object(tavily_tool.tavily_client, "search", return_value={
                      "results": [{"title": "模拟公开促销观察", "url": PUBLIC_URL,
                                   "content": "离线测试摘要"}]}) as network,
                  patch.object(db_tools.monitor, "report_tool"),
                  patch.object(tavily_tool.monitor, "report_tool")):
                messages = self.run_graph(responses, question)
                retrieve.assert_called_once()
                self.assertEqual(connector.call_count, 3)
                network.assert_called_once()
                evidence = formatted.call_args.args[0]
                self.assertEqual([item.citation.citation_id for item in evidence],
                                 ["C1", "C2"])
                self.assertEqual({item.citation.document_name for item in evidence},
                                 {"618复盘.md", "转化排查SOP.md"})
        finally:
            reset_selected_knowledge_base_context(token)
        results = self.delegated(messages)
        self.assertEqual(len(results), 3)
        self.assertNotIn("[C1]", results[0].content + results[2].content)
        self.assertNotIn("[C2]", results[0].content + results[2].content)
        self.assertIn("[C1]", results[1].content)
        self.assertIn("[C2]", results[1].content)
        self.assertIn(PUBLIC_URL, results[2].content)
        self.assertEqual([sql for sql, _ in queries], [
            "SHOW TABLES", db_tools._TABLE_EXISTS_QUERY,
            "SELECT * FROM `drugs` LIMIT 100", SQL])
        self.assertEqual(messages[-1].content.count("[C1]"), 1)
        self.assertEqual(messages[-1].content.count("[C2]"), 1)
        self.assertIn("不能代表美妆", messages[-1].content)

    def test_no_selected_kb_does_not_block_other_agents(self):
        responses = [
            task_call(database_query_agent["name"], "核对数据库", "task-d"),
            AIMessage(content="数据库没有可比的美妆指标。"),
            task_call(network_search_agent["name"], "查询公开信息", "task-n"),
            AIMessage(content="公开信息待核实。"),
            AIMessage(content="未选企业知识库；现有信息不足以分析历史内部资料。"),
        ]
        token = set_selected_knowledge_base_context(None)
        try:
            with patch.object(local_knowledge_base_tool, "_get_retriever") as knowledge:
                messages = self.run_graph(responses, "结合内部复盘、数据库与公开信息。")
                knowledge.assert_not_called()
        finally:
            reset_selected_knowledge_base_context(token)
        self.assertEqual(len(self.delegated(messages)), 2)
        self.assertIn("未选企业知识库", messages[-1].content)

    def test_no_evidence_does_not_block_other_sources(self):
        kb = self.storage.create_knowledge_base("空知识库")
        question = "查询复盘并结合其他来源。"
        responses = [
            task_call(knowledge_base_agent["name"],
                      build_knowledge_agent_task(kb.knowledge_base_id, question,
                                                 storage=self.storage), "task-k"),
            tool_call("search_local_knowledge_base",
                      {"knowledge_base_id": kb.knowledge_base_id, "query": question,
                       "top_k": 1}, "search-k"),
            AIMessage(content="当前知识库没有足够证据。"),
            task_call(database_query_agent["name"], "核对结构化来源", "task-d"),
            AIMessage(content="数据库来源仍可单独使用。"),
            task_call(network_search_agent["name"], "核对公开来源", "task-n"),
            AIMessage(content="公开来源仍可单独使用。"),
            AIMessage(content="内部文档证据不足；其他来源不能补造内部事实。"),
        ]
        token = set_selected_knowledge_base_context(kb.knowledge_base_id)
        try:
            retriever = MagicMock()
            retriever.retrieve.return_value = []
            with patch.object(local_knowledge_base_tool, "_get_retriever", return_value=retriever):
                messages = self.run_graph(responses, question)
                retriever.retrieve.assert_called_once()
        finally:
            reset_selected_knowledge_base_context(token)
        self.assertEqual(len(self.delegated(messages)), 3)
        self.assertNotIn("[C1]", messages[-1].content)

    def test_database_error_is_visible_and_main_can_continue(self):
        responses = [
            task_call(database_query_agent["name"], "查询数据库", "task-d"),
            tool_call("list_sql_tables", {}, "tables-d"),
            AIMessage(content="数据库连接失败，内部经营数据无法核实。"),
            task_call(network_search_agent["name"], "查询公开信息", "task-n"),
            AIMessage(content="公开来源不能代替内部经营数据。"),
            AIMessage(content="数据库来源不可用；不能据公开信息猜测内部销售表现。"),
        ]
        with (patch.object(db_tools, "get_db_config", return_value={
                "database": "deepsearch_db", "user": "deepsearch_ro"}),
              patch.object(db_tools, "connect", side_effect=MySQLError("offline failure")) as connect,
              patch.object(db_tools.monitor, "report_tool"),
              patch.object(local_knowledge_base_tool, "_get_retriever") as knowledge):
            messages = self.run_graph(responses, "核对内部经营数据与公开信息。")
            connect.assert_called_once()
            knowledge.assert_not_called()
        self.assertEqual(len(self.delegated(messages)), 2)
        self.assertIn("数据库来源不可用", messages[-1].content)
        self.assertNotIn("[C1]", messages[-1].content)
