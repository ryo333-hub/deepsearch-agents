r"""Opt-in integration: real MySQL + local E5/RAG; scripted LLM + fake Tavily.

PowerShell (from repository root):
  $env:RUN_LOCAL_SERVICE_INTEGRATION = '1'
  .\.venv\Scripts\python.exe -B -m unittest discover -s tests/integration -v
  Remove-Item Env:RUN_LOCAL_SERVICE_INTEGRATION

This directory deliberately has no __init__.py: ordinary unittest discovery
(-s tests) keeps its offline scope. The explicit environment gate also protects
other test collectors. No connector, cursor, SQL result, tokenizer or embedding
is mocked. Only LLM planning, Tavily transport, monitor collection and temporary
storage roots are replaced. All fixture data is removed at process completion.
The network-failure case is an error response, not a transport-exception test.
"""

import asyncio
from collections import Counter
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.api.context import reset_thread_context, set_thread_context
from app.local_rag.embeddings import LocalEmbeddingAdapter
from app.local_rag.ingestion import ingest_document
from app.local_rag.loaders import UploadedDocumentLoader
from app.local_rag.retrieval import LocalRAGRetriever
from app.local_rag.storage import LocalRAGStorage
from app.local_rag.vector_store import LocalVectorStore
from app.tools import db_tools, local_knowledge_base_tool, tavily_tool


ROOT = Path(__file__).resolve().parents[2]
COUNTS_SQL = (
    "SELECT (SELECT COUNT(*) FROM drugs) AS drugs, "
    "(SELECT COUNT(*) FROM inventory) AS inventory, "
    "(SELECT COUNT(*) FROM sales_records) AS sales_records;"
)
BUSINESS_SQL = (
    "SELECT (SELECT COUNT(*) FROM drugs) AS drug_count, "
    "(SELECT SUM(quantity_on_hand) FROM inventory) AS stock_units, "
    "(SELECT SUM(quantity_sold) FROM sales_records) AS sold_units, "
    "(SELECT SUM(total_amount) FROM sales_records) AS sales_amount, "
    "(SELECT MIN(sale_date) FROM sales_records) AS sales_from, "
    "(SELECT MAX(sale_date) FROM sales_records) AS sales_to;"
)
QUESTION = "结合当前药品库存和销售统计、内部经营策略文档及一条模拟公开市场信息，汇总经营情况并明确每项来源。"
DOCS = {
    "库存经营SOP.md": "演示内部制度：药品库存管理应结合批次有效期和近期销量评估补货，优先处理近效期库存；不能仅凭总库存认定滞销。",
    "历史经营复盘.md": "演示历史复盘：药品经营分析应统一销售统计时间范围，结合库存和动销情况判断，不把市场促销消息直接当成内部销售变动原因。",
}
PUBLIC = {
    "results": [{"title": "模拟药品零售市场观察", "url": "https://example.org/simulated-pharmacy-market",
                 "content": "模拟信息：部分零售渠道开展会员促销；不能据此推断本企业销售变化。", "score": 0.9}],
}
FAILURE = {"error": "simulated_network_unavailable", "results": []}
REPORTS = []


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def docker_state():
    cli = shutil.which("docker") or str(Path(os.environ["LOCALAPPDATA"]) /
        "Programs/DockerDesktop/resources/bin/docker.exe")
    result = subprocess.run([cli, "inspect", "--format",
        '{{json .State}}|{{json .NetworkSettings.Ports}}|{{.Id}}', "deepsearch-mysql"],
        check=True, capture_output=True, text=True, timeout=20)
    state, ports, container_id = result.stdout.strip().split("|")
    state, ports = json.loads(state), json.loads(ports)
    if not state["Running"] or state.get("Health", {}).get("Status") != "healthy":
        raise AssertionError("Existing deepsearch-mysql must be running/healthy")
    if not any(p["HostPort"] == "3307" for p in ports.get("3306/tcp", [])):
        raise AssertionError("Expected host 3307 -> container 3306")
    return {"id": container_id, "started_at": state["StartedAt"],
            "health": "healthy", "ports": ports}


class ScriptedBusinessModel(BaseChatModel):
    """Deterministic decisions; worker and Main answers consume real messages.

    Provider identity only selects the existing disabled-general-purpose profile.
    This object never instantiates or calls a hosted model client.
    """

    model_name: str = "deepseek-flash"
    knowledge_base_id: str
    _trace: list = PrivateAttr(default_factory=list)
    _evidence: list = PrivateAttr(default_factory=list)
    _db_results: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-local-integration"

    def _get_ls_params(self, **kwargs):
        return {"ls_provider": "openai", "ls_model_name": self.model_name,
                "ls_model_type": "chat"}

    def bind_tools(self, tools, **kwargs):
        names = {t.name if hasattr(t, "name") else t["function"]["name"] for t in tools}
        role = ("main" if "task" in names else "database" if "execute_sql_query" in names
                else "knowledge" if "search_local_knowledge_base" in names else "network")
        return self.bind(integration_role=role)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        role = kwargs["integration_role"]
        results = [m for m in messages if isinstance(m, ToolMessage)]
        n = len(results)
        if len(self._trace) >= 24:
            raise AssertionError("Unexpected loop")

        def call(name, args):
            return AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": f"{role}-{n}"}])

        if role == "main":
            route = [
                ("数据库查询助手", "只读核对药品表结构、样例、库存总量和销售聚合，保留统计时间范围。"),
                ("企业知识助手", json.dumps({"knowledge_base_id": self.knowledge_base_id,
                    "query": "药品库存经营策略与历史经营复盘"}, ensure_ascii=False)),
                ("网络搜索助手", "查询一条药品零售市场信息，保留来源；本次为模拟公开信息。"),
            ]
            if n < len(route):
                name, description = route[n]
                answer = call("task", {"subagent_type": name, "description": description})
            else:
                if n != 3:
                    raise AssertionError("Unexpected delegation count")
                answer = AIMessage(content="\n\n".join(str(m.content) for m in results) +
                    "\n\n综合判断与建议：库存为当前快照，销售为上述历史区间，不能直接比较为同比或断言下降原因。"
                    "按内部文档建议先核对近效期批次和动销，再评估补货；模拟市场信息不作为真实经营因果证据。")
        elif role == "database":
            steps = [("list_sql_tables", {}),
                     *[("execute_sql_query", {"query": f"DESCRIBE {table};"})
                       for table in ("drugs", "inventory", "sales_records")],
                     ("get_table_data", {"table_name": "drugs"}),
                     ("execute_sql_query", {"query": BUSINESS_SQL})]
            if n < len(steps):
                answer = call(*steps[n])
            else:
                if n != len(steps):
                    raise AssertionError("Unexpected database steps")
                self._db_results[:] = results
                row = rows(str(results[-1].content))[0]
                answer = AIMessage(content=(
                    "内部经营数据（真实 MySQL / deepsearch_db）：\n"
                    f"药品 {row['drug_count']} 种，库存总量 {row['stock_units']}，"
                    f"销售数量 {row['sold_units']}，销售额 {row['sales_amount']}。"
                    f"销售时间：{row['sales_from']} 至 {row['sales_to']}。\n"
                    f"查询依据：{BUSINESS_SQL}"))
        elif role == "knowledge":
            if not results:
                answer = call("search_local_knowledge_base", {
                    "knowledge_base_id": self.knowledge_base_id,
                    "query": "药品库存经营策略与历史经营复盘", "top_k": 2})
            else:
                if n != 1:
                    raise AssertionError("Repeated knowledge lookup")
                self._evidence[:] = results[0].artifact or []
                answer = AIMessage(content="内部文档（临时演示知识库，真实本地检索）：\n" + str(results[0].content))
        else:
            if not results:
                answer = call("internet_search", {"query": "药品零售市场会员促销",
                    "topic": "general", "max_results": 3, "include_raw_content": False})
            else:
                if n != 1:
                    raise AssertionError("Repeated network lookup")
                payload = json.loads(str(results[0].content))
                if payload.get("error"):
                    answer = AIMessage(content="公开信息来源当前不可用（模拟网络错误），未取得公开市场证据，不补造市场数据。")
                else:
                    item = payload["results"][0]
                    answer = AIMessage(content=f"外部市场（模拟 Network source）：{item['title']}\n{item['content']}\n来源：{item['url']}")
        self._trace.append({"role": role, "tool_calls": answer.tool_calls, "content": answer.content})
        return ChatResult(generations=[ChatGeneration(message=answer)])


@unittest.skipUnless(os.getenv("RUN_LOCAL_SERVICE_INTEGRATION") == "1",
                     "integration: explicit opt-in, existing MySQL and local E5 required")
class BusinessMultiAgentIntegrationTests(unittest.TestCase):
    integration = True

    @classmethod
    def setUpClass(cls):
        cls.adapter = LocalEmbeddingAdapter()

    def run_case(self, network_failure):
        before = docker_state()
        cfg = db_tools.get_db_config()
        self.assertEqual((cfg["host"], cfg["port"], cfg["database"], cfg["user"]),
                         ("localhost", 3307, "deepsearch_db", "deepsearch_ro"))
        # Real tool calls: no connector/config/cursor patching anywhere.
        identity = rows(db_tools.execute_sql_query.invoke({
            "query": "SELECT DATABASE() AS db, CURRENT_USER() AS account;"}))[0]
        self.assertEqual(identity["db"], "deepsearch_db")
        self.assertTrue(identity["account"].startswith("deepsearch_ro@"))
        counts_before = rows(db_tools.execute_sql_query.invoke({"query": COUNTS_SQL}))[0]
        tmp_root = ROOT / ".tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root, prefix="business-integration-") as temp:
            base = Path(temp).resolve()
            session = "business-integration-failure" if network_failure else "business-integration-success"
            upload = base / "updated" / f"session_{session}"
            upload.mkdir(parents=True)
            storage = LocalRAGStorage(base / "store")
            token = set_thread_context(session)
            try:
                kb = storage.create_knowledge_base("药品经营演示知识库")
                chunks = []
                for name, text in DOCS.items():
                    (upload / name).write_text(text, encoding="utf-8")
                    ingested = ingest_document(kb.knowledge_base_id, name, storage=storage,
                        loader=UploadedDocumentLoader(base / "updated"))
                    chunks.extend(ingested.chunks)
                self.assertEqual(len(chunks), 2)
                LocalVectorStore(storage).build_index(kb.knowledge_base_id,
                    self.adapter.embed_documents(chunks), self.adapter.metadata)
            finally:
                reset_thread_context(token)

            scripted = ScriptedBusinessModel(knowledge_base_id=kb.knowledge_base_id)
            retriever = LocalRAGRetriever(storage, self.adapter)
            # Execute the actual Main module's assembly with only its LLM replaced.
            # The three registered workers, tool implementations, prompts, saver,
            # run_deep_agent and context/KB authorization all remain project code.
            with (patch("app.agent.llm.model", scripted),
                  patch("app.local_rag.storage.LocalRAGStorage", return_value=storage),
                  patch.object(local_knowledge_base_tool, "_get_retriever", return_value=retriever),
                  patch.object(tavily_tool.tavily_client, "search",
                               return_value=FAILURE if network_failure else PUBLIC) as tavily,
                  patch("requests.sessions.Session.request", side_effect=AssertionError("External HTTP prohibited")) as requests,
                  patch("httpx.Client.send", side_effect=AssertionError("External HTTP prohibited")) as httpx,
                  patch("httpx.AsyncClient.send", side_effect=AssertionError("External HTTP prohibited")) as async_httpx):
                namespace = runpy.run_path(str(ROOT / "app/agent/main_agent.py"), run_name="integration_main")
                # Only session output/upload root is redirected to the fixture.
                namespace["run_deep_agent"].__globals__["project_root_path"] = base
                graph = namespace["main_agent"]
                registry = graph.nodes["tools"].bound.tools_by_name
                self.assertIn("task", registry)
                task_description = registry["task"].description
                agents = task_description.split("Available agent types")[1].split("When using the Task tool")[0]
                self.assertNotIn("general-purpose", agents)
                for name in ("数据库查询助手", "企业知识助手", "网络搜索助手"):
                    self.assertIn(name, agents)
                self.assertNotIn("execute_sql_query", registry)
                monitor = namespace["monitor"]
                with patch.object(monitor, "_emit") as events:
                    asyncio.run(namespace["run_deep_agent"](QUESTION, session, kb.knowledge_base_id))
                errors = [c.args for c in events.call_args_list if c.args[0] == "error"]
                self.assertEqual(errors, [])
                completed = [c.args[2]["result"] for c in events.call_args_list if c.args[0] == "task_result"]
                self.assertEqual(len(completed), 1)
                final = completed[0]
                tavily.assert_called_once_with(query="药品零售市场会员促销", topic="general",
                    max_results=3, include_raw_content=False)
                requests.assert_not_called()
                httpx.assert_not_called()
                async_httpx.assert_not_called()

            trace = scripted._trace
            calls = [c for step in trace for c in step["tool_calls"]]
            self.assertEqual([c["args"]["subagent_type"] for c in calls if c["name"] == "task"],
                             ["数据库查询助手", "企业知识助手", "网络搜索助手"])
            self.assertEqual(Counter(c["name"] for c in calls), {
                "task": 3, "list_sql_tables": 1, "execute_sql_query": 4,
                "get_table_data": 1, "search_local_knowledge_base": 1, "internet_search": 1})
            db_results = scripted._db_results
            for table in ("drugs", "inventory", "sales_records"):
                self.assertIn(table, db_results[0].content)
            for result in db_results[1:4]:
                self.assertIn("Field", rows(result.content)[0])
            self.assertEqual(len(rows(db_results[4].content)), min(int(counts_before["drugs"]), 100))
            aggregate = rows(db_results[-1].content)[0]
            self.assertEqual(aggregate["drug_count"], counts_before["drugs"])
            evidence = scripted._evidence
            self.assertEqual({e.citation.document_name for e in evidence}, set(DOCS))
            self.assertEqual([e.citation.citation_id for e in evidence], ["C1", "C2"])
            for e in evidence:
                self.assertEqual(e.text, DOCS[e.citation.document_name])
                self.assertIn(e.citation.chunk_id, {chunk.chunk_id for chunk in chunks})
                self.assertIn(e.text, final)
                self.assertIn(f"[{e.citation.citation_id}]", final)
            for role in ("database", "network"):
                worker_answer = [s["content"] for s in trace if s["role"] == role][-1]
                self.assertIn(worker_answer, final)
                self.assertNotRegex(worker_answer, r"\[C\d+\]")
            if network_failure:
                self.assertIn("公开信息来源当前不可用", final)
                self.assertNotIn(PUBLIC["results"][0]["url"], final)
                self.assertNotIn(PUBLIC["results"][0]["content"], final)
            else:
                self.assertIn(PUBLIC["results"][0]["url"], final)
                self.assertIn("模拟 Network source", final)
            self.assertNotIn("DSML", final)
            counts_after = rows(db_tools.execute_sql_query.invoke({"query": COUNTS_SQL}))[0]
            self.assertEqual(counts_before, counts_after)
            after = docker_state()
            self.assertEqual(before, after)
            REPORTS.append({"scenario": "network_error_response" if network_failure else "success",
                "database_identity": identity, "counts": counts_after, "mysql": after,
                "aggregate": aggregate, "calls": calls,
                "scripted_model_steps": dict(Counter(s["role"] for s in trace)),
                "citations": [e.citation.model_dump(mode="json") for e in evidence],
                "network": FAILURE if network_failure else PUBLIC, "final": final,
                "real_llm_requests": 0, "real_tavily_requests": 0})

    def test_three_sources_real_mysql_and_local_rag(self):
        self.run_case(False)

    def test_network_error_preserves_database_and_knowledge(self):
        self.run_case(True)
