"""Phase E harness-only tests. No provider, real DB, embedding or search calls."""
import asyncio
import contextlib
import io
import json
import logging
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent / 'integration'))
from real_llm_harness import (Budget, Limits, AcceptanceBudgetExceeded, ToolTrace,
                              Redactor, CaseState, outcome_dimensions)
from ecommerce_real_llm_runner import Acceptance
from langchain_core.tools import StructuredTool, tool


def call(name, args, ident):
    return {'type': 'tool_call', 'name': name, 'args': args, 'id': ident}


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.trace = ToolTrace(self.events, Budget(), Redactor(['test-api-secret', 'test-db-password']))
        self.actual_queries = []

        @tool
        def execute_sql_query(query: str) -> str:
            """Offline SQL event fixture, never a database connection."""
            self.trace.connection({'user': 'ecommerce_ro', 'database': 'insight_ecommerce_db',
                                   'password': 'test-db-password'})
            self.actual_queries.append(query)
            return 'value\n17\n23'
        self.db = execute_sql_query

    def test_sync_database_complete_event(self):
        with self.trace.install():
            result = self.db.invoke(call(self.db.name, {'query': 'SELECT 17'}, 'db-1'))
        e = self.events[0]
        self.assertEqual(e['output']['content'], result.content)
        self.assertTrue(e['success'])
        for key in ('timestamp', 'agent_name', 'tool_name', 'call_id', 'arguments', 'result_summary'):
            self.assertIn(key, e)
        self.assertEqual(e['sql']['returned_rows'], 2)
        self.assertEqual(e['sql']['database_user'], 'ecommerce_ro')
        self.assertEqual(e['sql']['database_name'], 'insight_ecommerce_db')

    def test_async_entry_sync_structured_tool_not_missed_or_duplicated(self):
        with self.trace.install():
            result = asyncio.run(self.db.ainvoke(call(self.db.name, {'query': 'SELECT 17'}, 'pool-1')))
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]['call_id'], result.tool_call_id)

    def test_native_async_tool_and_url(self):
        @tool
        async def internet_search(query: str) -> dict:
            """Offline async search fixture."""
            return {'results': [{'url': 'https://example.invalid/source'}]}
        with self.trace.install():
            asyncio.run(internet_search.ainvoke(call('internet_search', {'query': 'fixture'}, 'a-1')))
        self.assertEqual(self.events[0]['network']['source_urls'], ['https://example.invalid/source'])
        self.assertEqual(self.events[0]['call_id'], 'a-1')
        self.assertTrue(self.events[0]['success'])

    def test_parallel_delegations_keep_their_own_agent_and_call_ids(self):
        async def delegate(subagent_type: str, description: str) -> str:
            await asyncio.sleep(0)
            result=await self.db.ainvoke(call(self.db.name,{'query':description},subagent_type+'-child'))
            return result.content
        task=StructuredTool.from_function(coroutine=delegate,name='task',description='Offline delegation')
        async def run():
            await asyncio.gather(*(task.ainvoke(call('task',{
                'subagent_type':name,'description':'SELECT 17'},name+'-parent')) for name in ('a','b')))
        with self.trace.install():asyncio.run(run())
        children=[e for e in self.events if e['name']==self.db.name]
        self.assertEqual(len(children),2)
        for e in children:
            self.assertEqual(e['parent_call_id'],e['agent_name']+'-parent')
            self.assertEqual(e['call_id'],e['agent_name']+'-child')

    def test_delegation_child_agent_parent_id_and_result(self):
        async def delegate(subagent_type: str, description: str) -> str:
            result = await self.db.ainvoke(call(self.db.name, {'query': description}, 'child-1'))
            return result.content
        task = StructuredTool.from_function(coroutine=delegate, name='task', description='Offline delegation')
        with self.trace.install():
            asyncio.run(task.ainvoke(call('task', {'subagent_type': '数据库查询助手', 'description': 'SELECT 17'}, 'parent-1')))
        parent, child = self.events
        self.assertEqual(parent['agent_name'], 'Main Agent')
        self.assertEqual(child['agent_name'], '数据库查询助手')
        self.assertEqual(child['parent_call_id'], 'parent-1')
        self.assertEqual(parent['output']['tool_call_id'], 'parent-1')
        self.assertEqual(child['output']['tool_call_id'], 'child-1')

    def test_sync_exception_and_secret_redaction(self):
        @tool
        def list_sql_tables() -> str:
            """Offline failure fixture."""
            raise ValueError('test-api-secret test-db-password')
        with self.trace.install(), self.assertRaises(ValueError):
            list_sql_tables.invoke(call('list_sql_tables', {}, 'bad-1'))
        self.assertFalse(self.events[0]['success'])
        text = json.dumps(self.events)
        self.assertNotIn('test-api-secret', text)
        self.assertNotIn('test-db-password', text)
        self.assertIn('ValueError', self.events[0]['error'])

    def test_async_exception_is_recorded(self):
        @tool
        async def internet_search(query: str) -> str:
            """Offline async failure fixture."""
            raise TimeoutError('fixture')
        with self.trace.install(), self.assertRaises(TimeoutError):
            asyncio.run(internet_search.ainvoke(call('internet_search', {'query': 'x'}, 'bad-2')))
        self.assertEqual(self.events[0]['status'], 'failure')

    def test_returned_guard_refusal_is_not_success(self):
        @tool
        def execute_sql_query(query: str) -> str:
            """Offline refusal fixture."""
            return '拒绝执行：仅允许单条只读 SQL 查询。'
        with self.trace.install():
            execute_sql_query.invoke(call('execute_sql_query', {'query': 'UPDATE demo SET x=1'}, 'refused'))
        self.assertFalse(self.events[0]['success'])
        self.assertIsNone(self.events[0]['sql']['database_user'])
        self.assertIsNone(self.events[0]['sql']['returned_rows'])

    def test_sql_is_not_rewritten_or_paid_at_injected(self):
        query = "SELECT\n order_date\nFROM example_orders;"
        with self.trace.install():
            self.db.invoke(call(self.db.name, {'query': query}, 'sql-raw'))
        self.assertEqual(self.actual_queries, [query])
        self.assertEqual(self.events[0]['sql']['original'], query)
        self.assertNotIn('paid_at', json.dumps(self.events))

    def test_rag_query_kb_citation_artifact(self):
        @tool(response_format='content_and_artifact')
        def search_local_knowledge_base(knowledge_base_id: str, query: str) -> tuple:
            """Offline evidence fixture."""
            return 'fixture [C1]', [{'citation': {'citation_id': 'C1', 'document_name': 'fixture.md'}}]
        with self.trace.install():
            search_local_knowledge_base.invoke(call('search_local_knowledge_base',
                {'knowledge_base_id': 'kb_fixture', 'query': 'fixture'}, 'rag-1'))
        self.assertEqual(self.events[0]['rag']['knowledge_base_id'], 'kb_fixture')
        self.assertEqual(self.events[0]['rag']['citations'][0]['citation_id'], 'C1')

    def test_runtime_object_not_serialized(self):
        runtime = SimpleNamespace(tool_call_id='runtime-id', password='test-db-password')
        e = self.trace.begin(self.db, {'query': 'SELECT 17', 'runtime': runtime}, {})
        self.assertEqual(e['call_id'], 'runtime-id')
        self.assertNotIn('runtime', e['arguments'])

    def test_sync_file_tool_is_blocked_before_body(self):
        body = Mock()
        @tool
        def read_file_content(filename: str) -> str:
            """Offline forbidden file fixture."""
            body()
            return 'never'
        with self.trace.install(), self.assertRaisesRegex(RuntimeError, 'capability blocked'):
            asyncio.run(read_file_content.ainvoke(call('read_file_content', {'filename': 'x.md'}, 'file-1')))
        body.assert_not_called()
        self.assertFalse(self.events[0]['success'])

    def test_tool_budget_blocks_body_and_stays_latched(self):
        self.trace.budget = Budget(Limits(tool_calls=1))
        with self.trace.install():
            self.db.invoke({'query': 'SELECT 17'})
            for _ in range(2):
                with self.assertRaises(AcceptanceBudgetExceeded):
                    self.db.invoke({'query': 'SELECT 23'})
        self.assertEqual(self.actual_queries, ['SELECT 17'])
        self.assertEqual(self.trace.budget.failure['category'], 'acceptance_budget_exceeded')


class IsolationTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'.tmp'
        base.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base,prefix='harness-offline-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def preflight(self, state, **kwargs):
        return state.preflight(db='insight_ecommerce_db', user='ecommerce_ro',
            provider='https://api.deepseek.com', **kwargs)

    def test_database_empty_state(self):
        s=CaseState(self.root,'database');p=self.preflight(s)
        self.assertTrue(p['passed'])
        self.assertIsNone(p['selected_knowledge_base_id'])
        self.assertEqual(p['uploaded_file_count'],0)
        self.assertEqual(p['uploaded_file_names'],[])
        self.assertEqual(p['network'],'not required')

    def test_network_has_no_kb(self):
        s=CaseState(self.root,'network');p=self.preflight(s)
        self.assertIsNone(p['selected_knowledge_base_id'])
        self.assertEqual(p['network'],'enabled')

    def test_knowledge_binds_real_case_owned_manifest(self):
        from app.local_rag.storage import LocalRAGStorage
        from app.api.context import set_thread_context,reset_thread_context
        s=CaseState(self.root,'knowledge');store=LocalRAGStorage(s.store_root)
        token=set_thread_context(s.session)
        try:
            s.kb_id=store.create_knowledge_base('fixture').knowledge_base_id
            p=self.preflight(s,validate_kb=store.get_knowledge_base)
            self.assertTrue(p['passed'])
            self.assertEqual(p['selected_knowledge_base_id'],s.kb_id)
        finally:reset_thread_context(token)

    def test_staging_documents_not_uploads(self):
        s=CaseState(self.root,'knowledge')
        s.ingestion_dir.mkdir(parents=True)
        (s.ingestion_dir/'fixture.md').write_text('fixture',encoding='utf-8')
        self.assertEqual(list(s.upload_dir.iterdir()),[])
        self.assertFalse(s.ingestion_dir.is_relative_to(s.runtime_root))

    def test_new_case_does_not_inherit_state(self):
        a=CaseState(self.root,'knowledge')
        (a.upload_dir/'old.md').write_text('old',encoding='utf-8')
        a.kb_id='kb_old';a.trace.append({'old':True});a.messages.append('old message')
        b=CaseState(self.root,'database')
        self.assertNotEqual(a.session,b.session)
        self.assertNotEqual(a.base,b.base)
        self.assertIsNone(b.kb_id)
        self.assertEqual(b.trace,[]);self.assertEqual(b.messages,[])
        self.assertEqual(list(b.upload_dir.iterdir()),[])

    def test_preflight_rejects_attachment_pollution(self):
        s=CaseState(self.root,'database');(s.upload_dir/'unexpected.md').write_text('x')
        self.assertFalse(self.preflight(s)['passed'])

    def test_preflight_rejects_unexpected_kb(self):
        s=CaseState(self.root,'database');s.kb_id='kb_old'
        self.assertFalse(self.preflight(s)['passed'])

    def test_preflight_rejects_prior_trace_or_messages(self):
        s=CaseState(self.root,'database');s.trace.append({});s.messages.append('old')
        self.assertFalse(self.preflight(s)['passed'])

    def test_preflight_rejects_wrong_target_account_provider(self):
        s=CaseState(self.root,'database')
        for kwargs in ({'db':'other','user':'ecommerce_ro','provider':'https://api.deepseek.com'},
                       {'db':'insight_ecommerce_db','user':'root','provider':'https://api.deepseek.com'},
                       {'db':'insight_ecommerce_db','user':'ecommerce_ro','provider':'https://example.invalid'}):
            self.assertFalse(s.preflight(**kwargs)['passed'])

    def test_invalid_case_kb_reports_preflight_failure(self):
        s=CaseState(self.root,'knowledge');s.kb_id='kb_invalid'
        p=self.preflight(s,validate_kb=Mock(side_effect=ValueError('private path')))
        self.assertFalse(p['passed'])
        self.assertNotIn('private path',json.dumps(p))

    def test_runner_preflight_fails_before_graph_or_services(self):
        import ecommerce_real_llm_runner as runner
        a=Acceptance.__new__(Acceptance)
        a.report={'cases':[]};a.redactor=Redactor();a.write_report=Mock()
        a.state=CaseState(self.root,'database');a.storage=Mock()
        a.model=SimpleNamespace(root_client=SimpleNamespace(base_url='https://api.deepseek.com'))
        a.setup_case=Mock()
        (a.state.upload_dir/'unexpected.md').write_text('fixture')
        with patch('app.tools.db_tools.get_db_config',return_value={
            'database':'insight_ecommerce_db','user':'ecommerce_ro'}), \
             patch.object(runner.runpy,'run_path') as graph, \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError,'harness_preflight_failed'):
            a.run_case('database')
        graph.assert_not_called()
        self.assertEqual(a.current['llm'],[])
        self.assertEqual(a.current['connect_count'],0)

    def test_report_append_preserves_historical_observations(self):
        import ecommerce_real_llm_runner as runner
        from threading import RLock
        historical={'name':'database','llm':[{'http_status':200,'usage':{'total_tokens':123}}],
                    'tools':[{'name':'task','input':{'subagent_type':'企业知识助手'}}],
                    'runtime':{'incomplete_trace':True},'observed_sql':'SELECT order_date FROM fixture',
                    'manual_review':{'harness_limitations':['attachment pollution']}}
        a=Acceptance.__new__(Acceptance)
        a.report_lock=RLock();a.redactor=Redactor(['fixture-secret'])
        a.report={'cases':[historical,{'name':'database','llm':[],'tools':[]}],'external_http':[]}
        path=self.root/'report.json'
        with patch.object(runner,'REPORT_PATH',path):a.write_report()
        saved=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(saved['cases'][0],historical)
        self.assertEqual(len(saved['cases']),2)

    def test_complete_runner_case_cleans_context_and_uses_no_kb(self):
        # Harness control-flow fixture only: no acceptance result or business claim.
        from app.api import context
        a=Acceptance.__new__(Acceptance)
        a.state=CaseState(self.root,'database');a.base=a.state.runtime_root
        a.retriever=None;a.storage=Mock();a.setup_case=Mock()
        a.report={'cases':[]};a.redactor=Redactor();a.write_report=Mock()
        a.verify_local_database=Mock()
        a.model=SimpleNamespace(root_client=SimpleNamespace(base_url='https://api.deepseek.com'))
        observed=[];monitor=SimpleNamespace(_emit=Mock())
        async def entry(question,session,kb):
            observed.append((question,session,kb,context.get_thread_context(),
                             context.get_selected_knowledge_base_context()))
            monitor._emit('task_result','fixture',{'result':'offline fixture'})
        task=SimpleNamespace(description='Available agent types 数据库查询助手 企业知识助手 网络搜索助手 When using the Task tool')
        graph=SimpleNamespace(nodes={'tools':SimpleNamespace(bound=SimpleNamespace(tools_by_name={'task':task}))},
                              get_state=Mock(return_value=SimpleNamespace(values={})))
        ns={'main_agent':graph,'run_deep_agent':entry,'monitor':monitor}
        token=context.set_thread_context('previous-session')
        kbtoken=context.set_selected_knowledge_base_context('kb_previous')
        try:
            with patch('ecommerce_real_llm_runner.runpy.run_path',return_value=ns), \
                 patch('app.tools.db_tools.get_db_config',return_value={
                     'database':'insight_ecommerce_db','user':'ecommerce_ro'}), \
                 contextlib.redirect_stderr(io.StringIO()):
                case=a.run_case('database')
            self.assertEqual(observed[0],('2026 年 8 月 GMV 是多少？',a.state.session,None,None,None))
            self.assertEqual(case['preflight']['uploaded_file_count'],0)
            self.assertTrue(case['preflight']['fresh_checkpointer'])
            self.assertEqual(context.get_thread_context(),'previous-session')
            self.assertEqual(context.get_selected_knowledge_base_context(),'kb_previous')
        finally:
            context.reset_thread_context(token)
            context.reset_selected_knowledge_base_context(kbtoken)

    def test_formal_entry_receives_empty_uploads_and_fresh_history(self):
        # Fake model only tests harness plumbing; never an acceptance result.
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage
        from app.agent import llm
        from app.api.context import get_selected_knowledge_base_context,get_thread_context
        class OfflineModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):return self
        graphs=[];inputs=[]
        for _ in range(2):
            s=CaseState(self.root,'database')
            fake=OfflineModel(responses=[AIMessage(content='offline fixture reply')])
            with patch.object(llm,'model',fake), contextlib.redirect_stdout(io.StringIO()):
                ns=runpy.run_path(str(Path(__file__).resolve().parents[1]/'app/agent/main_agent.py'))
                ns['run_deep_agent'].__globals__['project_root_path']=s.runtime_root
                graph=ns['main_agent'];graphs.append(graph)
                cfg={'configurable':{'thread_id':s.session}}
                self.assertFalse(graph.get_state(cfg).values)
                asyncio.run(ns['run_deep_agent']('offline fixture question',s.session,None))
                history=graph.get_state(cfg).values['messages']
                inputs.append(history[0].content)
                self.assertEqual(len(history),2)
                self.assertNotIn('[已上传文件]',history[0].content)
                self.assertIn('未选择企业知识库',history[0].content)
                self.assertEqual(len(graph.nodes['tools'].bound.tools_by_name['task'].description.split('数据库查询助手'))>1,True)
        self.assertIsNot(graphs[0].checkpointer,graphs[1].checkpointer)
        self.assertEqual(inputs[0],inputs[1])


class ProtectionTests(unittest.TestCase):
    def test_async_http_budget_counts_exactly_and_blocks_subsequent_attempts(self):
        import httpx
        a=Acceptance.__new__(Acceptance)
        a.current={'name':'database','preflight':{'passed':True},'llm':[]}
        a.report={'external_http':[]};a.state=SimpleNamespace(session='offline')
        a.write_report=Mock();a.budget=Budget(Limits(llm_requests=2));a.redactor=Redactor()
        a.tracer=ToolTrace([],a.budget)
        sent=[]
        async def transport(client,request,**kwargs):
            sent.append(request)
            return httpx.Response(200,json={'choices':[{'message':{'content':'offline'}}]},request=request)
        async def run():
            async with httpx.AsyncClient() as client:
                req=httpx.Request('POST','https://api.deepseek.com/chat/completions',json={})
                for _ in range(2):await client.send(req)
                for _ in range(2):
                    with self.assertRaises(AcceptanceBudgetExceeded):await client.send(req)
        with patch('httpx.AsyncClient.send',new=transport),a.instrumentation():asyncio.run(run())
        self.assertEqual(len(sent),2)
        self.assertEqual(len(a.current['llm']),2)
        self.assertEqual(a.budget.failure['category'],'acceptance_budget_exceeded')

    def test_deadline_prevents_database_connection_boundary(self):
        tick=[0];budget=Budget(Limits(seconds=1),clock=lambda:tick[0])
        trace=ToolTrace([],budget);tick[0]=1
        with self.assertRaises(AcceptanceBudgetExceeded):
            trace.connection({'user':'ecommerce_ro','database':'insight_ecommerce_db'})

    def test_budget_cancels_active_case_once_and_latches(self):
        async def run():
            b=Budget(Limits(llm_requests=0))
            loop=asyncio.get_running_loop();current=asyncio.current_task()
            b.on_exceeded=lambda:loop.call_soon_threadsafe(current.cancel)
            try:
                try:b.consume('llm_requests')
                except AcceptanceBudgetExceeded:pass
                await asyncio.sleep(0.01)
                self.fail('active case should have been cancelled')
            except asyncio.CancelledError:
                self.assertEqual(b.failure['category'],'acceptance_budget_exceeded')
        asyncio.run(run())

    def test_llm_budget_and_wall_clock(self):
        b=Budget(Limits(llm_requests=1));b.consume('llm_requests')
        with self.assertRaises(AcceptanceBudgetExceeded):b.consume('llm_requests')
        tick=[0];b=Budget(Limits(seconds=5),clock=lambda:tick[0]);tick[0]=5
        with self.assertRaises(AcceptanceBudgetExceeded):b.check()

    def test_budget_cancellation_notification_once(self):
        b=Budget(Limits(tool_calls=0));cancel=Mock();b.on_exceeded=cancel
        for _ in range(2):
            with self.assertRaises(AcceptanceBudgetExceeded):b.consume('tool_calls')
        cancel.assert_called_once()

    def test_api_key_and_database_password_not_logged(self):
        stream=io.StringIO();handler=logging.StreamHandler(stream)
        logger=logging.getLogger('phase_e_offline_secrets');logger.addHandler(handler)
        previous=logger.level;logger.setLevel(logging.ERROR)
        try:
            with Redactor(['key-fixture','password-fixture']).logs():
                try:raise ValueError('password-fixture')
                except ValueError:logger.exception('API %s','key-fixture')
        finally:logger.removeHandler(handler);logger.setLevel(previous)
        self.assertNotIn('key-fixture',stream.getvalue())
        self.assertNotIn('password-fixture',stream.getvalue())
        self.assertIn('[REDACTED]',stream.getvalue())

    def test_acceptance_json_and_error_fields_redacted(self):
        r=Redactor(['secret-fixture'])
        value=r({'api_key':'unlisted','password':'unlisted','args':'secret-fixture',
                 'exception':'Bearer abc.def','nested':{'Authorization':'Bearer hidden'}})
        text=json.dumps(value)
        for forbidden in ('secret-fixture','unlisted','abc.def','Bearer hidden'):
            self.assertNotIn(forbidden,text)

    def test_historical_failure_remains_unknown_where_trace_incomplete(self):
        case={'name':'database','tools':[],'errors':['blocked'],'final':'',
              'manual_review':{'harness_limitations':['incomplete'],
                               'database_worker':{'metric_contract_issue':'order_date'}}}
        d=outcome_dimensions(case)
        self.assertFalse(d['harness_valid'])
        self.assertFalse(d['business_semantics_pass'])
        self.assertIsNone(d['tool_execution_pass'])
        self.assertFalse(d['final_answer_pass'])

    def test_fake_http_preflight_blocks_transport(self):
        import httpx
        a=Acceptance.__new__(Acceptance)
        a.current={'preflight':{'passed':False}}
        a.budget=Budget();a.redactor=Redactor()
        a.tracer=ToolTrace([],a.budget)
        reached=Mock()
        with patch('httpx.Client.send',reached):
            with a.instrumentation(),self.assertRaisesRegex(RuntimeError,'preflight'):
                httpx.Client().send(httpx.Request('POST','https://api.deepseek.com/chat/completions',json={}))
        reached.assert_not_called()

    def test_transport_budget_stops_before_http(self):
        import httpx
        a=Acceptance.__new__(Acceptance)
        a.current={'preflight':{'passed':True}}
        a.budget=Budget(Limits(llm_requests=0));a.redactor=Redactor()
        a.tracer=ToolTrace([],a.budget)
        reached=Mock()
        with patch('httpx.Client.send',reached):
            with a.instrumentation(),self.assertRaises(AcceptanceBudgetExceeded):
                httpx.Client().send(httpx.Request('POST','https://api.deepseek.com/chat/completions',json={}))
        reached.assert_not_called()


if __name__=='__main__':unittest.main()
