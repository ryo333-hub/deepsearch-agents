"""Phase D explicit child-process acceptance: real MySQL/E5/Tavily.

Only model decisions are scripted. Failure and empty-response cases inject at
the Tavily SDK boundary and are labelled as such. No live response is replayed
as an Agent tool result. The three exploratory requests precede one fresh
Network worker request. Artifacts describe one run, not a fixed web fixture.
"""
import asyncio
from collections import Counter
import contextlib
from decimal import Decimal
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
import tempfile
from unittest.mock import patch
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from ecommerce_knowledge_runner import cited, parse_rows, select_evidence
from ecommerce_network_sources import QUERIES, format_network, normalize_sources, now, public_target
from scripts.build_ecommerce_knowledge_docs import DOC_DIR, FILENAMES
from scripts.seed_ecommerce_demo import load_scenario, require
from scripts.verify_ecommerce_demo import acceptance_queries

REPORT_PATH = ROOT / 'demo/ecommerce/network_acceptance_results.json'
DB_KEYS = ('august_gmv', 'top5_sku', 'A_inventory_risk', 'B_inventory_backlog',
           'H_category_divergence', 'C_platform_roas', 'G_historical_sku')
RAG_QUERIES = (
    '公司库存 SOP 如何判断缺货风险、超过90天的积压和最近30天零动销？',
    'SKU 3 在 6 月表现如何？',
    '历史上哪个渠道广告效率存在问题？',
)


def compose_final(db, knowledge, network):
    """Harness-only deterministic synthesis of actual worker results."""
    lines = ['当前企业经营数据（MySQL；2026 年 8 月，库存快照 2026-09-01）：']
    for key in DB_KEYS:
        lines.append(key + '：' + json.dumps(db[key], ensure_ascii=False))
    lines.append('\n企业内部规则 / 历史经验（Local RAG）：')
    # Citation IDs are scoped to each retrieval, never silently renumbered.
    for i, group in enumerate(knowledge, 1):
        lines.append(f'内部检索 K{i}：{group["query"]}')
        for evidence in group['evidence']:
            lines.append(cited(evidence) + '\n' + evidence['text'])
    lines += ['\n外部公开信息（Network，非本企业经营事实）：', format_network(network)]
    stock_evidence = knowledge[0]['evidence']
    risk = select_evidence(stock_evidence, '库存覆盖天数 <')
    backlog = select_evidence(stock_evidence, '库存覆盖天数 >')
    zero = select_evidence(stock_evidence, '最近 30 天销量为 0')
    risk_days = Decimal(re.search(r'库存覆盖天数 < (\d+) 天', risk['text']).group(1))
    backlog_days = Decimal(re.search(r'库存覆盖天数 > (\d+) 天', backlog['text']).group(1))
    risky = [r['product_id'] for r in db['A_inventory_risk'] if Decimal(r['stock']) < Decimal(r['safety'])
             or r['coverage_days'] != 'None' and Decimal(r['coverage_days']) < risk_days]
    slow = [r['product_id'] for r in db['B_inventory_backlog'] if r['coverage_days'] != 'None'
            and Decimal(r['coverage_days']) > backlog_days]
    zero_ids = [r['product_id'] for r in db['B_inventory_backlog'] if Decimal(r['recent_units']) == 0]
    category = {r['month']: r for r in db['H_category_divergence']}
    previous, current = category['2026-07'], category['2026-08']
    weak_ad = min(db['C_platform_roas'], key=lambda r: Decimal(r['roas']))
    lines += ['\n三个优先经营问题与建议：',
        '1. 缺货风险：SKU '+ '、'.join(risky)+f' 满足企业 SOP 风险条件。建议先核对在途与补货周期。依据：MySQL A_inventory_risk；内部检索 K1 {cited(risk)}。',
        '2. 库存结构：SKU '+ '、'.join(zero_ids)+' 为零动销，覆盖天数不可计算；SKU '+ '、'.join(slow)+
        f' 为积压关注。先核查上架、需求与仓间分布，再评估减补货或受控促销。依据：MySQL B_inventory_backlog；内部检索 K1 {cited(zero)}；{cited(backlog)}。',
        f"3. 品类与投放需分开复盘：护肤 GMV 七月 {previous['category_gmv']}、八月 {current['category_gmv']}；同期整体 GMV {previous['gmv']} → {current['gmv']}。"
        f"八月最低汇总 ROAS 为 {weak_ad['platform']} {weak_ad['roas']}。建议按品类、素材和归因口径核查，不能用整体增长掩盖局部问题。依据：MySQL H_category_divergence / C_platform_roas。"]
    historical = select_evidence(knowledge[1]['evidence'], 'SKU 3 在 2026 年 6 月的有效销量')
    june = re.search(r'6 月的有效销量为 (\d+) 件', historical['text']).group(1)
    august = next(r['units'] for r in db['G_historical_sku'] if r['month'] == '2026-08')
    lines += ['\n来源差异与限制：',
        f'六月月报 SKU 3 销量 {june} 件（内部检索 K2 {cited(historical)}）；八月数据库 {august} 件。历史判断和当前表现存在时间口径差异，不代表月报错误。',
        '公开资料描述其他企业或整个市场，内部数据库描述虚构 Demo 企业；主体、时间和指标口径不同。即使外部增长也不能覆盖本企业下降，不据此推断因果。',
        '网络片段仅作背景，不据此给内部 SKU 编造增长预测；发布日期未知或非同期资料不作为八月背景证据。']
    return '\n'.join(lines)


class ScriptedThreeSourceModel(BaseChatModel):
    model_name: str = 'deepseek-flash'
    kb_id: str
    question: str
    network_query: str
    _trace: list = PrivateAttr(default_factory=list)
    _db: dict = PrivateAttr(default_factory=dict)
    _knowledge: list = PrivateAttr(default_factory=list)
    _network: dict = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self):
        return 'scripted-phase-d'

    def _get_ls_params(self, **kwargs):
        return {'ls_provider': 'openai', 'ls_model_name': self.model_name, 'ls_model_type': 'chat'}

    def bind_tools(self, tools, **kwargs):
        names = {t.name if hasattr(t, 'name') else t['function']['name'] for t in tools}
        role = ('main' if 'task' in names else 'db' if 'execute_sql_query' in names else
                'knowledge' if 'search_local_knowledge_base' in names else
                'network' if 'internet_search' in names else 'forbidden')
        return self.bind(integration_role=role)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        role = kwargs['integration_role']
        results = [m for m in messages if isinstance(m, ToolMessage)]
        n = len(results)
        require(len(self._trace) < 40, 'Unexpected model loop')

        def call(name, args):
            return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': f'{role}-{n}'}])

        if role == 'main':
            tasks = [('数据库查询助手', '只读查询八月 GMV、核心 SKU、九月一日库存、品类分化、八月广告和 SKU 3 月度历史；保留时间口径。'),
                     ('企业知识助手', json.dumps({'knowledge_base_id': self.kb_id,
                         'query': '查询库存 SOP、六月 SKU 3 月报和六月广告复盘，保留各次检索 Citation。'}, ensure_ascii=False)),
                     ('网络搜索助手', self.network_query + '；仅公开背景，保留真实 URL 与已提供日期；不补造同期证据。')]
            if n < len(tasks):
                name, description = tasks[n]
                answer = call('task', {'subagent_type': name, 'description': description})
            else:
                require(n == 3, 'Repeated delegation')
                worker = [json.loads(m.content) for m in results]
                require([x['source'] for x in worker] == ['mysql', 'local_rag', 'network'], 'Source contract mismatch')
                answer = AIMessage(content=compose_final(worker[0]['rows'], worker[1]['groups'], worker[2]['result']))
        elif role == 'db':
            sql = acceptance_queries(load_scenario())
            prefix = [('list_sql_tables', {}), ('get_table_data', {'table_name': 'products'}),
                      *[('execute_sql_query', {'query': f'DESCRIBE {t};'}) for t in ('orders','order_items','inventory','ad_metrics')]]
            steps = prefix + [('execute_sql_query', {'query': sql[key]}) for key in DB_KEYS]
            if n < len(steps):
                answer = call(*steps[n])
            else:
                require(n == len(steps), 'Repeated SQL')
                for key, result in zip(DB_KEYS, results[len(prefix):]):
                    rows = parse_rows(result.content)
                    require(bool(rows), 'Missing actual SQL rows: '+key)
                    self._db[key] = rows
                answer = AIMessage(content=json.dumps({'source': 'mysql', 'rows': self._db}, ensure_ascii=False))
        elif role == 'knowledge':
            if n < len(RAG_QUERIES):
                answer = call('search_local_knowledge_base', {'knowledge_base_id': self.kb_id,
                    'query': RAG_QUERIES[n], 'top_k': 5})
            else:
                require(n == len(RAG_QUERIES), 'Repeated RAG')
                self._knowledge[:] = [{'query': q, 'evidence': [e.model_dump(mode='json') for e in m.artifact]}
                                     for q, m in zip(RAG_QUERIES, results)]
                answer = AIMessage(content=json.dumps({'source': 'local_rag', 'groups': self._knowledge}, ensure_ascii=False))
        elif role == 'network':
            if n == 0:
                answer = call('internet_search', {'query': self.network_query, 'topic': 'general',
                    'max_results': 5, 'include_raw_content': False})
            else:
                require(n == 1, 'Repeated search after sufficient result/error')
                self._network.update(normalize_sources(json.loads(results[0].content)))
                answer = AIMessage(content=json.dumps({'source': 'network', 'result': self._network}, ensure_ascii=False))
        else:
            raise AssertionError('Unexpected worker')
        self._trace.append({'role': role, 'tool_calls': answer.tool_calls})
        return ChatResult(generations=[ChatGeneration(message=answer)])


def run_acceptance():
    require(os.getenv('RUN_NETWORK_INTEGRATION') == '1', 'Explicit integration opt-in required')
    creds = json.loads((ROOT/'.data/ecommerce/ecommerce_ro.json').read_text(encoding='utf-8'))
    require((creds['host'], creds['port'], creds['database'], creds['user']) ==
            ('localhost', 3307, 'insight_ecommerce_db', 'ecommerce_ro'), 'Unexpected DB target')
    os.environ.update({'MYSQL_'+k.upper(): str(v) for k, v in creds.items()})
    os.environ.update({'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                       'LANGSMITH_TRACING':'false', 'LANGCHAIN_TRACING_V2':'false'})
    from app.tools import db_tools, local_knowledge_base_tool, tavily_tool
    from app.api.context import reset_thread_context, set_thread_context
    from app.local_rag.embeddings import LocalEmbeddingAdapter
    from app.local_rag.ingestion import ingest_document
    from app.local_rag.loaders import UploadedDocumentLoader
    from app.local_rag.retrieval import LocalRAGRetriever
    from app.local_rag.storage import LocalRAGStorage
    from app.local_rag.vector_store import LocalVectorStore
    from test_business_multi_agent_integration import docker_state
    import requests
    from tavily.errors import TimeoutError as TavilyTimeoutError

    require(bool(os.getenv('TAVILY_API_KEY')), 'Tavily Key not configured')
    require(tavily_tool.tavily_client.base_url == 'https://api.tavily.com', 'Unexpected Tavily endpoint')
    state_before = docker_state()
    identity = parse_rows(db_tools.execute_sql_query.invoke({'query': 'SELECT DATABASE() AS db,CURRENT_USER() AS account;'}))[0]
    require(identity == {'db':'insight_ecommerce_db', 'account':'ecommerce_ro@%'}, 'Wrong runtime DB identity')
    counts_sql = acceptance_queries(load_scenario())['counts']
    counts_before = db_tools.execute_sql_query.invoke({'query': counts_sql})
    report = {'snapshot_notice': 'Single live acceptance, not a replay fixture. Model decisions are scripted.',
              'started_at': now(), 'identity': identity, 'discovery': [], 'cases': {}, 'http': []}
    # Optional continuation after a *local harness* preflight interruption. It
    # reuses only the already-recorded exploratory queries, never Agent results.
    # Ordinary opt-in test runs make all four fresh calls.
    if os.getenv('PHASE_D_RESUME_DISCOVERY') == '1':
        previous=json.loads(REPORT_PATH.read_text(encoding='utf-8'))
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(previous['started_at'])).total_seconds()
        require(0 <= age < 86400 and not previous['cases'] and not previous['success'], 'Only current interrupted preflight may resume')
        require([q['query'] for q in previous['discovery']]==QUERIES, 'Exploratory query mismatch')
        require(len(previous['http'])==3 and all(h['status']==200 and h['kind']=='tavily' for h in previous['http']), 'Unverified exploratory requests')
        report['http']=previous['http']
        report['started_at']=previous['started_at']
        report['initial_preflight_stop']={'reason':'Local source-URL DNS check rejected known proxy fake-IP range before GET',
            'page_checks':[s['page_check'] for q in previous['discovery'] for s in q['sources'] if s.get('page_check')]}
        for q in previous['discovery']:
            report['discovery'].append({'query':q['query'],'queried_at':q['queried_at'],
                'returned_count':q['returned_count'],**normalize_sources({'results':q['sources']})})
    original_send = requests.sessions.Session.send
    allow_public = set()

    def guarded_send(session, request, **kwargs):
        parsed = urlsplit(request.url)
        is_tavily = request.method == 'POST' and request.url == 'https://api.tavily.com/search'
        require(is_tavily or request.method == 'GET' and request.url in allow_public, 'Unapproved HTTP target')
        if is_tavily:
            require(sum(x['kind']=='tavily' for x in report['http']) < 4, 'Tavily request budget exceeded')
        record = {'kind': 'tavily' if is_tavily else 'public_page', 'method': request.method,
                  'url': request.url, 'requested_at': now()}
        report['http'].append(record)
        try:
            response = original_send(session, request, **kwargs)
            record['status'] = response.status_code
            return response
        except requests.exceptions.RequestException as exc:
            record['error_type'] = type(exc).__name__
            raise

    def check_page(url):
        result = {'url': url, 'checked_at': now(), 'accessible': False}
        try:
            with requests.Session() as client:
                # Never reuse the Tavily session/Authorization on source URLs.
                for _ in range(3):
                    public_target(url)
                    allow_public.add(url)
                    with client.get(url, timeout=15, allow_redirects=False, stream=True) as response:
                        result['status'] = response.status_code
                        if response.status_code in (301,302,303,307,308) and response.headers.get('Location'):
                            url = urljoin(url, response.headers['Location'])
                            continue
                        result['accessible'] = 200 <= response.status_code < 300
                        result['final_url'] = url
                        break
        except (requests.exceptions.RequestException, ValueError, OSError) as exc:
            result['error_type'] = type(exc).__name__
        return result

    with (patch('requests.sessions.Session.send', new=guarded_send),
          patch('httpx.Client.send', side_effect=AssertionError('Hosted model/external httpx prohibited')),
          patch('httpx.AsyncClient.send', side_effect=AssertionError('Hosted model/external httpx prohibited'))):
        try:
            for query in ([] if report['discovery'] else QUERIES):
                payload = tavily_tool.internet_search.invoke({'query':query, 'topic':'general',
                    'max_results':5, 'include_raw_content':False})
                normalized = normalize_sources(payload)
                record = {'query':query, 'queried_at':now(), 'returned_count':len(payload.get('results', [])), **normalized}
                report['discovery'].append(record)
                require(record['status'] != 'unavailable', 'Live Tavily failed; see sanitized acceptance record')
            candidates = [s for r in report['discovery'] for s in r['sources'] if s['selected']]
            require(bool(candidates), 'No minimally qualified public source; no fabricated fallback')
            for source in candidates[:3]:
                source['page_check'] = check_page(source['url'])
                if source['page_check']['accessible']:
                    break
            require(any(s.get('page_check',{}).get('accessible') for s in candidates), 'Could not confirm an accessible source')
            best = max(report['discovery'], key=lambda r: sum(s['selected'] for s in r['sources']))
            question = ('结合 2026 年 8 月经营数据、企业历史经营策略以及本次检索到的美妆/消费品公开背景，'
                        '分析最值得关注的三个经营问题，给出建议并分别标明来源。')
            if not any(r['same_period_evidence'] for r in report['discovery']):
                question += '若未找到足够同期公开证据，请明确说明，不把非同期或日期未知材料用于解释八月表现。'
            report['question'] = question
            report['question_adjustment'] = 'Uses actual exploratory source availability; no presumed market-growth conclusion.'
            with tempfile.TemporaryDirectory(dir=ROOT/'.tmp', prefix='ecommerce-network-') as tmp:
                base = Path(tmp).resolve()
                session = 'ecommerce-network-phase-d'
                upload = base/'updated'/f'session_{session}'
                upload.mkdir(parents=True)
                storage = LocalRAGStorage(base/'store')
                adapter = LocalEmbeddingAdapter()
                parsed, chunks = {}, []
                token = set_thread_context(session)
                try:
                    kb = storage.create_knowledge_base('ecommerce-business-kb')
                    loader = UploadedDocumentLoader(base/'updated')
                    for name in FILENAMES:
                        shutil.copyfile(DOC_DIR/name, upload/name)
                        parsed[name] = loader.load(name)
                        chunks.extend(ingest_document(kb.knowledge_base_id,name,storage=storage,loader=loader).chunks)
                    LocalVectorStore(storage).build_index(kb.knowledge_base_id,adapter.embed_documents(chunks),adapter.metadata)
                finally:
                    reset_thread_context(token)
                retriever = LocalRAGRetriever(storage,adapter)
                report['rag'] = {'knowledge_base_id':kb.knowledge_base_id,'model':adapter.metadata.model_name,
                    'dimension':adapter.metadata.embedding_dimension,'chunks':len(chunks),'temporary_fixture':True}
                for mode in ('live','injected_timeout','injected_empty'):
                    scripted = ScriptedThreeSourceModel(kb_id=kb.knowledge_base_id,question=question,network_query=best['query'])
                    with contextlib.ExitStack() as stack:
                        if mode == 'injected_timeout':
                            # Real SDK search/parser remains; only its HTTP request raises.
                            stack.enter_context(patch.object(tavily_tool.tavily_client.session, 'post',
                                side_effect=requests.exceptions.Timeout('injected transport timeout')))
                        elif mode == 'injected_empty':
                            stack.enter_context(patch.object(tavily_tool.tavily_client,'search',return_value={'results':[]}))
                        stack.enter_context(patch('app.agent.llm.model',scripted))
                        stack.enter_context(patch('app.local_rag.storage.LocalRAGStorage',return_value=storage))
                        stack.enter_context(patch.object(local_knowledge_base_tool,'_get_retriever',return_value=retriever))
                        ns = runpy.run_path(str(ROOT/'app/agent/main_agent.py'),run_name='phase_d_formal_main')
                        ns['run_deep_agent'].__globals__['project_root_path'] = base
                        with patch.object(ns['monitor'],'_emit') as monitor:
                            asyncio.run(ns['run_deep_agent'](question,session,kb.knowledge_base_id))
                        errors = [c.args for c in monitor.call_args_list if c.args[0]=='error']
                        require(not errors, 'Formal Main failed: '+str(errors))
                        finals = [c.args[2]['result'] for c in monitor.call_args_list if c.args[0]=='task_result']
                        require(len(finals)==1,'Missing final report')
                    calls = [call for step in scripted._trace for call in step['tool_calls']]
                    targets = [c['args']['subagent_type'] for c in calls if c['name']=='task']
                    require(targets==['数据库查询助手','企业知识助手','网络搜索助手'],'Incorrect routing')
                    tool_counts = Counter(c['name'] for c in calls)
                    require(tool_counts=={'task':3,'list_sql_tables':1,'get_table_data':1,
                        'execute_sql_query':11,'search_local_knowledge_base':3,'internet_search':1},'Unexpected tools/retries')
                    for group in scripted._knowledge:
                        for e in group['evidence']:
                            c=e['citation']
                            block=parsed[c['document_name']].blocks[c['source_block_index']]
                            require(e['text']==block.text[c['start_char']:c['end_char']],'Invalid citation position')
                            require(c['knowledge_base_id']==kb.knowledge_base_id,'Foreign citation')
                    final = finals[0]
                    db_section, rest = final.split('\n企业内部规则 / 历史经验',1)
                    require(not re.search(r'\[C\d+\]|https?://',db_section),'DB mislabelled as citation/URL')
                    network_section = rest.split('\n外部公开信息',1)[1].split('\n三个优先经营问题',1)[0]
                    require(not re.search(r'\[C\d+\]',network_section),'Web mislabelled as internal citation')
                    require('DSML' not in final,'Nonstandard protocol')
                    if mode=='live':
                        require(scripted._network['status']=='ok','Fresh Network result lacks usable sources')
                        require(any(s['url'] in final for s in scripted._network['sources'] if s['selected']),'Real URL lost')
                    elif mode=='injected_timeout':
                        require('公开信息来源当前不可用' in final,'Failure not preserved')
                    else:
                        require('未找到足够公开资料' in final,'Empty source fabricated')
                    for s in scripted._network.get('sources',[]):
                        s['used_by_main'] = s['selected'] and s['url'] in final
                    report['cases'][mode]={'fault_injection':mode!='live','db_rows':scripted._db,
                        'knowledge':scripted._knowledge,'network':scripted._network,'final':final,
                        'tool_counts':dict(tool_counts),'calls':calls,
                        'scripted_model_steps':dict(Counter(t['role'] for t in scripted._trace))}
                report['temporary_fixture_cleaned'] = True
            require(not base.exists(),'Temporary fixture not cleaned')
            require(db_tools.execute_sql_query.invoke({'query':counts_sql})==counts_before,'Database rows changed')
            state_after = docker_state()
            require(state_before==state_after,'MySQL restarted or state changed')
            report.update({'mysql':state_after,'database_counts_unchanged':True,'real_llm_requests':0,
                'tavily_requests':sum(x['kind']=='tavily' for x in report['http']),
                'external_http_requests':len(report['http']),'finished_at':now(),'success':True})
        finally:
            # This is a sanitized result snapshot, including failures if any.
            report.setdefault('success',False)
            report['finished_at']=now()
            serialized=json.dumps(report,ensure_ascii=False,indent=2)
            for secret in (os.getenv('TAVILY_API_KEY'),creds['password'],os.getenv('OPENAI_API_KEY')):
                require(not secret or secret not in serialized,'Secret found in report; refusing to save')
            REPORT_PATH.write_text(serialized+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        result=run_acceptance()
    print(json.dumps(result,ensure_ascii=True))
