"""Isolated Phase C harness; real DB/RAG, scripted decisions, no external APIs.

Only executed in an explicitly enabled child process, never imported by normal
offline discovery. Credentials/environment changes cannot affect the parent or
the long-lived application. Temporary KB/index/session files are always removed.
"""

import asyncio
from collections import Counter
import contextlib
import csv
from decimal import Decimal
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from scripts.build_ecommerce_knowledge_docs import (
    DOC_DIR, FILENAMES, query_historical_metrics, render_documents,
)
from scripts.seed_ecommerce_demo import require

QUERIES = [
    ('公司的库存缺货风险是怎么定义的？',FILENAMES[1],'库存覆盖天数 < 7 天'),
    ('SKU 3 在 6 月表现如何？',FILENAMES[0],'SKU 3 在 2026 年 6 月的有效销量'),
    ('库存超过 90 天怎么处理？',FILENAMES[1],'库存覆盖天数 > 90 天'),
    ('历史上哪个渠道广告效率存在问题？',FILENAMES[2],'投放效率偏低'),
    ('促销销量上涨是否能直接证明促销有效？',FILENAMES[3],'不能仅凭前后变化断言因果'),
]
QUESTIONS = {
    'A':'分析 2026 年 8 月销量最高的商品，并根据公司的库存 SOP 判断 9 月 1 日是否存在缺货风险。',
    'B':'找出当前库存积压商品，并根据公司 SOP 给出处理建议。',
    'C':'SKU 3 的当前销售情况和历史经营月报相比发生了什么变化？',
    'missing_kb':'分析 2026 年 8 月销量最高的商品，并根据公司的库存 SOP 判断 9 月 1 日是否存在缺货风险。',
}
STOCK_SQL = """WITH aug AS (
 SELECT i.product_id,SUM(i.quantity) AS august_units FROM order_items i JOIN orders o USING(order_id)
 WHERE o.order_status IN ('paid','completed') AND o.paid_at >= '2026-08-01' AND o.paid_at < '2026-09-01'
 GROUP BY i.product_id
), recent AS (
 SELECT i.product_id,SUM(i.quantity) AS recent_units FROM order_items i JOIN orders o USING(order_id)
 WHERE o.order_status IN ('paid','completed') AND o.paid_at >= '2026-08-02' AND o.paid_at < '2026-09-01'
 GROUP BY i.product_id
), stock AS (
 SELECT product_id,SUM(stock_quantity) AS stock,SUM(safety_stock) AS safety_stock FROM inventory GROUP BY product_id
)
SELECT s.product_id,COALESCE(a.august_units,0) AS august_units,s.stock,s.safety_stock,
 COALESCE(r.recent_units,0) AS recent_units,s.stock*30/NULLIF(r.recent_units,0) AS coverage_days
FROM stock s LEFT JOIN aug a USING(product_id) LEFT JOIN recent r USING(product_id)
"""
SQLS = {
    'A':STOCK_SQL+' ORDER BY august_units DESC,s.product_id LIMIT 5;',
    'B':STOCK_SQL+' WHERE s.stock>=1000 AND COALESCE(r.recent_units,0)<=5 ORDER BY s.product_id;',
    'C':"""SELECT i.product_id,SUM(i.quantity) AS august_units FROM order_items i JOIN orders o USING(order_id)
 WHERE o.order_status IN ('paid','completed') AND o.paid_at >= '2026-08-01' AND o.paid_at < '2026-09-01'
 AND i.product_id=3 GROUP BY i.product_id;""",
}
SQLS['missing_kb']=SQLS['A']
KNOWLEDGE_QUERIES={
    'A':QUERIES[0][0],
    'B':'根据公司库存经营 SOP，库存覆盖超过90天和最近30天零动销库存分别怎么处理？',
    'C':QUERIES[1][0],
}


def parse_rows(content):
    return list(csv.DictReader(io.StringIO(str(content))))


def cited(e):
    c=e['citation']
    return f"[{c['citation_id']}] {c['document_name']}；{c['heading']}；第 {c['start_line']}～{c['end_line']} 行"


def select_evidence(evidence, phrase):
    found=next((e for e in evidence if phrase in e['text']),None)
    require(found is not None,'Required rule/history not retrieved: '+phrase)
    return found


class ScriptedDualModel(BaseChatModel):
    """Responses depend on actual ToolMessages. No hardcoded SQL answers.

    Deterministic rule parsing is harness-only, not production Agent capability.
    The provider identity reuses the project's existing disabled-GP profile.
    """
    model_name:str='deepseek-flash'
    case:str
    kb_id:str|None=None
    _trace:list=PrivateAttr(default_factory=list)
    _evidence:list=PrivateAttr(default_factory=list)
    _rows:list=PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return 'scripted-phase-c'

    def _get_ls_params(self,**kwargs):
        return {'ls_provider':'openai','ls_model_name':self.model_name,'ls_model_type':'chat'}

    def bind_tools(self,tools,**kwargs):
        names={t.name if hasattr(t,'name') else t['function']['name'] for t in tools}
        role='main' if 'task' in names else 'db' if 'execute_sql_query' in names else 'knowledge' if 'search_local_knowledge_base' in names else 'forbidden'
        return self.bind(integration_role=role)

    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        role=kwargs['integration_role']
        results=[m for m in messages if isinstance(m,ToolMessage)]
        n=len(results)
        require(len(self._trace)<24,'Unexpected loop')
        def call(name,args):
            return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':f'{role}-{n}'}])
        if role=='main':
            if n==0:
                answer=call('task',{'subagent_type':'数据库查询助手','description':QUESTIONS[self.case]+' 只查询数据库事实，使用2026年8月销量、2026-09-01库存快照和08-02至08-31日均销量口径。'})
            elif n==1 and self.kb_id is not None:
                answer=call('task',{'subagent_type':'企业知识助手','description':json.dumps({
                    'knowledge_base_id':self.kb_id,'query':KNOWLEDGE_QUERIES[self.case]},ensure_ascii=False)})
            else:
                db=json.loads(results[0].content)
                require(db['source']=='mysql','Missing DB worker result')
                rows=db['rows']
                lines=['当前经营数据（真实 MySQL / insight_ecommerce_db）：',
                       '销售期间：2026 年 8 月；库存快照：2026-09-01；覆盖天数分母：2026-08-02～08-31 日均有效销量。']
                for r in rows:
                    lines.append('SKU '+r['product_id']+'：'+ '；'.join(f'{k}={v}' for k,v in r.items() if k!='product_id'))
                if self.kb_id is None:
                    require(n==1,'Unexpected no-KB task')
                    lines.append('\n内部知识库未提供可用依据。保留上述数据库事实，不能据此编造公司 SOP 或声称已按内部规则判定。')
                else:
                    require(n==2,'Unexpected task count')
                    knowledge=json.loads(results[1].content)
                    require(knowledge['source']=='local_rag','Missing Knowledge worker result')
                    evidence=knowledge['evidence']
                    if self.case=='A':
                        rule=select_evidence(evidence,'库存覆盖天数 <')
                        threshold=int(re.search(r'库存覆盖天数 < (\d+) 天',rule['text']).group(1))
                        risky=[r['product_id'] for r in rows if Decimal(r['stock'])<Decimal(r['safety_stock']) or
                               r['coverage_days']!='None' and Decimal(r['coverage_days'])<threshold]
                        lines += ['\n内部规则（Local RAG）：'+rule['text']+'\n来源：'+cited(rule),
                                  '按该企业 SOP，以上数据库结果中的 SKU '+ '、'.join(risky)+' 存在高缺货风险；建议复核可售库存、在途到货和补货周期。']
                    elif self.case=='B':
                        slow=select_evidence(evidence,'库存覆盖天数 >')
                        zero=select_evidence(evidence,'最近 30 天销量为 0')
                        threshold=int(re.search(r'库存覆盖天数 > (\d+) 天',slow['text']).group(1))
                        lines += ['\n内部规则（Local RAG）：'+zero['text']+'\n来源：'+cited(zero),slow['text']+'\n来源：'+cited(slow)]
                        for r in rows:
                            if Decimal(r['recent_units'])==0 and Decimal(r['stock'])>0:
                                lines.append(f"SKU {r['product_id']} 属于零动销库存，覆盖天数不可计算；按企业 SOP 核对上架、曝光和需求，暂停未经论证的追加补货。")
                            elif r['coverage_days']!='None' and Decimal(r['coverage_days'])>threshold:
                                lines.append(f"SKU {r['product_id']} 属于库存积压关注，按企业 SOP 核查动销、仓间分布和需求，再评估减少补货、调拨或受控促销。")
                    else:
                        historical=select_evidence(evidence,'SKU 3 在 2026 年 6 月的有效销量')
                        june=int(re.search(r'6 月的有效销量为 (\d+) 件',historical['text']).group(1))
                        current=int(rows[0]['august_units'])
                        direction='下降' if current<june else '上升' if current>june else '持平'
                        lines += ['\n历史内部文档（Local RAG）：'+historical['text']+'\n来源：'+cited(historical),
                            f'六月月报记录 {june} 件；八月数据库查询为 {current} 件，月度有效销量{direction}。',
                            '历史文档反映 2026 年 6 月，数据库查询反映 2026 年 8 月。历史判断和当前表现存在时间口径差异，不能据此否定六月月报；月份天数也不同，不能直接当作日均变化或因果解释。']
                answer=AIMessage(content='\n'.join(lines))
        elif role=='db':
            steps=[('list_sql_tables',{}),('get_table_data',{'table_name':'products'}),
                   *[('execute_sql_query',{'query':f'DESCRIBE {t};'}) for t in ('orders','order_items','inventory')],
                   ('execute_sql_query',{'query':SQLS[self.case]})]
            if n<len(steps):
                answer=call(*steps[n])
            else:
                require(n==len(steps),'Unexpected SQL repetition')
                self._rows[:]=parse_rows(results[-1].content)
                require(bool(self._rows) and 'product_id' in self._rows[0],'Invalid database result')
                answer=AIMessage(content=json.dumps({'source':'mysql','rows':self._rows},ensure_ascii=False))
        elif role=='knowledge':
            if not results:
                answer=call('search_local_knowledge_base',{'knowledge_base_id':self.kb_id,
                    'query':KNOWLEDGE_QUERIES[self.case],'top_k':5})
            else:
                require(n==1,'Unexpected retrieval repetition')
                self._evidence[:]=[e.model_dump(mode='json') for e in results[0].artifact]
                answer=AIMessage(content=json.dumps({'source':'local_rag','evidence':self._evidence},ensure_ascii=False))
        else:
            raise AssertionError('Network/other worker prohibited')
        self._trace.append({'role':role,'tool_calls':answer.tool_calls,'content':answer.content})
        return ChatResult(generations=[ChatGeneration(message=answer)])


def run_acceptance():
    # Configure exactly once, in this dedicated process, BEFORE application import.
    creds=json.loads((ROOT/'.data/ecommerce/ecommerce_ro.json').read_text(encoding='utf-8'))
    require((creds['host'],creds['port'],creds['database'],creds['user'])==
            ('localhost',3307,'insight_ecommerce_db','ecommerce_ro'),'Unexpected DB target')
    os.environ.update({'MYSQL_'+k.upper():str(v) for k,v in creds.items()})
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','LANGSMITH_TRACING':'false','LANGCHAIN_TRACING_V2':'false'})
    from app.api.context import set_thread_context,reset_thread_context
    from app.local_rag.embeddings import LocalEmbeddingAdapter
    from app.local_rag.ingestion import ingest_document
    from app.local_rag.loaders import UploadedDocumentLoader
    from app.local_rag.retrieval import LocalRAGRetriever
    from app.local_rag.storage import LocalRAGStorage
    from app.local_rag.vector_store import LocalVectorStore
    from app.tools import db_tools,local_knowledge_base_tool,tavily_tool
    from scripts.verify_ecommerce_demo import acceptance_queries
    from scripts.seed_ecommerce_demo import load_scenario

    with (patch('requests.sessions.Session.request',side_effect=AssertionError('External HTTP prohibited')) as requests,
          patch('httpx.Client.send',side_effect=AssertionError('External HTTP prohibited')) as httpx,
          patch('httpx.AsyncClient.send',side_effect=AssertionError('External HTTP prohibited')) as ahttpx,
          patch.object(tavily_tool.tavily_client,'search',side_effect=AssertionError('Tavily prohibited')) as tavily):
        current_metrics=query_historical_metrics()
        expected_docs=render_documents(current_metrics)
        require(all((DOC_DIR/name).read_text(encoding='utf-8')==text for name,text in expected_docs.items()),'Documents differ from current real June SQL')
        identity=parse_rows(db_tools.execute_sql_query.invoke({'query':'SELECT DATABASE() AS db,CURRENT_USER() AS account;'}))[0]
        require(identity=={'db':'insight_ecommerce_db','account':'ecommerce_ro@%'},'Actual connection is not ecommerce_ro')
        counts_sql=acceptance_queries(load_scenario())['counts']
        counts_before=db_tools.execute_sql_query.invoke({'query':counts_sql})
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp',prefix='ecommerce-knowledge-') as tmp:
            base=Path(tmp).resolve()
            session='ecommerce-knowledge-phase-c'
            upload=base/'updated'/f'session_{session}'
            upload.mkdir(parents=True)
            storage=LocalRAGStorage(base/'store')
            adapter=LocalEmbeddingAdapter()
            token=set_thread_context(session)
            try:
                kb=storage.create_knowledge_base('ecommerce-business-kb')
                chunks=[]
                parsed={}
                loader=UploadedDocumentLoader(base/'updated')
                for name in FILENAMES:
                    shutil.copyfile(DOC_DIR/name,upload/name)
                    parsed[name]=loader.load(name)
                    ingested=ingest_document(kb.knowledge_base_id,name,storage=storage,loader=loader)
                    chunks.extend(ingested.chunks)
                LocalVectorStore(storage).build_index(kb.knowledge_base_id,adapter.embed_documents(chunks),adapter.metadata)
                retriever=LocalRAGRetriever(storage,adapter)
                stored_chunks={c.chunk_id:c for c in chunks}

                def verify_evidence(evidence):
                    for e in evidence:
                        c=e['citation']
                        require(c['knowledge_base_id']==kb.knowledge_base_id,'Foreign KB citation')
                        require(c['chunk_id'] in stored_chunks,'Unknown cited chunk')
                        chunk=stored_chunks[c['chunk_id']]
                        require(c['document_id']==chunk.document_id,'Citation document mismatch')
                        require(c['heading'] and c['start_line'] and c['end_line']>=c['start_line'],'Missing citation location')
                        block=parsed[c['document_name']].blocks[c['source_block_index']]
                        require(e['text']==block.text[c['start_char']:c['end_char']],'Citation offset does not resolve to original content')
                        require(c['score'] is not None,'Missing similarity')

                searches=[]
                with patch.object(local_knowledge_base_tool,'_get_retriever',return_value=retriever):
                    for i,(query,name,phrase) in enumerate(QUERIES):
                        result=local_knowledge_base_tool.search_local_knowledge_base.invoke({
                            'type':'tool_call','id':f'retrieval-{i}','name':'search_local_knowledge_base',
                            'args':{'knowledge_base_id':kb.knowledge_base_id,'query':query,'top_k':5}})
                        evidence=[e.model_dump(mode='json') for e in result.artifact]
                        verify_evidence(evidence)
                        # Report ranking even on a mismatch; never substitute an answer.
                        matches=[e for e in evidence if e['citation']['document_name']==name and phrase in e['text']]
                        require(bool(matches),'Retrieval mismatch: '+query+'; '+json.dumps(evidence,ensure_ascii=False))
                        searches.append({'query':query,'top_k':5,'expected_document':name,
                            'matched_citations':[e['citation'] for e in matches],'evidence':evidence})
            finally:
                reset_thread_context(token)

            cases={}
            for case in QUESTIONS:
                scripted=ScriptedDualModel(case=case,kb_id=None if case=='missing_kb' else kb.knowledge_base_id)
                with (patch('app.agent.llm.model',scripted),
                      patch('app.local_rag.storage.LocalRAGStorage',return_value=storage),
                      patch.object(local_knowledge_base_tool,'_get_retriever',return_value=retriever)):
                    ns=runpy.run_path(str(ROOT/'app/agent/main_agent.py'),run_name='phase_c_formal_main')
                    ns['run_deep_agent'].__globals__['project_root_path']=base
                    registry=ns['main_agent'].nodes['tools'].bound.tools_by_name
                    require('task' in registry and 'search_local_knowledge_base' not in registry,'Main tool isolation changed')
                    agents=registry['task'].description.split('Available agent types')[1].split('When using the Task tool')[0]
                    require('general-purpose' not in agents,'General-purpose unexpectedly enabled')
                    with patch.object(ns['monitor'],'_emit') as monitor:
                        asyncio.run(ns['run_deep_agent'](QUESTIONS[case],session,scripted.kb_id))
                    errors=[c.args for c in monitor.call_args_list if c.args[0]=='error']
                    require(not errors,'Formal Main failed: '+str(errors))
                    finals=[c.args[2]['result'] for c in monitor.call_args_list if c.args[0]=='task_result']
                    require(len(finals)==1,'Missing final answer')
                calls=[c for step in scripted._trace for c in step['tool_calls']]
                targets=[c['args']['subagent_type'] for c in calls if c['name']=='task']
                require(targets==(['数据库查询助手'] if case=='missing_kb' else ['数据库查询助手','企业知识助手']),'Unexpected routing')
                tool_counts=Counter(c['name'] for c in calls)
                require(tool_counts=={'task':len(targets),'list_sql_tables':1,'get_table_data':1,'execute_sql_query':4,
                    **({} if case=='missing_kb' else {'search_local_knowledge_base':1})},'Unexpected tool path')
                final=finals[0]
                if case!='missing_kb':
                    verify_evidence(scripted._evidence)
                    valid_ids={e['citation']['citation_id'] for e in scripted._evidence}
                    referenced=set(re.findall(r'\[(C\d+)\]',final))
                    require(bool(referenced) and referenced<=valid_ids,'Fabricated citation')
                    # Main database facts precede a separately cited internal source.
                    db_section=final.split('\n内部规则')[0] if case!='C' else final.split('\n历史内部文档')[0]
                    require(not re.search(r'\[C\d+\]',db_section),'Database facts mislabeled as RAG citations')
                else:
                    require('内部知识库未提供可用依据' in final and not re.search(r'\[C\d+\]',final),'Invalid no-KB degradation')
                    require(not scripted._evidence,'No-KB case retrieved evidence')
                require('DSML' not in final,'Nonstandard protocol content')
                cases[case]={'question':QUESTIONS[case],'db_rows':scripted._rows,'sql':SQLS[case],
                    'evidence':scripted._evidence,'final':final,'calls':calls,'tool_counts':dict(tool_counts),
                    'scripted_model_steps':dict(Counter(s['role'] for s in scripted._trace))}
            require(db_tools.execute_sql_query.invoke({'query':counts_sql})==counts_before,'Ecommerce counts changed')
            require({r['product_id'] for r in cases['A']['db_rows']} >= {'1','2'},'Missing hot SKUs')
            require({r['product_id'] for r in cases['B']['db_rows']} == {'5','6'},'Missing backlog SKUs')
            require(cases['C']['db_rows'][0]['august_units']=='40','Unexpected current units')
            require('六月月报记录 600 件' in cases['C']['final'],'Historical evidence missing')
            require('时间口径差异' in cases['C']['final'],'Time boundary missing')
            require(all(x.call_count==0 for x in (requests,httpx,ahttpx,tavily)),'External service attempted')
            report={'knowledge_base_id':kb.knowledge_base_id,'kb_name':'ecommerce-business-kb',
                'session_id':session,'temporary_fixture':True,'model':adapter.metadata.model_name,
                'model_revision':adapter.metadata.model_revision,'dimension':adapter.metadata.embedding_dimension,
                'index':'persisted NumPy exact vector index; normalized inner product',
                'chunk_count':len(chunks),'document_count':len(FILENAMES),'identity':identity,
                'documents_verified_against_live_june_sql':True,'searches':searches,'cases':cases,
                'external_http_requests':0,'real_llm_requests':0,'network_agent_calls':0,
                'database_counts_unchanged':True}
        report['temporary_fixture_cleaned']=not base.exists()
    return report


if __name__=='__main__':
    require(os.getenv('RUN_ECOMMERCE_KNOWLEDGE_INTEGRATION')=='1','Explicit integration opt-in required')
    # Keep project prompt/debug prints out of the machine-readable report.
    with contextlib.redirect_stdout(io.StringIO()):
        report=run_acceptance()
    path=ROOT/'demo/ecommerce/knowledge_acceptance_results.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))
