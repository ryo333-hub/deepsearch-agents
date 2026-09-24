"""Explicit live test: RUN_NETWORK_INTEGRATION=1 (never ordinary offline CI).

One isolated child performs the live acceptance once for all test methods.
Tavily is billable; no fixed URL assertions or automatic live retries.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.getenv('RUN_NETWORK_INTEGRATION')=='1',
                     'integration: explicit real Tavily + local MySQL/E5 opt-in required')
class EcommerceNetworkIntegrationTests(unittest.TestCase):
    integration=True

    @classmethod
    def setUpClass(cls):
        before={k:v for k,v in os.environ.items() if k.startswith('MYSQL_')}
        completed=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('ecommerce_network_runner.py'))],
            cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8',timeout=420)
        if completed.returncode:
            raise AssertionError('Network acceptance failed (no automatic retry):\n'+completed.stderr[-12000:])
        cls.report=json.loads(completed.stdout)
        if before!={k:v for k,v in os.environ.items() if k.startswith('MYSQL_')}:
            raise AssertionError('Child MySQL configuration leaked')

    def test_real_three_query_search(self):
        r=self.report
        self.assertTrue(r['success'])
        self.assertEqual(len(r['discovery']),3)
        self.assertEqual(r['tavily_requests'],4)
        self.assertTrue(any(s.get('page_check',{}).get('accessible') for q in r['discovery'] for s in q['sources']))
        self.assertTrue(all(x['status']==200 for x in r['http'] if x['kind']=='tavily'))

    def test_network_source_contract(self):
        for result in self.report['discovery']+[self.report['cases']['live']['network']]:
            for s in result['sources']:
                self.assertTrue(s['url'].startswith(('https://','http://')))
                self.assertTrue(s['title'] and s['domain'] and s['content'])
                self.assertIn('published_date',s)
                if s['published_date'] is None:
                    self.assertEqual(s['period'],'unknown')
        self.assertTrue(any(s['used_by_main'] for s in self.report['cases']['live']['network']['sources']))

    def test_formal_three_source_graph(self):
        case=self.report['cases']['live']
        self.assertEqual(case['tool_counts']['task'],3)
        self.assertEqual(case['tool_counts']['internet_search'],1)
        self.assertEqual(case['tool_counts']['search_local_knowledge_base'],3)
        self.assertEqual(self.report['identity']['account'],'ecommerce_ro@%')
        self.assertEqual(self.report['rag']['model'],'intfloat/multilingual-e5-small')
        self.assertTrue(case['db_rows']['august_gmv'])
        self.assertEqual(len(case['knowledge']),3)
        self.assertIn('三个优先经营问题',case['final'])

    def test_injected_timeout_degrades(self):
        case=self.report['cases']['injected_timeout']
        self.assertTrue(case['fault_injection'])
        self.assertEqual(case['network']['error']['type'],'TimeoutError')
        self.assertIn('公开信息来源当前不可用',case['final'])
        self.assertEqual(case['tool_counts']['internet_search'],1)
        self.assertTrue(case['db_rows'] and case['knowledge'])

    def test_injected_empty_does_not_invent(self):
        case=self.report['cases']['injected_empty']
        self.assertTrue(case['fault_injection'])
        self.assertEqual(case['network']['sources'],[])
        self.assertIn('未找到足够公开资料',case['final'])
        self.assertNotRegex(case['final'],r'\[N\d+\]')
        self.assertTrue(case['db_rows'] and case['knowledge'])

    def test_source_isolation_and_time_conflict(self):
        for case in self.report['cases'].values():
            final=case['final']
            self.assertIn('主体、时间和指标口径不同',final)
            self.assertIn('六月月报 SKU 3 销量 600 件',final)
            self.assertIn('八月数据库 40 件',final)
            db=final.split('企业内部规则 / 历史经验')[0]
            self.assertNotRegex(db,r'\[C\d+\]|\[N\d+\]|https?://')
            network=final.split('外部公开信息')[1].split('三个优先经营问题')[0]
            self.assertNotRegex(network,r'\[C\d+\]')
            self.assertIn('[C',final)

    def test_no_extra_services_or_persistent_fixture(self):
        self.assertEqual(self.report['real_llm_requests'],0)
        self.assertEqual(self.report['mysql']['health'],'healthy')
        self.assertTrue(self.report['database_counts_unchanged'])
        self.assertTrue(self.report['temporary_fixture_cleaned'])
        for case in self.report['cases'].values():
            self.assertEqual(set(case['tool_counts']),{'task','list_sql_tables','get_table_data',
                'execute_sql_query','search_local_knowledge_base','internet_search'})
