"""Phase C opt-in tests. The full acceptance runs once in an isolated child.

PowerShell:
  $env:RUN_ECOMMERCE_KNOWLEDGE_INTEGRATION = '1'
  python -B -m unittest discover -s tests/integration -p test_ecommerce_knowledge_integration.py -v

No database environment mutation leaks into legacy tests or the application.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.getenv('RUN_ECOMMERCE_KNOWLEDGE_INTEGRATION')=='1',
                     'integration: requires ecommerce MySQL and local E5 snapshot')
class EcommerceKnowledgeIntegrationTests(unittest.TestCase):
    integration=True

    @classmethod
    def setUpClass(cls):
        before={k:v for k,v in os.environ.items() if k.startswith('MYSQL_')}
        completed=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('ecommerce_knowledge_runner.py'))],
            cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8',timeout=300)
        if completed.returncode:
            raise AssertionError('Isolated knowledge acceptance failed:\n'+completed.stderr[-16000:])
        cls.report=json.loads(completed.stdout)
        if {k:v for k,v in os.environ.items() if k.startswith('MYSQL_')}!=before:
            raise AssertionError('Database configuration leaked to parent')

    def test_five_real_e5_searches_and_citations(self):
        r=self.report
        self.assertEqual(r['document_count'],4)
        self.assertEqual(r['model'],'intfloat/multilingual-e5-small')
        self.assertEqual(r['dimension'],384)
        self.assertTrue(r['documents_verified_against_live_june_sql'])
        self.assertEqual(len(r['searches']),5)
        for result in r['searches']:
            self.assertEqual(result['top_k'],5)
            self.assertTrue(result['matched_citations'])
            for citation in result['matched_citations']:
                self.assertEqual(citation['document_name'],result['expected_document'])
                self.assertIsNotNone(citation['score'])
                self.assertTrue(citation['heading'])
                self.assertGreaterEqual(citation['start_line'],1)
        self.assertTrue(r['temporary_fixture_cleaned'])

    def test_a_database_and_inventory_rule(self):
        case=self.report['cases']['A']
        self.assertEqual(case['tool_counts']['task'],2)
        self.assertEqual(case['tool_counts']['search_local_knowledge_base'],1)
        for pid in ('1','2'):
            row=next(r for r in case['db_rows'] if r['product_id']==pid)
            self.assertEqual(row['august_units'],'720')
            self.assertEqual(row['stock'],'20')
        self.assertIn('SKU 1、2 存在高缺货风险',case['final'])
        self.assertIn('02_库存经营SOP.md',case['final'])

    def test_b_zero_movement_and_backlog(self):
        case=self.report['cases']['B']
        self.assertIn('SKU 5 属于零动销库存',case['final'])
        self.assertIn('SKU 6 属于库存积压关注',case['final'])
        self.assertIn('覆盖天数不可计算',case['final'])
        self.assertIn('02_库存经营SOP.md',case['final'])

    def test_c_historical_and_current_time_scope(self):
        final=self.report['cases']['C']['final']
        self.assertIn('六月月报记录 600 件；八月数据库查询为 40 件',final)
        self.assertIn('历史判断和当前表现存在时间口径差异',final)
        self.assertIn('01_经营分析月报.md',final)
        self.assertNotIn('月报是错的',final)

    def test_missing_kb_retains_database_without_invented_rules(self):
        case=self.report['cases']['missing_kb']
        self.assertEqual(case['tool_counts'].get('search_local_knowledge_base',0),0)
        self.assertEqual(case['tool_counts']['task'],1)
        self.assertTrue(case['db_rows'])
        self.assertIn('内部知识库未提供可用依据',case['final'])
        self.assertNotRegex(case['final'],r'\[C\d+\]')
        self.assertEqual(self.report['external_http_requests'],0)
        self.assertEqual(self.report['network_agent_calls'],0)
        self.assertEqual(self.report['real_llm_requests'],0)
