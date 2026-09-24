"""Real-model acceptance, explicitly opt-in; never part of offline discovery.

RUN_LLM_INTEGRATION=1 runs missing real cases through the formal graph.
PHASE_E_VALIDATE_EXISTING=1 validates recorded real traces without more API use.
The latter is not a live rerun and is reported as trace validation.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.getenv('RUN_LLM_INTEGRATION')=='1','integration: real LLM/Tavily requires explicit opt-in')
class EcommerceRealLLMTests(unittest.TestCase):
    integration=True

    @classmethod
    def setUpClass(cls):
        if os.getenv('PHASE_E_VALIDATE_EXISTING')!='1':
            completed=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('ecommerce_real_llm_runner.py')),
                '--stage','all'],cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8',timeout=7200)
            if completed.returncode:
                raise AssertionError('Real acceptance process failed:\n'+completed.stderr[-12000:])
        cls.report=json.loads((ROOT/'demo/ecommerce/real_llm_acceptance_results.json').read_text(encoding='utf-8'))
        if not cls.report['real_model'] or cls.report['scripted_model']:
            raise AssertionError('Real model required')
        cls.cases={case['name']:case for case in cls.report['cases']}

    def check_case(self,name):
        self.assertIn(name,self.cases,'Real case has not run')
        case=self.cases[name]
        self.assertIs(case.get('dimensions',{}).get('harness_valid'),True,
                      'Historical or contaminated harness result is not valid acceptance evidence')
        self.assertNotIn('budget_failure',case,'Acceptance budget was exceeded')
        assessment=case['assessment']
        failed=[k for k in assessment['required_checks'] if not assessment['checks'][k]]
        self.assertFalse(failed,f'{name}: failed observable checks {failed}')
        self.assertTrue(case['llm'],'No real request recorded')
        self.assertTrue(all(c.get('http_status')==200 for c in case['llm']))
        return case

    def test_llm_smoke_and_schema(self):
        case=self.check_case('smoke')
        self.assertIn('task',case['llm'][0]['tools_schema'])

    def test_database_autonomous(self):
        self.check_case('database')

    def test_knowledge_autonomous(self):
        self.check_case('knowledge')

    def test_network_autonomous(self):
        self.check_case('network')

    def test_unnecessary_source_avoidance(self):
        self.check_case('knowledge_selection')

    def test_dual_source_autonomous(self):
        self.check_case('dual')

    def test_write_refused(self):
        self.check_case('write_denial')

    def test_real_bad_sql_recovery(self):
        self.check_case('sql_recovery')

    def test_absent_knowledge(self):
        self.check_case('no_knowledge')

    def test_causal_boundary(self):
        case=self.check_case('causality')
        self.assertRegex(case['final'],r'不能|无法|不足|不等于|相关.*因果')

    def test_network_failure_degradation(self):
        self.check_case('network_failure')

    def test_five_real_core_repetitions(self):
        for i in range(1,6):
            with self.subTest(repetition=i):
                self.check_case(f'core_{i}')

    def test_local_database_integrity(self):
        self.assertTrue(self.report['local_integrity']['unchanged'])
        self.assertEqual(self.report['local_integrity']['counts'],{
            'products':40,'orders':1200,'order_items':2400,'inventory':80,'ad_metrics':828})
        self.assertEqual(self.report['local_integrity']['identity']['account'],'ecommerce_ro@%')
