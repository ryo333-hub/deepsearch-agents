"""Offline checks for real-run trace assessment; no API/model construction."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('real_llm_assessment',Path(__file__).parent/'integration/ecommerce_real_llm_runner.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def case(name='smoke',final='连接正常。',tools=None):
    return {'name':name,'final':final,'tools':tools or [],'llm':[{'content':final}],
            'errors':[],'connect_count':0}


class RealLLMAssessmentTests(unittest.TestCase):
    def test_smoke_has_no_business_tools(self):
        self.assertTrue(runner.basic_assessment(case())['automatic_pass'])

    def test_runtime_error_not_success(self):
        value=case();value['errors']=['timeout']
        self.assertFalse(runner.basic_assessment(value)['automatic_pass'])

    def test_unknown_citation_rejected(self):
        self.assertFalse(runner.basic_assessment(case(final='资料 [C9]'))['checks']['citation_ids_known'])

    def test_database_wrong_number_rejected(self):
        value=case('database','GMV 100 元。',[{'name':'task','input':{'subagent_type':'数据库查询助手'}},
            {'name':'execute_sql_query','input':{'query':'SELECT ...'},'output':'gmv\n922395.80'}])
        self.assertFalse(runner.basic_assessment(value)['automatic_pass'])

    def test_database_unneeded_network_rejected(self):
        value=case('database','GMV 922,395.80 元。',[{'name':'task','input':{'subagent_type':'数据库查询助手'}},
            {'name':'task','input':{'subagent_type':'网络搜索助手'}},
            {'name':'execute_sql_query','input':{},'output':'gmv\n922395.80'}])
        self.assertFalse(runner.basic_assessment(value)['checks']['only_database_route'])

    def test_real_url_not_model_invented_url(self):
        value=case('network','来源 https://invented.example.org',[{'name':'task','input':{'subagent_type':'网络搜索助手'}},
            {'name':'internet_search','input':{},'output':{'content':'{"results":[{"url":"https://actual.example.org"}]}'}}])
        self.assertFalse(runner.basic_assessment(value)['checks']['real_url_preserved'])

    def test_write_with_connection_fails_strict_prompt_boundary(self):
        value=case('write_denial','当前仅支持只读查询。');value['connect_count']=1
        self.assertFalse(runner.basic_assessment(value)['automatic_pass'])

    def test_dsml_not_standard_tool_success(self):
        self.assertFalse(runner.basic_assessment(case(final='<DSML> ls'))['checks']['no_dsml'])

    def test_presence_is_not_claimed_as_semantic_validation(self):
        value=runner.basic_assessment(case())
        self.assertIn('numeric_presence_not_semantic_accuracy',value)
        self.assertTrue(value['semantic_review'].startswith('pending'))
