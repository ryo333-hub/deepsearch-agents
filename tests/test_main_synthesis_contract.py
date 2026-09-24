"""Main synthesis instruction contracts; no claim of real-model compliance."""
from pathlib import Path
from unittest import TestCase
import yaml

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / 'app/prompt/prompts.yml').read_text(encoding='utf-8'))
PROMPT = CONFIG['main_agent']['system_prompt']
CONTRACT = PROMPT.split('【综合时的派生指标与证据边界】')[1].split('你的工作流程通常涉及：')[0]


class MainSynthesisContractTests(TestCase):
    def test_a_mismatched_window_requires_exact_followup_or_omission(self):
        """SOP window differs from DB's adjacent/monthly aggregate."""
        for rule in ('精确统计窗口、过滤条件、单位和计算口径',
                     '不得用相邻窗口、整月数据或其他相近数据替代缺失输入',
                     '不得自行近似', '再次委派数据库查询助手补查精确所需数据',
                     '或明确说明证据不足并省略该派生指标', '不得自行编写 SQL'):
            self.assertIn(rule, CONTRACT)

    def test_b_public_category_does_not_reclassify_internal_products(self):
        """DB personal care + Web functional skincare does not prove membership."""
        for rule in ('保留数据库或内部资料已证实的商品分类',
                     '没有数据库、知识库或用户提供的对应证据',
                     '不得据公开趋势重新分类内部商品'):
            self.assertIn(rule, CONTRACT)

    def test_c_external_background_does_not_prove_internal_stockout_cause(self):
        for rule in ('公开趋势只能作为背景证据', '不得自动映射为内部商品事实',
                     '不得断言该 SKU 未来需求上涨、缺货风险扩大、必须增加补货',
                     '把公开趋势当作内部销量变化或缺货的原因',
                     '内部事实（Database）、内部规则（Knowledge）、公开背景（Web）和分析推断',
                     '分析推断必须标明“推断”', '说明依据、适用条件及证据限制'):
            self.assertIn(rule, CONTRACT)

    def test_d_sufficient_consistent_evidence_allows_calculation(self):
        self.assertIn('输入证据充分且口径一致时，可以按已确认的公式正常计算，并说明依据', CONTRACT)
        self.assertNotRegex(CONTRACT, r'\d|https?://|SELECT|商品 [124]')
        self.assertNotIn('固定调用顺序', CONTRACT)
