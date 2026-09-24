"""Instruction contracts only; real-model compliance is assessed separately."""
from pathlib import Path
from unittest import TestCase
import yaml

PROMPT = yaml.safe_load((Path(__file__).resolve().parents[1] / 'app/prompt/prompts.yml').read_text(encoding='utf-8'))['main_agent']['system_prompt']
RULE = PROMPT.split('【事实字段保真】')[1].split('你的工作流程通常涉及：')[0]


class FactFidelityContractTests(TestCase):
    def test_a_internal_category_cannot_be_replaced_by_external_category_without_mapping(self):
        for text in ('外部分类体系不得覆盖内部分类体系', '没有可靠证据明确说明有效业务映射时，保留内部原分类',
                     '不能先断言另一分类再用“缺少证据”作补充说明', '映射尚未证实'):
            self.assertIn(text, RULE)

    def test_b_explicit_applicable_mapping_allows_corresponding_category(self):
        self.assertIn('存在明确、可靠且适用于该对象的映射证据时，允许使用对应分类', RULE)
        self.assertIn('同时说明内部原分类、映射依据及适用范围', RULE)

    def test_c_original_values_and_nonsemantic_paraphrase_are_allowed(self):
        self.assertIn('可以原样引用字段值，也可以润色句式', RULE)
        self.assertIn('必须保留源字段的原始业务语义', RULE)
        for field in ('商品分类', '状态', '品牌', 'SKU 类型', '仓库', '地区', '用户等级', '订单状态', '业务标签'):
            self.assertIn(field, RULE)
        self.assertNotRegex(RULE, r'洁面乳|面霜|个护|护肤|\d')

    def test_d_external_background_remains_allowed_without_overwriting_facts(self):
        self.assertIn('公开趋势仍可作为背景或条件性分析依据', RULE)
        self.assertIn('不得据此改写任何内部事实字段', RULE)
        self.assertIn('可以省略与具体对象的趋势映射', RULE)
