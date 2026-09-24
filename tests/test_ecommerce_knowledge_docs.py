"""Offline document contracts and independently computed June seed metrics."""

from collections import defaultdict
from decimal import Decimal
import json
import re
import unittest
from unittest.mock import patch

from scripts.build_ecommerce_knowledge_docs import (
    DOC_DIR, FILENAMES, NOTICE, SOURCE_PATH, fmt, ratio, render_documents,
)
from scripts.seed_ecommerce_demo import generate_data, line_amount


class EcommerceKnowledgeDocsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs={name:(DOC_DIR/name).read_text(encoding='utf-8') for name in FILENAMES}
        cls.source=json.loads(SOURCE_PATH.read_text(encoding='utf-8'))
        cls.metrics=cls.source['metrics']
        data=generate_data()
        cls.data=data
        orders={o['order_id']:o for o in data['orders']}
        products={p['product_id']:p for p in data['products']}
        cls.june=[]
        for item in data['order_items']:
            o=orders[item['order_id']]
            if o['paid_at'] and o['paid_at'].month==6:
                cls.june.append((item,o,products[item['product_id']]))

    def test_exactly_four_markdown_files(self):
        self.assertEqual({p.name for p in DOC_DIR.glob('*.md')},set(FILENAMES))

    def test_nonempty_utf8(self):
        for text in self.docs.values():
            self.assertGreater(len(text),400)
            self.assertNotIn('\ufffd',text)

    def test_common_header_and_version(self):
        for text in self.docs.values():
            for field in ('文档名称：','文档类型：','文档版本：v1.0','成文日期：','数据覆盖期间：','适用范围：'):
                self.assertIn(field,text)

    def test_synthetic_notice(self):
        for text in self.docs.values():
            self.assertIn(NOTICE,text)

    def test_fixed_authored_dates(self):
        for name,day in zip(FILENAMES,['2026-07-05','2026-06-01','2026-07-06','2026-08-01']):
            self.assertIn('成文日期：'+day,self.docs[name])

    def test_history_coverage(self):
        for name in (FILENAMES[0],FILENAMES[2]):
            self.assertIn('2026-06-01 ～ 2026-06-30',self.docs[name])
            self.assertNotRegex(self.docs[name],r'8 月|八月|2026-08|2026-09')

    def test_rule_coverage(self):
        self.assertIn('2026-08-02 至 2026-08-31',self.docs[FILENAMES[1]])
        self.assertIn('2026-09-01',self.docs[FILENAMES[1]])
        self.assertIn('2026-08-15 ～ 2026-08-21',self.docs[FILENAMES[3]])

    def test_june_sku3_history_is_real(self):
        units=sum(i['quantity'] for i,_,_ in self.june if i['product_id']==3)
        self.assertEqual(units,600)
        self.assertIn(f'SKU 3 在 2026 年 6 月的有效销量为 {units} 件',self.docs[FILENAMES[0]])
        self.assertIn('6 月属于重点销售商品之一',self.docs[FILENAMES[0]])

    def test_june_total_matches_generated_data(self):
        gmv=sum(line_amount(i) for i,_,_ in self.june)
        units=sum(i['quantity'] for i,_,_ in self.june)
        self.assertEqual(Decimal(self.metrics['total'][0]['gmv']),gmv)
        self.assertEqual(int(self.metrics['total'][0]['units']),units)
        self.assertIn(fmt(gmv)+' 元',self.docs[FILENAMES[0]])

    def test_category_numbers_match_seed(self):
        for row in self.metrics['categories']:
            selected=[i for i,_,p in self.june if p['category']==row['category']]
            self.assertEqual(Decimal(row['gmv']),sum(line_amount(i) for i in selected))
            self.assertEqual(int(row['units']),sum(i['quantity'] for i in selected))
            self.assertIn(f"| {row['category']} | {fmt(row['gmv'])} |",self.docs[FILENAMES[0]])

    def test_sku_ranking_matches_seed(self):
        units=defaultdict(int)
        for i,_,_ in self.june:
            units[i['product_id']]+=i['quantity']
        top=sorted(units,key=lambda pid:(-units[pid],pid))[:5]
        self.assertEqual([r['product_id'] for r in self.metrics['top_skus']],top)

    def test_platform_gmv_matches_seed(self):
        for row in self.metrics['platforms']:
            expected=sum(line_amount(i) for i,o,_ in self.june if o['platform']==row['platform'])
            self.assertEqual(Decimal(row['gmv']),expected)

    def test_ad_numbers_match_seed(self):
        for row in self.metrics['ads']:
            selected=[a for a in self.data['ad_metrics'] if a['metric_date'].month==6 and a['platform']==row['platform']]
            for output,field in [('spend','ad_spend'),('revenue','revenue'),('clicks','clicks'),('impressions','impressions'),('conversions','conversions')]:
                self.assertEqual(Decimal(row[output]),sum(a[field] for a in selected))

    def test_ad_ratios_are_ratio_of_sums(self):
        text=self.docs[FILENAMES[2]]
        for row in self.metrics['ads']:
            for field,denominator,percent in [('revenue','spend',False),('clicks','impressions',True),('conversions','clicks',True)]:
                self.assertIn(ratio(Decimal(row[field]),Decimal(row[denominator]),percent),text)
        self.assertIn('不平均行级比率',text)

    def test_zero_denominator_not_disguised(self):
        self.assertEqual(ratio(10,0),'不可计算（分母为零）')
        self.assertEqual(ratio(0,100,True),'0.0000%')

    def test_inventory_risk_rule(self):
        text=self.docs[FILENAMES[1]]
        self.assertIn('当前库存 < safety_stock',text)
        self.assertIn('库存覆盖天数 < 7 天',text)
        self.assertIn('7～30 天',text)

    def test_inventory_backlog_and_zero_movement(self):
        text=self.docs[FILENAMES[1]]
        self.assertIn('库存覆盖天数 > 90 天',text)
        self.assertIn('最近 30 天销量为 0 且当前库存 > 0',text)
        self.assertIn('NULL',text)

    def test_internal_rules_not_industry_standard(self):
        self.assertIn('不是行业规定',self.docs[FILENAMES[1]])
        self.assertIn('根据企业库存 SOP',self.docs[FILENAMES[1]])

    def test_no_future_inventory_in_june_report(self):
        self.assertIn('未获得六月末库存快照',self.docs[FILENAMES[0]])
        self.assertNotIn('库存各 20',self.docs[FILENAMES[0]])

    def test_promotion_causal_limit(self):
        self.assertIn('不能仅凭前后变化断言因果',self.docs[FILENAMES[3]])
        for factor in ('同期趋势','渠道投放','库存','商品曝光','其他促销因素'):
            self.assertIn(factor,self.docs[FILENAMES[3]])

    def test_promotion_document_does_not_know_future_result(self):
        self.assertNotIn('135',self.docs[FILENAMES[3]])
        self.assertIn('活动前规则',self.docs[FILENAMES[3]])

    def test_no_private_customer_fields_or_credentials(self):
        for text in self.docs.values():
            self.assertNotRegex(text,r'customer_id|demo_customer_|MYSQL_PASSWORD|API_KEY|Authorization|sk-[A-Za-z0-9]|1[3-9]\d{9}')

    def test_no_external_links_or_runtime_dependencies(self):
        for text in self.docs.values():
            self.assertNotRegex(text,r'https?://|!\[|<script|<iframe')

    def test_committed_docs_equal_reproducible_renderer(self):
        self.assertEqual(render_documents(self.metrics),self.docs)

    def test_renderer_is_offline(self):
        with patch('socket.socket.connect',side_effect=AssertionError('No network')) as connect:
            self.assertEqual(render_documents(self.metrics),self.docs)
            connect.assert_not_called()

    def test_source_sql_restricts_june_and_valid_orders(self):
        for name,query in self.source['queries'].items():
            self.assertTrue(query.startswith('SELECT'))
            self.assertIn("'2026-06-01'",query)
            self.assertIn("< '2026-07-01'",query)
            if name!='ads':
                self.assertIn("('paid','completed')",query)
