"""Offline synthetic ecommerce contract tests. No MySQL or Agent needed."""

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import random
import unittest
from unittest.mock import patch

from scripts.seed_ecommerce_demo import (
    TABLES, feature_evidence, fingerprint, generate_data, line_amount, load_scenario,
    safe_ratio, schema_statements, validate_data,
)


class EcommerceDemoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = load_scenario()
        cls.data = generate_data(cls.scenario)
        cls.features = validate_data(cls.data, cls.scenario)

    def reject(self, mutate, pattern):
        data = deepcopy(self.data)
        mutate(data)
        with self.assertRaisesRegex(ValueError, pattern):
            validate_data(data, self.scenario)

    def test_counts(self):
        self.assertEqual({t:len(self.data[t]) for t in TABLES},
            dict(products=40, orders=1200, order_items=2400, inventory=80, ad_metrics=828))

    def test_reproducible_without_network_or_global_random_state(self):
        previous = random.getstate()
        self.addCleanup(random.setstate, previous)
        random.seed(9)
        state = random.getstate()
        with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')) as connect:
            self.assertEqual(fingerprint(generate_data()),fingerprint(self.data))
            self.assertEqual(state, random.getstate())
            connect.assert_not_called()

    def test_seed_controls_noise_without_breaking_features(self):
        changed = {**self.scenario,'random_seed':20260902}
        other = generate_data(changed)
        self.assertNotEqual(fingerprint(other),fingerprint(self.data))
        validate_data(other,changed)

    def test_duplicate_primary_key_rejected(self):
        self.reject(lambda d:d['products'][1].update(product_id=1),'duplicate primary key')

    def test_orphan_order_rejected(self):
        self.reject(lambda d:d['order_items'][0].update(order_id=99999),'orphan order item')

    def test_orphan_product_rejected(self):
        self.reject(lambda d:d['order_items'][0].update(product_id=99999),'orphan order item')

    def test_inventory_unique(self):
        self.reject(lambda d:d['inventory'][1].update(warehouse=d['inventory'][0]['warehouse']),'duplicate inventory grain')

    def test_order_money_exact_for_every_status(self):
        sums = {}
        for i in self.data['order_items']:
            sums[i['order_id']] = sums.get(i['order_id'],Decimal(0)) + i['quantity']*i['unit_price']-i['discount_amount']
        for o in self.data['orders']:
            self.assertEqual(o['total_amount'],sums[o['order_id']])
        self.reject(lambda d:d['orders'][0].update(total_amount=Decimal('0.01')),'order amount mismatch')

    def test_discount_cannot_exceed_gross(self):
        self.reject(lambda d:d['order_items'][0].update(discount_amount=Decimal('999999')),'invalid discount')

    def test_money_rejects_float(self):
        self.reject(lambda d:d['products'][0].update(sale_price=1.1),'Decimal')

    def test_paid_requires_timestamp(self):
        self.reject(lambda d:d['orders'][2].update(paid_at=None),'paid_at/status mismatch')

    def test_unpaid_and_cancelled_have_no_paid_timestamp(self):
        for o in self.data['orders']:
            if o['order_status'] in ('cancelled','unpaid'):
                self.assertIsNone(o['paid_at'])
        self.reject(lambda d:d['orders'][0].update(paid_at=datetime(2026,6,1,11)),'paid_at/status mismatch')

    def test_invalid_orders_excluded_from_gmv_quantity_and_margin(self):
        valid = {o['order_id'] for o in self.data['orders'] if o['order_status'] in ('paid','completed')}
        before = feature_evidence(self.data,self.scenario)
        altered = deepcopy(self.data)
        for i in altered['order_items']:
            if i['order_id'] not in valid:
                i.update(quantity=100000,unit_price=Decimal('99999'),unit_cost_snapshot=Decimal('0'))
        self.assertEqual(before,feature_evidence(altered,self.scenario))

    def test_clicks_bounded(self):
        self.reject(lambda d:d['ad_metrics'][0].update(clicks=99999999),'invalid ad metrics')

    def test_conversions_bounded(self):
        self.reject(lambda d:d['ad_metrics'][0].update(conversions=99999999),'invalid ad metrics')

    def test_negative_ad_money_rejected(self):
        self.reject(lambda d:d['ad_metrics'][0].update(ad_spend=Decimal('-1')),'invalid ad metrics')

    def test_attribution_only_once_per_order(self):
        self.reject(lambda d:d['attributions'].append(d['attributions'][0].copy()),'duplicate order attribution')

    def test_attribution_is_derived_from_real_generated_sales(self):
        items={i['order_item_id']:i for i in self.data['order_items']}
        orders={o['order_id']:o for o in self.data['orders']}
        for a in self.data['attributions']:
            self.assertEqual(a['revenue'],line_amount(items[a['order_item_id']]))
            self.assertEqual(a['platform'],orders[a['order_id']]['platform'])
            self.assertIn(orders[a['order_id']]['order_status'],('paid','completed'))
        self.assertLess(sum(a['revenue'] for a in self.data['ad_metrics']),
                        sum(o['total_amount'] for o in self.data['orders'] if o['order_status'] in ('paid','completed')))

    def test_false_attribution_amount_rejected(self):
        self.reject(lambda d:d['attributions'][0].update(revenue=Decimal('1')),'attribution revenue mismatch')

    def test_false_ad_amount_rejected(self):
        self.reject(lambda d:d['ad_metrics'][0].update(revenue=Decimal('1')),'ad ledger mismatch')

    def test_a_hot_skus_are_top5_with_low_stock(self):
        for row in self.features['A']['risk_skus']:
            self.assertIn(row['product_id'],self.features['A']['top5'])
            self.assertLess(row['stock'],row['safety'])
            self.assertLess(row['coverage_days'],7)

    def test_b_zero_movement_and_slow_stock(self):
        self.assertEqual(self.features['B'][0]['recent_units'],0)
        self.assertIsNone(self.features['B'][0]['coverage_days'])
        self.assertEqual(self.features['B'][1]['recent_units'],1)
        self.assertTrue(all(r['stock']>=1000 for r in self.features['B']))

    def test_c_distinct_platform_roas(self):
        values=sorted(v['roas'] for v in self.features['C'].values())
        self.assertTrue(all(b-a>Decimal('0.25') for a,b in zip(values,values[1:])))

    def test_d_poor_campaign(self):
        row=self.features['D']
        self.assertGreater(row['spend'],10000)
        self.assertLess(row['roas'],1)
        self.assertLess(row['conversion_rate'],Decimal('0.02'))

    def test_e_high_margin_low_volume(self):
        for row in self.features['E']:
            self.assertGreater(row['margin_rate'],Decimal('0.70'))
            self.assertLess(row['units'],100)

    def test_f_promotion_window_not_causal_claim(self):
        self.assertGreater(self.features['F']['promotion_units'],self.features['F']['baseline_units'])
        self.assertFalse(self.features['F']['causal_claim'])

    def test_g_history_vs_current(self):
        self.assertGreater(self.features['G']['june_units'],2*self.features['G']['august_units'])

    def test_h_category_divergence(self):
        row=self.features['H']
        self.assertGreaterEqual(row['august_gmv'],row['july_gmv'])
        self.assertLess(row['august_category_gmv'],row['july_category_gmv'])

    def test_zero_denominator_is_unknown_not_zero(self):
        self.assertIsNone(safe_ratio(0,0))
        self.assertIsNone(safe_ratio(100,0))
        self.assertEqual(safe_ratio(0,100),Decimal(0))

    def test_fixed_dates_and_grain(self):
        self.assertEqual(min(a['metric_date'].isoformat() for a in self.data['ad_metrics']),'2026-06-01')
        self.assertEqual(max(a['metric_date'].isoformat() for a in self.data['ad_metrics']),'2026-08-31')
        self.assertEqual({i['updated_at'].isoformat() for i in self.data['inventory']},{'2026-09-01T00:00:00'})
        self.assertEqual(len({(a['metric_date'],a['platform'],a['product_id']) for a in self.data['ad_metrics']}),828)

    def test_schema_is_additive_and_password_parameterized(self):
        statements=schema_statements()
        self.assertEqual(len(statements),9)
        self.assertFalse(any(x.upper().startswith(('DROP','TRUNCATE','DELETE','ALTER')) for x in statements))
        self.assertFalse(any('deepsearch_db' in x for x in statements))
        self.assertEqual(sum(x.startswith('CREATE TABLE') for x in statements),5)
        self.assertIn('IDENTIFIED BY %s',statements[-2])
        self.assertIn('GRANT SELECT ON',statements[-1])
        self.assertNotIn('GRANT OPTION',statements[-1])
