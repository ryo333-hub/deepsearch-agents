"""Offline prompt contracts and scripted traces, NOT real-model compliance.

SQLite fixtures deliberately separate creation/payment months and order states.
The scripted model checks the delivered system prompt, but does not interpret
natural language or predict how DeepSeek will behave. No MySQL is contacted.
"""
import contextlib
import io
import json
import re
import sqlite3
from unittest import TestCase

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

with contextlib.redirect_stdout(io.StringIO()):
    from app.agent.subagents.database_query_agent import database_query_agent

PROMPT = database_query_agent['system_prompt']
# Explicit policy assertions complement the scripted fixtures; script success
# alone would not detect removal of the instructions from the product prompt.
RULES = (
    '业务事件字段', 'GMV、成交额、销量、商品销售表现、历史成交趋势和按月成交分析',
    'order_status 为 paid 或 completed', '排除 cancelled 和 unpaid',
    '按 paid_at 归属业务日期', '不得用 order_date 代替成交日期',
    '仅当用户明确询问下单行为、下单量、订单创建量或订单创建时间',
    '不得默认套用有效成交状态过滤', '折后商品成交金额',
    '不得混用有效成交金额与全状态订单金额',
    '不得因数字更大或查询更简单而选用', '最少数据库探索和 SQL 查询',
    '停止探索并返回答案', '不再浏览无关表或拆分无关维度',
    '仅在问题确实需要商品、库存或广告信息时',
    '允许必要的 Schema 确认、错误修复和验证查询',
)


def require_contract(prompt):
    for rule in RULES:
        if rule not in prompt:
            raise AssertionError('Missing Database prompt contract: ' + rule)


class ContractScript(FakeMessagesListChatModel):
    """Check prompt delivery; scripted actions; final value from tool evidence."""
    answer_column: str = 'value'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        require_contract('\n'.join(str(m.content) for m in messages if isinstance(m, SystemMessage)))
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        msg = result.generations[0].message
        if msg.content == 'ANSWER_FROM_EVIDENCE':
            evidence = next(m for m in reversed(messages)
                            if isinstance(m, ToolMessage) and m.name == 'execute_sql_query')
            rows = json.loads(evidence.content)
            msg.content = json.dumps([row[self.answer_column] for row in rows])
        return result


def action(name, args, number):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': f'fixture-{number}'}])


class DatabasePromptContractTests(TestCase):
    def run_fixture(self, question='本月 GMV 是多少？', *, event='paid_at',
                    valid=True, metric='gmv', previews=(), prompt=PROMPT):
        # Independent invented values, unrelated to the real acceptance answer.
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.addCleanup(connection.close)
        connection.row_factory = sqlite3.Row
        connection.executescript('''
            CREATE TABLE orders(order_id INTEGER, order_date TEXT, paid_at TEXT,
                                order_status TEXT, total_amount INTEGER);
            CREATE TABLE order_items(order_id INTEGER, product_id INTEGER,
                                     quantity INTEGER, unit_price INTEGER, discount_amount INTEGER);
            CREATE TABLE products(product_id INTEGER, product_name TEXT);
            INSERT INTO orders VALUES
              (1,'2031-03-31','2031-04-01','paid',17),
              (2,'2031-04-30','2031-05-01','completed',39),
              (3,'2031-04-10','2031-04-10','completed',20),
              (4,'2031-04-11','2031-04-11','cancelled',500),
              (5,'2031-04-12',NULL,'unpaid',700),
              (6,'2031-04-13','2031-04-13','unpaid',900);
            INSERT INTO order_items VALUES
              (1,7,2,10,3),(2,7,3,14,3),(3,8,1,25,5),
              (4,7,1,500,0),(5,7,1,700,0),(6,7,1,900,0);
            INSERT INTO products VALUES (7,'Alpha'),(8,'Beta');
        ''')
        calls = []

        @tool
        def get_table_data(table_name: str) -> str:
            """Offline fixture schema only."""
            calls.append(('get_table_data', table_name))
            assert table_name in {'orders', 'order_items', 'products'}
            return json.dumps([dict(r) for r in connection.execute(f'PRAGMA table_info({table_name})')])

        @tool
        def execute_sql_query(query: str) -> str:
            """Execute a scripted SELECT against an in-memory fixture only."""
            calls.append(('execute_sql_query', query))
            assert query.lstrip().upper().startswith('SELECT ')
            return json.dumps([dict(row) for row in connection.execute(query)])

        period = f"o.{event} >= '2031-04-01' AND o.{event} < '2031-05-01'"
        statuses = " AND o.order_status IN ('paid','completed')" if valid else ''
        amount = 'i.quantity * i.unit_price - i.discount_amount'
        expression = {'gmv': f'SUM({amount})', 'sales': 'SUM(i.quantity)',
                      'trend': f'SUM({amount})', 'created': 'COUNT(*)',
                      'creation_time': 'MIN(o.order_date)', 'product': f'SUM({amount})',
                      'mixed': f"SUM(CASE WHEN o.order_status IN ('paid','completed') THEN {amount} ELSE 0 END)"}[metric]
        columns = expression + ' AS value'
        if metric == 'mixed':
            columns += f', SUM({amount}) AS all_status_amount'
        if metric == 'product':
            columns += ', p.product_name'
        query = f'SELECT {columns} FROM orders o JOIN order_items i ON i.order_id=o.order_id'
        if metric == 'product':
            query += ' JOIN products p ON p.product_id=i.product_id'
        query += ' WHERE ' + period + statuses
        if metric == 'trend':
            query += f' GROUP BY substr(o.{event},1,7)'
        if metric == 'product':
            query += ' GROUP BY p.product_name ORDER BY p.product_name'
        responses = [action('get_table_data', {'table_name': table}, n) for n, table in enumerate(previews)]
        if metric == 'mixed':
            # Separate earlier query has a larger, incompatible all-state total.
            broad_query = (f'SELECT SUM({amount}) AS all_status_amount FROM orders o '
                           f'JOIN order_items i ON i.order_id=o.order_id WHERE {period}')
            responses.append(action('execute_sql_query', {'query': broad_query}, len(responses)))
        responses += [action('execute_sql_query', {'query': query}, len(responses)),
                      AIMessage(content='ANSWER_FROM_EVIDENCE')]
        model = ContractScript(responses=responses)
        graph = create_agent(model=model, tools=[get_table_data, execute_sql_query], system_prompt=prompt)
        result = graph.invoke({'messages': [HumanMessage(content=question)]}, {'recursion_limit': 12})
        return calls, json.loads(result['messages'][-1].content), result['messages']

    def test_gmv_uses_payment_month_not_creation_month(self):
        calls, answer, _ = self.run_fixture()
        self.assertEqual(answer, [37])
        self.assertIn('o.paid_at', calls[-1][1])
        self.assertNotIn('order_date', calls[-1][1])

    def test_gmv_includes_both_paid_and_completed(self):
        calls, answer, _ = self.run_fixture()
        self.assertEqual(answer, [17 + 20])
        self.assertRegex(calls[-1][1], r"order_status\s+IN\s*\('paid','completed'\)")

    def test_sales_uses_paid_at(self):
        calls, answer, _ = self.run_fixture('本月有效销量？', metric='sales')
        self.assertEqual(answer, [3])
        self.assertIn('o.paid_at', calls[-1][1])

    def test_monthly_transaction_trend_uses_paid_at(self):
        calls, answer, _ = self.run_fixture('按月查看历史成交趋势', metric='trend')
        self.assertEqual(answer, [37])
        self.assertRegex(calls[-1][1], r'GROUP BY.*paid_at')

    def test_explicit_order_creation_count_allows_order_date(self):
        calls, answer, _ = self.run_fixture('本月下单量？', event='order_date', valid=False, metric='created')
        self.assertEqual(answer, [5])
        self.assertNotIn('order_status IN', calls[-1][1])

    def test_explicit_order_creation_time_allows_order_date(self):
        calls, answer, _ = self.run_fixture('本月最早的订单创建时间？', event='order_date', valid=False, metric='creation_time')
        self.assertEqual(answer, ['2031-04-10'])
        self.assertIn('MIN(o.order_date)', calls[-1][1])

    def test_cancelled_and_unpaid_excluded_even_with_payment_timestamp(self):
        _, valid, _ = self.run_fixture()
        _, all_states, _ = self.run_fixture(valid=False)
        self.assertEqual(valid, [37])
        self.assertEqual(all_states, [37 + 500 + 900])

    def test_simple_gmv_does_not_preview_inventory(self):
        calls, _, _ = self.run_fixture(previews=('orders', 'order_items'))
        self.assertNotIn(('get_table_data', 'inventory'), calls)

    def test_simple_gmv_does_not_preview_ad_metrics(self):
        calls, _, _ = self.run_fixture(previews=('orders', 'order_items'))
        self.assertNotIn(('get_table_data', 'ad_metrics'), calls)

    def test_simple_gmv_does_not_preview_products(self):
        calls, _, _ = self.run_fixture(previews=('orders', 'order_items'))
        self.assertNotIn(('get_table_data', 'products'), calls)

    def test_product_information_allowed_when_requested(self):
        calls, answer, _ = self.run_fixture('按商品名称列出成交额', metric='product', previews=('products',))
        self.assertIn(('get_table_data', 'products'), calls)
        self.assertEqual(answer, [17, 20])

    def test_stops_after_sufficient_result_without_more_tools(self):
        calls, answer, messages = self.run_fixture()
        self.assertEqual(len(calls), 1)
        self.assertEqual(answer, [37])
        self.assertIsInstance(messages[-2], ToolMessage)
        self.assertFalse(messages[-1].tool_calls)

    def test_different_amount_definitions_not_mixed(self):
        calls, answer, messages = self.run_fixture(metric='mixed', valid=False)
        self.assertEqual(len(calls), 2)
        earlier = next(m for m in messages if isinstance(m, ToolMessage))
        self.assertEqual(json.loads(earlier.content)[0]['all_status_amount'], 1437)
        evidence = json.loads(messages[-2].content)[0]
        self.assertEqual(evidence['all_status_amount'], 1437)
        self.assertEqual(answer, [evidence['value']])
        self.assertEqual(answer, [37])

    def test_wrong_date_mutation_changes_result(self):
        _, correct, _ = self.run_fixture()
        _, wrong, _ = self.run_fixture(event='order_date')
        self.assertEqual(wrong, [59])
        self.assertNotEqual(correct, wrong)

    def test_gmv_deducts_whole_line_discount(self):
        _, answer, _ = self.run_fixture()
        self.assertEqual(answer, [2 * 10 - 3 + 25 - 5])
        self.assertNotEqual(answer, [2 * 10 + 25])

    def test_missing_policy_fails_before_any_scripted_tool(self):
        for rule in RULES:
            with self.subTest(rule=rule), self.assertRaisesRegex(AssertionError, 'Missing Database prompt contract'):
                require_contract(PROMPT.replace(rule, ''))
        with self.assertRaisesRegex(AssertionError, 'Missing Database prompt contract'):
            self.run_fixture(prompt=PROMPT.replace('按 paid_at 归属业务日期', ''))

    def test_hardening_has_no_answer_month_sql_or_harness_budget(self):
        hardening = PROMPT.split('成交指标与时间口径：')[1].split('只读规则：')[0]
        self.assertIsNone(re.search(r'\d|SELECT|SKU|Harness|DeepSeek', hardening, re.I))
        self.assertNotIn('只能调用一次', hardening)

    def test_necessary_validation_and_error_recovery_remain_allowed(self):
        self.assertIn('允许必要的 Schema 确认、错误修复和验证查询', PROMPT)
        self.assertIn('依据已经确认的 Schema 修正查询', PROMPT)
        self.assertIn('不得猜测表名、字段名', PROMPT)
