"""Explicit, non-destructive synthetic ecommerce provisioning (no Agent imports).

Default: generate and validate entirely offline. --apply additionally requires
ECOMMERCE_ADMIN_PASSWORD in this process; connects only to localhost:3307.
Never reads/changes the application .env. Never resets an existing database.
RO credentials go only to ignored .data/ecommerce/ecommerce_ro.json. A sanitized
SQL acceptance report is written to demo/ecommerce/acceptance_results.json.
DDL is not transactional: on failure stop, retain partial state, never auto-drop.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import random
import secrets

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / 'demo/ecommerce/scenario.json'
DATABASE = 'insight_ecommerce_db'
TABLES = ('products', 'orders', 'order_items', 'inventory', 'ad_metrics')
CENT = Decimal('0.01')
ZERO = Decimal('0.00')


def money(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_value)


def fingerprint(data):
    return hashlib.sha256(serialized(data).encode('utf-8')).hexdigest()


def load_scenario():
    return json.loads(SCENARIO_PATH.read_text(encoding='utf-8'))


def line_amount(item):
    return item['quantity'] * item['unit_price'] - item['discount_amount']


def safe_ratio(numerator, denominator):
    return Decimal(numerator) / Decimal(denominator) if denominator else None


def generate_data(scenario=None):
    s = scenario or load_scenario()
    rng = random.Random(s['random_seed'])
    data = {name: [] for name in TABLES}
    data['attributions'] = []  # Audit-only ledger, not a sixth application table.
    kinds = ['洁面乳', '面霜', '精华', '口红', '防晒', '清洁剂', '粉底', '唇釉', '修护乳', '爽肤水']
    for pid in range(1, s['product_count'] + 1):
        category = s['categories'][(pid - 1) // 10]
        if pid in s['hot_product_ids']:
            category = '个护'
        elif pid in [s['promotion']['product_id'], *s['premium_product_ids']]:
            category = '彩妆'
        elif pid == s['slow_product_id']:
            category = '家清'
        price = money(50 + 4 * pid)
        cost = money(price * Decimal('0.45'))
        if pid in s['premium_product_ids']:
            price, cost = money(160), money(10)
        if pid == s['historical_product_id']:
            price, cost = money(180), money(60)
        kind = kinds[pid - 1] if pid <= 10 else {'护肤': '保湿乳', '彩妆': '彩妆盘', '个护': '沐浴露', '家清': '洗衣液'}[category]
        data['products'].append(dict(product_id=pid, product_name=f'虚构演示星禾{kind}{pid:02d}',
            category=category, cost_price=cost, sale_price=price, status='active',
            created_at=datetime.fromisoformat(s['product_created_at'])))
    products = {p['product_id']: p for p in data['products']}
    start, end = date.fromisoformat(s['transaction_start']), date.fromisoformat(s['transaction_end'])
    months = sorted({(start + timedelta(days=i)).replace(day=1) for i in range((end-start).days+1)})
    promotion = s['promotion']
    for mi, month in enumerate(months):
        days = [month + timedelta(days=i) for i in range(31)
                if (month + timedelta(days=i)).month == month.month]
        for n in range(s['orders_per_month']):
            oid = len(data['orders']) + 1
            day = days[n % len(days)]
            placed = datetime.combine(day, time(10, n % 60))
            status = 'unpaid' if n % 20 == 0 else 'cancelled' if n % 20 == 1 else 'paid' if n % 2 else 'completed'
            paid = placed + timedelta(minutes=5) if status in s['metric_contract']['valid_order_statuses'] else None
            platform = s['platforms'][n % len(s['platforms'])]
            slot = n % 10
            if slot < 4:
                pid, quantity = s['hot_product_ids'][slot % 2], s['monthly_hot_quantity'][mi]
            elif slot == 4:
                pid, quantity = s['historical_product_id'], s['monthly_historical_quantity'][mi]
            elif slot == 5:
                pid = promotion['product_id']
                quantity = 15 if promotion['start'] <= day.isoformat() <= promotion['end'] else 1
            elif slot == 6:
                pid, quantity = s['premium_product_ids'][(n // 10) % 2], 1
            else:
                pid, quantity = rng.randint(9, 30), rng.randint(1, 3)
            if mi == len(months)-1 and n == s['orders_per_month']-1:
                pid, quantity = s['slow_product_id'], 1
            entries = []
            for item_pid, qty in [(pid, quantity), (31 + n % 10, s['monthly_household_quantity'][mi])]:
                product = products[item_pid]
                rate = Decimal('0.30') if item_pid == promotion['product_id'] and promotion['start'] <= day.isoformat() <= promotion['end'] else Decimal(rng.choice(['0', '0.05', '0.10']))
                item = dict(order_item_id=len(data['order_items'])+1, order_id=oid,
                    product_id=item_pid, quantity=qty, unit_price=product['sale_price'],
                    discount_amount=money(product['sale_price'] * qty * rate), unit_cost_snapshot=product['cost_price'])
                data['order_items'].append(item)
                entries.append(item)
            data['orders'].append(dict(order_id=oid, customer_id=f'demo_customer_{1+n%250:04d}',
                order_date=placed, paid_at=paid, platform=platform, order_status=status,
                total_amount=sum((line_amount(i) for i in entries), ZERO)))
            # Only the primary line of a selected paid order is attributed, once.
            if paid and pid in s['ad_focus_product_ids'] and oid % 4 != 0:
                data['attributions'].append(dict(order_id=oid, order_item_id=entries[0]['order_item_id'],
                    metric_date=paid.date(), platform=platform, product_id=pid, revenue=line_amount(entries[0])))
    attributed = defaultdict(list)
    for a in data['attributions']:
        attributed[(a['metric_date'], a['platform'], a['product_id'])].append(a)
    for offset in range((end-start).days+1):
        day = start + timedelta(days=offset)
        for pi, platform in enumerate(s['platforms']):
            for pid in s['ad_focus_product_ids']:
                evidence = attributed[(day, platform, pid)]
                revenue = sum((a['revenue'] for a in evidence), ZERO)
                poor = pi == 2 and pid == s['historical_product_id']
                clicks = len(evidence) + (1000 if poor else [12, 35, 90][pi]) + rng.randint(0, 8)
                spend = money(revenue / Decimal(s['channel_target_roas'][pi]) + (500 if poor else 10))
                data['ad_metrics'].append(dict(metric_id=len(data['ad_metrics'])+1,
                    metric_date=day, platform=platform, campaign_name=s['poor_campaign'] if poor else f'演示{platform}商品投放',
                    product_id=pid, impressions=clicks*20, clicks=clicks, ad_spend=spend,
                    conversions=len(evidence), revenue=revenue))
    for pid in products:
        for warehouse in s['warehouses']:
            quantity = 10 if pid in s['hot_product_ids'] else 1000 if pid in [s['zero_sales_product_id'],s['slow_product_id']] else 100
            data['inventory'].append(dict(inventory_id=len(data['inventory'])+1, product_id=pid,
                stock_quantity=quantity, safety_stock=50, warehouse=warehouse,
                updated_at=datetime.fromisoformat(s['inventory_snapshot_date'])))
    return data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def feature_evidence(data, s):
    orders = {o['order_id']: o for o in data['orders']}
    products = {p['product_id']: p for p in data['products']}
    monthly_units, monthly_gmv, category_gmv = defaultdict(int), defaultdict(lambda: ZERO), defaultdict(lambda: ZERO)
    recent, august_profit, august_amount = defaultdict(int), defaultdict(lambda: ZERO), defaultdict(lambda: ZERO)
    promo, baseline = 0, 0
    for item in data['order_items']:
        order = orders[item['order_id']]
        if order['order_status'] not in s['metric_contract']['valid_order_statuses']:
            continue
        day = order['paid_at'].date().isoformat()
        month, pid = day[:7], item['product_id']
        amount = line_amount(item)
        monthly_units[(month,pid)] += item['quantity']
        monthly_gmv[month] += amount
        category_gmv[(month,products[pid]['category'])] += amount
        if s['recent_sales_start'] <= day < s['recent_sales_end_exclusive']:
            recent[pid] += item['quantity']
        if month == s['default_analysis_month']:
            august_amount[pid] += amount
            august_profit[pid] += amount - item['quantity']*item['unit_cost_snapshot']
        if pid == s['promotion']['product_id']:
            if s['promotion']['start'] <= day <= s['promotion']['end']:
                promo += item['quantity']
            if s['promotion']['baseline_start'] <= day <= s['promotion']['baseline_end']:
                baseline += item['quantity']
    stock, safety = defaultdict(int), defaultdict(int)
    for row in data['inventory']:
        stock[row['product_id']] += row['stock_quantity']
        safety[row['product_id']] += row['safety_stock']
    top = sorted(products, key=lambda pid: (-monthly_units[(s['default_analysis_month'],pid)],pid))[:5]
    hot = [dict(product_id=pid, sold_units=monthly_units[(s['default_analysis_month'],pid)], stock=stock[pid],
                safety=safety[pid], coverage_days=safe_ratio(stock[pid]*30,recent[pid])) for pid in s['hot_product_ids']]
    slow = [dict(product_id=pid,stock=stock[pid],recent_units=recent[pid], coverage_days=safe_ratio(stock[pid]*30,recent[pid]))
            for pid in (s['zero_sales_product_id'],s['slow_product_id'])]
    channels, poor = {}, {'spend':ZERO,'revenue':ZERO,'clicks':0,'conversions':0}
    for platform in s['platforms']:
        ads = [a for a in data['ad_metrics'] if a['platform']==platform and a['metric_date'].isoformat().startswith(s['default_analysis_month'])]
        spend, revenue = sum((a['ad_spend'] for a in ads),ZERO), sum((a['revenue'] for a in ads),ZERO)
        channels[platform] = {'spend':spend,'revenue':revenue,'roas':safe_ratio(revenue,spend)}
        for a in ads:
            if a['campaign_name']==s['poor_campaign']:
                for field, key in [('ad_spend','spend'),('revenue','revenue'),('clicks','clicks'),('conversions','conversions')]:
                    poor[key] += a[field]
    poor['roas'],poor['conversion_rate'] = safe_ratio(poor['revenue'],poor['spend']),safe_ratio(poor['conversions'],poor['clicks'])
    premium = [dict(product_id=pid,units=monthly_units[(s['default_analysis_month'],pid)],
                    margin_rate=safe_ratio(august_profit[pid],august_amount[pid])) for pid in s['premium_product_ids']]
    historical = s['historical_product_id']
    category = s['declining_category']
    return {'A':{'top5':top,'risk_skus':hot}, 'B':slow,'C':channels,'D':poor,'E':premium,
            'F':{'baseline_units':baseline,'promotion_units':promo,'causal_claim':False},
            'G':{'product_id':historical,'june_units':monthly_units[('2026-06',historical)],'august_units':monthly_units[('2026-08',historical)]},
            'H':{'july_gmv':monthly_gmv['2026-07'],'august_gmv':monthly_gmv['2026-08'],
                 'category':category,'july_category_gmv':category_gmv[('2026-07',category)],'august_category_gmv':category_gmv[('2026-08',category)]}}


def validate_data(data, scenario=None):
    s = scenario or load_scenario()
    expected = dict(zip(TABLES,[s['product_count'],s['order_target_count'],s['order_item_count'],s['inventory_count'],s['ad_metric_count']]))
    keys = dict(zip(TABLES,['product_id','order_id','order_item_id','inventory_id','metric_id']))
    for table in TABLES:
        require(len(data[table]) == expected[table], f'{table}: count mismatch')
        require(len({r[keys[table]] for r in data[table]}) == len(data[table]), f'{table}: duplicate primary key')
        for row in data[table]:
            for name in ('cost_price','sale_price','total_amount','unit_price','discount_amount','unit_cost_snapshot','ad_spend','revenue'):
                if name in row:
                    require(isinstance(row[name],Decimal) and row[name].is_finite() and row[name]==money(row[name]), 'Money must be finite Decimal cents')
    products = {p['product_id']:p for p in data['products']}
    orders = {o['order_id']:o for o in data['orders']}
    items = {i['order_item_id']:i for i in data['order_items']}
    amounts = defaultdict(lambda:ZERO)
    for p in products.values():
        require(p['category'] in s['categories'] and p['cost_price']>=0 and p['sale_price']>=0, 'invalid product')
    for item in items.values():
        require(item['order_id'] in orders and item['product_id'] in products, 'orphan order item')
        require(item['quantity']>0 and item['unit_price']>=0 and item['unit_cost_snapshot']>=0, 'invalid item values')
        require(0<=item['discount_amount']<=item['quantity']*item['unit_price'], 'invalid discount')
        amounts[item['order_id']] += line_amount(item)
    for o in orders.values():
        require(o['order_status'] in ['paid','completed','unpaid','cancelled'], 'invalid order status')
        valid = o['order_status'] in s['metric_contract']['valid_order_statuses']
        require((o['paid_at'] is not None)==valid, 'paid_at/status mismatch')
        require(o['total_amount']==amounts[o['order_id']] and o['total_amount']>=0, 'order amount mismatch')
        require(s['transaction_start']<=o['order_date'].date().isoformat()<=s['transaction_end'], 'order date out of range')
        require(not valid or o['order_date']<=o['paid_at'] and o['paid_at'].date().isoformat()<=s['transaction_end'], 'paid date out of range')
        require(o['platform'] in s['platforms'], 'unknown platform')
    inv = data['inventory']
    require(len({(r['product_id'],r['warehouse']) for r in inv})==len(inv), 'duplicate inventory grain')
    for r in inv:
        require(r['product_id'] in products and r['warehouse'] in s['warehouses'], 'invalid inventory FK/warehouse')
        require(r['stock_quantity']>=0 and r['safety_stock']>=0, 'negative inventory')
        require(r['updated_at'].date().isoformat()==s['inventory_snapshot_date'], 'inventory date mismatch')
    ledger = defaultdict(list)
    require(len({a['order_id'] for a in data['attributions']})==len(data['attributions']), 'duplicate order attribution')
    for a in data['attributions']:
        require(a['order_item_id'] in items and a['order_id'] in orders, 'orphan attribution')
        i,o = items[a['order_item_id']],orders[a['order_id']]
        require(o['paid_at'] is not None and i['order_id']==a['order_id'], 'invalid attributed order')
        require(a['platform']==o['platform'] and a['metric_date']==o['paid_at'].date() and a['product_id']==i['product_id'], 'attribution mismatch')
        require(a['revenue']==line_amount(i), 'attribution revenue mismatch')
        ledger[(a['metric_date'],a['platform'],a['product_id'])].append(a)
    ads = data['ad_metrics']
    require(len({(a['metric_date'],a['platform'],a['campaign_name'],a['product_id']) for a in ads})==len(ads), 'duplicate ad grain')
    used = set()
    for a in ads:
        require(a['product_id'] in products and a['platform'] in s['platforms'], 'invalid ad FK/platform')
        require(s['transaction_start']<=a['metric_date'].isoformat()<=s['transaction_end'], 'ad date out of range')
        require(0<=a['conversions']<=a['clicks']<=a['impressions'] and a['ad_spend']>=0 and a['revenue']>=0, 'invalid ad metrics')
        key = a['metric_date'],a['platform'],a['product_id']
        require(key not in used, 'attribution credited to multiple campaigns')
        used.add(key)
        require(a['conversions']==len(ledger[key]) and a['revenue']==sum((r['revenue'] for r in ledger[key]),ZERO), 'ad ledger mismatch')
    require(set(k for k,v in ledger.items() if v)<=used, 'unreported attribution')
    f,t = feature_evidence(data,s),s['thresholds']
    require(all(r['product_id'] in f['A']['top5'] and (r['stock']<r['safety'] or r['coverage_days']<7) for r in f['A']['risk_skus']), 'feature A missing')
    require(all(r['stock']>=t['high_stock'] and r['recent_units']<=t['low_30_day_units'] for r in f['B']) and f['B'][0]['recent_units']==0, 'feature B missing')
    roas = sorted(r['roas'] for r in f['C'].values())
    require(all(b-a>=Decimal(t['channel_roas_gap']) for a,b in zip(roas,roas[1:])), 'feature C missing')
    require(f['D']['spend']>=Decimal(t['poor_spend']) and f['D']['roas']<Decimal(t['poor_roas']) and f['D']['conversion_rate']<Decimal(t['poor_conversion_rate']), 'feature D missing')
    require(all(r['margin_rate']>=Decimal(t['high_margin_rate']) and r['units']<min(h['sold_units'] for h in f['A']['risk_skus'])/4 for r in f['E']), 'feature E missing')
    require(f['F']['promotion_units']>f['F']['baseline_units']>0, 'feature F missing')
    require(f['G']['june_units']>=2*f['G']['august_units']>0, 'feature G missing')
    require(f['H']['august_gmv']>=f['H']['july_gmv'] and f['H']['august_category_gmv']<f['H']['july_category_gmv'], 'feature H missing')
    return f


def schema_statements():
    # Controlled script contains no semicolons inside literals or procedures.
    content = (ROOT/'docker/mysql/ecommerce_schema.sql').read_text(encoding='utf-8')
    content = '\n'.join(line for line in content.splitlines() if not line.lstrip().startswith('--'))
    return [statement.strip() for statement in content.split(';') if statement.strip()]


def apply_data(data, admin_config, credential_path):
    """Management-only entry. Preflight before DDL; batch INSERT is transactional."""
    import mysql.connector
    validate_data(data)
    require(admin_config.get('host')=='localhost' and admin_config.get('port')==3307, 'Only the local demo server is allowed')
    require(credential_path.resolve().is_relative_to((ROOT/'.data/ecommerce').resolve()), 'Credential path must stay in ignored local data')
    config = {**admin_config,'database':None,'autocommit':False}
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=%s',(DATABASE,))
            if cur.fetchone():
                raise ValueError('Demo data/schema already exists. No overwrite or reset is allowed.')
            cur.execute("SELECT User FROM mysql.user WHERE User='ecommerce_ro'")
            require(cur.fetchone() is None, 'ecommerce_ro already exists; permissions will not be altered')
            require(not credential_path.exists(), 'Local ecommerce credentials already exist; no overwrite')
            password = secrets.token_urlsafe(32)
            # Persist first, exclusively: if DDL fails, credentials are not lost.
            credential_path.parent.mkdir(parents=True,exist_ok=True)
            with credential_path.open('x',encoding='utf-8') as output:
                json.dump(dict(host='localhost',port=3307,user='ecommerce_ro',password=password,database=DATABASE),output)
            for statement in schema_statements():
                cur.execute(statement,(password,)) if statement.startswith('CREATE USER') else cur.execute(statement)
            conn.start_transaction()
            try:
                for table in TABLES:
                    columns = tuple(data[table][0])
                    sql = f"INSERT INTO `{table}` ({','.join('`'+c+'`' for c in columns)}) VALUES ({','.join(['%s']*len(columns))})"
                    cur.executemany(sql,[tuple(row[c] for c in columns) for row in data[table]])
                conn.commit()
            except Exception:
                conn.rollback()
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Explicit local MySQL management import')
    args = parser.parse_args()
    data = generate_data()
    features = validate_data(data)
    print(serialized({'counts':{t:len(data[t]) for t in TABLES},'fingerprint':fingerprint(data),'features':features}))
    if args.apply:
        password = os.environ.get('ECOMMERCE_ADMIN_PASSWORD')
        require(bool(password),'ECOMMERCE_ADMIN_PASSWORD is required; .env is not used')
        apply_data(data,dict(host='localhost',port=3307,user='root',password=password),ROOT/'.data/ecommerce/ecommerce_ro.json')
        print('Imported. Run the explicit ecommerce MySQL acceptance checks next.')


if __name__ == '__main__':
    main()
