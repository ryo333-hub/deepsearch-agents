"""Explicit real-MySQL acceptance; no Agent/LLM/web calls or default .env use.

Run after seeding: python -B -m scripts.verify_ecommerce_demo
Reads only .data/ecommerce/ecommerce_ro.json. Saves sanitized SQL/results for
Phase C. The three permission probes select/update/delete zero rows and rollback;
they must fail with MySQL 1142, not syntax/constraint errors.
"""

from decimal import Decimal
import json

from scripts.seed_ecommerce_demo import (
    DATABASE, ROOT, TABLES, fingerprint, generate_data, json_value, load_scenario,
    require, validate_data,
)


def acceptance_queries(s):
    # Dates come only from the reviewed local scenario, never from model input.
    from datetime import date, timedelta
    start = date.fromisoformat(s['default_analysis_month']+'-01')
    end = (start.replace(day=28)+timedelta(days=4)).replace(day=1)
    recent = date.fromisoformat(s['recent_sales_start'])
    cte = """WITH valid_items AS (
 SELECT i.*, o.paid_at, o.platform, p.category,
        i.quantity*i.unit_price-i.discount_amount AS amount
 FROM order_items i JOIN orders o ON o.order_id=i.order_id
 JOIN products p ON p.product_id=i.product_id
 WHERE o.order_status IN ('paid','completed')
), stock AS (
 SELECT product_id, SUM(stock_quantity) AS stock, SUM(safety_stock) AS safety
 FROM inventory GROUP BY product_id
), recent AS (
 SELECT product_id, SUM(quantity) AS units FROM valid_items
 WHERE paid_at >= '{recent}' AND paid_at < '{end}' GROUP BY product_id
)
""".format(recent=recent,end=end)
    window = f"paid_at >= '{start}' AND paid_at < '{end}'"
    ad_window = f"metric_date >= '{start}' AND metric_date < '{end}'"
    p = s['promotion']
    category = s['declining_category'].replace("'","''")
    return {
        'identity': 'SELECT DATABASE() AS db, CURRENT_USER() AS account',
        'tables': 'SHOW TABLES',
        'counts': 'SELECT '+', '.join(f'(SELECT COUNT(*) FROM {t}) AS {t}' for t in TABLES),
        'order_amount_mismatches': """SELECT COUNT(*) AS invalid_count FROM orders o
 LEFT JOIN (SELECT order_id, SUM(quantity*unit_price-discount_amount) AS amount
            FROM order_items GROUP BY order_id) i ON i.order_id=o.order_id
 WHERE i.amount IS NULL OR o.total_amount <> i.amount""",
        'august_gmv': cte+f"SELECT SUM(amount) AS gmv, SUM(quantity) AS units, SUM(amount-quantity*unit_cost_snapshot) AS product_gross_profit FROM valid_items WHERE {window}",
        'top5_sku': cte+f"SELECT product_id, SUM(quantity) AS units, SUM(amount) AS gmv FROM valid_items WHERE {window} GROUP BY product_id ORDER BY units DESC, product_id LIMIT 5",
        'A_inventory_risk': cte+f"""SELECT s.product_id, s.stock, s.safety, r.units AS recent_units,
 s.stock*30/NULLIF(r.units,0) AS coverage_days FROM stock s LEFT JOIN recent r USING(product_id)
 WHERE s.product_id IN ({','.join(map(str,s['hot_product_ids']))}) ORDER BY s.product_id""",
        'B_inventory_backlog': cte+f"""SELECT s.product_id,s.stock,COALESCE(r.units,0) AS recent_units,
 s.stock*30/NULLIF(r.units,0) AS coverage_days,
 CASE WHEN COALESCE(r.units,0)=0 THEN 'zero_movement' ELSE 'slow_movement' END AS movement
 FROM stock s LEFT JOIN recent r USING(product_id)
 WHERE s.stock >= {s['thresholds']['high_stock']} AND COALESCE(r.units,0) <= {s['thresholds']['low_30_day_units']}
 ORDER BY s.product_id""",
        'C_platform_roas': f"""SELECT platform, SUM(ad_spend) AS spend, SUM(revenue) AS revenue,
 SUM(revenue)/NULLIF(SUM(ad_spend),0) AS roas,
 SUM(clicks)/NULLIF(SUM(impressions),0) AS ctr,
 SUM(conversions)/NULLIF(SUM(clicks),0) AS conversion_rate
 FROM ad_metrics WHERE {ad_window} GROUP BY platform ORDER BY roas DESC""",
        'D_poor_campaign': f"""SELECT platform,campaign_name,SUM(ad_spend) AS spend,SUM(revenue) AS revenue,
 SUM(revenue)/NULLIF(SUM(ad_spend),0) AS roas,
 SUM(conversions)/NULLIF(SUM(clicks),0) AS conversion_rate FROM ad_metrics WHERE {ad_window}
 GROUP BY platform,campaign_name HAVING spend >= 10000 AND roas < 1 AND conversion_rate < 0.02
 ORDER BY spend DESC""",
        'E_high_margin': cte+f"""SELECT product_id,SUM(quantity) AS units,
 SUM(amount-quantity*unit_cost_snapshot)/NULLIF(SUM(amount),0) AS margin_rate
 FROM valid_items WHERE {window} AND product_id IN ({','.join(map(str,s['premium_product_ids']))})
 GROUP BY product_id ORDER BY product_id""",
        'F_promotion_comparison': cte+f"""SELECT
 SUM(CASE WHEN DATE(paid_at) BETWEEN '{p['baseline_start']}' AND '{p['baseline_end']}' THEN quantity ELSE 0 END) AS baseline_units,
 SUM(CASE WHEN DATE(paid_at) BETWEEN '{p['start']}' AND '{p['end']}' THEN quantity ELSE 0 END) AS promotion_units
 FROM valid_items WHERE product_id={p['product_id']}""",
        'G_historical_sku': cte+f"""SELECT DATE_FORMAT(paid_at,'%Y-%m') AS month,SUM(quantity) AS units
 FROM valid_items WHERE product_id={s['historical_product_id']} GROUP BY month ORDER BY month""",
        'H_category_divergence': cte+f"""SELECT DATE_FORMAT(paid_at,'%Y-%m') AS month,SUM(amount) AS gmv,
 SUM(CASE WHEN category='{category}' THEN amount ELSE 0 END) AS category_gmv
 FROM valid_items GROUP BY month ORDER BY month""",
    }


def verify_mysql(credential_path=None):
    import mysql.connector
    s = load_scenario()
    data = generate_data(s)
    expected = validate_data(data,s)
    credentials = json.loads((credential_path or ROOT/'.data/ecommerce/ecommerce_ro.json').read_text(encoding='utf-8'))
    require(tuple(credentials[k] for k in ('host','port','user','database'))==
            ('localhost',3307,'ecommerce_ro',DATABASE), 'Unexpected acceptance target/account')
    results = {}
    with mysql.connector.connect(**credentials,autocommit=False) as conn:
        with conn.cursor(dictionary=True) as cur:
            for name, sql in acceptance_queries(s).items():
                cur.execute(sql)
                results[name]={'sql':sql,'rows':cur.fetchall()}
            cur.execute('SHOW GRANTS')
            grants=[next(iter(row.values())) for row in cur.fetchall()]
            normalized=[g.replace('\\','') for g in grants]
            require(len(grants)==2 and any(g.startswith('GRANT USAGE ON *.* TO ') for g in grants)
                    and any(g.startswith('GRANT SELECT ON `insight_ecommerce_db`.* TO ') for g in normalized),
                    'Unexpected privileges or roles')
            require(all('GRANT OPTION' not in g for g in grants),'GRANT OPTION prohibited')
            # Snapshot all synthetic rows and compare exact values, not just counts.
            for table in TABLES:
                cur.execute(f'SELECT * FROM `{table}` ORDER BY 1')
                require(cur.fetchall()==data[table],f'{table}: persisted data differs from deterministic source')
        denied = {}
        probes = {
            'INSERT': "INSERT INTO products SELECT * FROM products WHERE 1=0",
            'UPDATE': "UPDATE products SET product_name=product_name WHERE 1=0",
            'DELETE': "DELETE FROM products WHERE 1=0",
        }
        for operation, sql in probes.items():
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
            except mysql.connector.Error as exc:
                require(exc.errno==1142,f'{operation}: expected permission error 1142, got {exc.errno}')
                denied[operation]={'sql':sql,'mysql_errno':exc.errno,'denied':True}
            else:
                raise AssertionError(f'{operation} unexpectedly permitted (zero rows targeted)')
            finally:
                conn.rollback()
        with conn.cursor(dictionary=True) as cur:
            for table in TABLES:
                cur.execute(f'SELECT * FROM `{table}` ORDER BY 1')
                require(cur.fetchall()==data[table],f'{table}: data changed during permission probes')
    r=lambda key:results[key]['rows']
    require(r('identity')[0]['db']==DATABASE and r('identity')[0]['account']=='ecommerce_ro@%','Identity mismatch')
    require({next(iter(row.values())) for row in r('tables')}==set(TABLES),'Table set mismatch')
    require(r('counts')[0]=={t:len(data[t]) for t in TABLES},'Count mismatch')
    require(r('order_amount_mismatches')[0]['invalid_count']==0,'Order totals mismatch')
    require(r('august_gmv')[0]['gmv']==expected['H']['august_gmv'],'GMV mismatch')
    require([row['product_id'] for row in r('top5_sku')]==expected['A']['top5'],'Top5 mismatch')
    require(len(r('A_inventory_risk'))==2 and all(x['stock']<x['safety'] and x['coverage_days']<7 for x in r('A_inventory_risk')),'Feature A')
    require({x['product_id'] for x in r('B_inventory_backlog')}=={s['zero_sales_product_id'],s['slow_product_id']},'Feature B')
    require(any(x['movement']=='zero_movement' and x['coverage_days'] is None for x in r('B_inventory_backlog')),'Zero-movement semantics')
    for row in r('C_platform_roas'):
        require(abs(row['roas']-expected['C'][row['platform']]['roas'])<Decimal('0.0001'),'Feature C')
    require(any(x['campaign_name']==s['poor_campaign'] for x in r('D_poor_campaign')),'Feature D')
    require(len(r('E_high_margin'))==2 and all(x['margin_rate']>Decimal('0.7') and x['units']<100 for x in r('E_high_margin')),'Feature E')
    require(r('F_promotion_comparison')[0]['promotion_units']==expected['F']['promotion_units'] and
            r('F_promotion_comparison')[0]['baseline_units']==expected['F']['baseline_units'],'Feature F')
    history={row['month']:row['units'] for row in r('G_historical_sku')}
    require(history['2026-06']==expected['G']['june_units'] and history['2026-08']==expected['G']['august_units'],'Feature G')
    months={row['month']:row for row in r('H_category_divergence')}
    require(months['2026-08']['gmv']>=months['2026-07']['gmv'] and months['2026-08']['category_gmv']<months['2026-07']['category_gmv'],'Feature H')
    return {'scenario_name':s['scenario_name'],'data_version':s['data_version'],'random_seed':s['random_seed'],
            'data_fingerprint':fingerprint(data),'synthetic_data_notice':s['synthetic_data_notice'],
            'queries':results,'offline_features':expected,'grants':grants,'permission_probes':denied,
            'persisted_rows_equal_generated_data':True,'unchanged_after_permission_probes':True,
            'attribution_count':len(data['attributions']),
            'attribution_reconciled_against_persisted_rows':True}


def main():
    report=verify_mysql()
    path=ROOT/'demo/ecommerce/acceptance_results.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=json_value)+'\n',encoding='utf-8')
    print(f'Acceptance passed; sanitized SQL/results saved to {path.relative_to(ROOT)}')


if __name__=='__main__':
    main()
