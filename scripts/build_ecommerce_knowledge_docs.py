"""Generate four synthetic internal documents from real June SQL, read-only.

Run explicitly: python -B -m scripts.build_ecommerce_knowledge_docs
Only ecommerce_ro credentials in ignored .data are used. No Agent or network API.
Pure render_documents() can be tested offline against deterministic seed data.
"""

from decimal import Decimal, ROUND_HALF_UP
import json

from scripts.seed_ecommerce_demo import DATABASE, ROOT, json_value, require

DOC_DIR = ROOT / 'demo/ecommerce/knowledge_base'
SOURCE_PATH = ROOT / 'demo/ecommerce/knowledge_docs_source.json'
FILENAMES = ('01_经营分析月报.md','02_库存经营SOP.md','03_广告投放复盘.md','04_促销策略.md')
NOTICE = '本文档为项目演示使用的合成企业资料，不代表真实企业经营数据。'
START, END = '2026-06-01', '2026-07-01'
VALID = """FROM orders o JOIN order_items i ON o.order_id=i.order_id
 JOIN products p ON p.product_id=i.product_id
 WHERE o.order_status IN ('paid','completed') AND o.paid_at >= '2026-06-01' AND o.paid_at < '2026-07-01'"""
QUERIES = {
    'total': 'SELECT SUM(i.quantity*i.unit_price-i.discount_amount) AS gmv, SUM(i.quantity) AS units '+VALID,
    'categories': 'SELECT p.category,SUM(i.quantity*i.unit_price-i.discount_amount) AS gmv,SUM(i.quantity) AS units '+VALID+' GROUP BY p.category ORDER BY p.category',
    'sku3': 'SELECT p.product_id,p.product_name,SUM(i.quantity) AS units,SUM(i.quantity*i.unit_price-i.discount_amount) AS gmv '+VALID+' AND p.product_id=3 GROUP BY p.product_id,p.product_name',
    'top_skus': 'SELECT p.product_id,p.product_name,SUM(i.quantity) AS units,SUM(i.quantity*i.unit_price-i.discount_amount) AS gmv '+VALID+' GROUP BY p.product_id,p.product_name ORDER BY units DESC,p.product_id LIMIT 5',
    'platforms': 'SELECT o.platform,SUM(i.quantity*i.unit_price-i.discount_amount) AS gmv '+VALID+' GROUP BY o.platform ORDER BY o.platform',
    'ads': """SELECT platform,SUM(ad_spend) AS spend,SUM(revenue) AS revenue,
 SUM(clicks) AS clicks,SUM(impressions) AS impressions,SUM(conversions) AS conversions
 FROM ad_metrics WHERE metric_date >= '2026-06-01' AND metric_date < '2026-07-01'
 GROUP BY platform ORDER BY platform""",
}


def fmt(value, places=2):
    return format(Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP),f'.{places}f')


def ratio(n, d, percent=False):
    if not d:
        return '不可计算（分母为零）'
    return fmt(Decimal(n)/Decimal(d)*(100 if percent else 1),4)+('%' if percent else '')


def header(name, kind, authored, coverage, scope):
    return (f'# {name}\n\n文档名称：{name}\n文档类型：{kind}\n文档版本：v1.0\n'
            f'成文日期：{authored}\n数据覆盖期间：{coverage}\n适用范围：{scope}\n'
            f'数据性质：{NOTICE}\n金额单位：人民币元；时间口径：Asia/Shanghai。\n\n')


def render_documents(metrics):
    total, sku = metrics['total'][0], metrics['sku3'][0]
    monthly = header('经营分析月报','历史经营月报','2026-07-05','2026-06-01 ～ 2026-06-30','虚构演示商品与三渠道的六月有效成交')
    monthly += ('## 六月整体经营与指标口径\n\n'
        f"2026 年 6 月整体 GMV 为 {fmt(total['gmv'])} 元，有效销量为 {int(total['units'])} 件。"
        '仅纳入 paid/completed 订单，按 paid_at 归属日期；明细金额为数量乘单价减去整条明细优惠。'
        '未建模退款、运费和税费，商品毛利不等于净利润。\n\n')
    monthly += '## 六月各品类表现\n\n| 品类 | GMV（元） | 有效销量（件） |\n|---|---:|---:|\n'
    monthly += ''.join(f"| {r['category']} | {fmt(r['gmv'])} | {int(r['units'])} |\n" for r in metrics['categories'])+'\n'
    monthly += ('## SKU 3 六月历史判断\n\n'
        f"SKU 3 在 2026 年 6 月的有效销量为 {int(sku['units'])} 件，成交金额为 {fmt(sku['gmv'])} 元。"
        f"商品名称为{sku['product_name']}。SKU 3 在 6 月属于重点销售商品之一。"
        '本历史判断只描述六月，不保证未来月份维持相同表现；后续应按同口径重新查询。\n\n')
    monthly += '## 六月核心 SKU（按有效销量排序）\n\n| SKU | 商品 | 有效销量（件） | GMV（元） |\n|---|---|---:|---:|\n'
    monthly += ''.join(f"| {r['product_id']} | {r['product_name']} | {int(r['units'])} | {fmt(r['gmv'])} |\n" for r in metrics['top_skus'])+'\n'
    monthly += '## 六月平台经营表现\n\n'
    monthly += '；'.join(f"{r['platform']}有效成交 GMV 为 {fmt(r['gmv'])} 元" for r in metrics['platforms'])+'。平台名称仅为合成场景标签，不代表真实平台业绩。\n\n'
    monthly += ('## 库存关注与月报建议\n\n月报未获得六月末库存快照，因此不作当时缺货或积压的数值结论。'
                '建议后续核对库存快照日期、近月销量与补货周期；优先关注核心 SKU，但不能依据历史销量直接断言当前库存风险。\n')

    sop = header('库存经营SOP','内部规则','2026-06-01','规则有效期：2026-06-01 ～ 2026-08-31；适用于 2026-09-01 期末快照评估','虚构演示企业库存，不是行业统一标准')
    sop += ('## 缺货风险判定\n\n根据企业库存 SOP，当前库存 < safety_stock，或者库存覆盖天数 < 7 天，标记为高缺货风险。'
            '先在同一 SKU、同一仓库口径比较；跨仓汇总时库存与 safety_stock 都必须求和。'
            '需复核可售库存、在途到货和补货周期后安排补货，不能仅凭汇总结果立即下采购单。\n\n'
            '## 库存覆盖与正常关注\n\n库存覆盖天数＝当前可售库存÷最近 30 天日均有效销量。'
            '对于 2026-09-01 快照，最近 30 天指 2026-08-02 至 2026-08-31，仅计算已付款或已完成订单。'
            '覆盖天数在 7～30 天属于正常关注；超过 30 天且不超过 90 天时持续监测，不自动判为积压。\n\n'
            '## 库存积压处理\n\n根据企业库存 SOP，库存覆盖天数 > 90 天，标记为库存积压关注。'
            '核查商品动销、效期、仓间分布和近期需求，再评估减少补货、调拨或受控促销；不要无条件打折清仓。\n\n'
            '## 零动销库存处理\n\n最近 30 天销量为 0 且当前库存 > 0，标记为零动销库存。'
            '此时库存覆盖天数不可计算，应显示 NULL 或明确不可计算，不应显示 0 天或伪造无限天数。'
            '优先核对上架状态、商品曝光、需求和库存记录，暂停未经论证的追加补货。\n\n'
            '## 规则边界\n\n上述阈值是企业内部 Demo SOP，不是行业规定。事实数据来自当次数据库查询，规则依据来自本文档；两者需分别标注来源。\n')

    ads = header('广告投放复盘','历史广告复盘','2026-07-06','2026-06-01 ～ 2026-06-30','虚构演示三渠道归因广告数据')
    ads += ('## 六月广告指标定义\n\nROAS＝广告归因成交金额÷广告花费；CTR＝点击量÷曝光量；广告转化率＝归因成交次数÷点击量。'
            '均先求和后相除，不平均行级比率。分母为零时不可计算。ROAS 不是利润 ROI。'
            '广告收入仅含部分有效成交明细，每笔订单最多归因一次，不应与商城 GMV 相加。\n\n')
    ads += '## 六月各渠道投放表现\n\n| 渠道 | 花费（元） | 归因收入（元） | ROAS | CTR | 广告转化率 |\n|---|---:|---:|---:|---:|---:|\n'
    ads += ''.join(f"| {r['platform']} | {fmt(r['spend'])} | {fmt(r['revenue'])} | {ratio(r['revenue'],r['spend'])} | {ratio(r['clicks'],r['impressions'],True)} | {ratio(r['conversions'],r['clicks'],True)} |\n" for r in metrics['ads'])+'\n'
    valid_ads = [r for r in metrics['ads'] if r['spend']]
    poorest = min(valid_ads,key=lambda r:Decimal(r['revenue'])/Decimal(r['spend']))
    ads += ('## 历史广告效率问题与后续观察\n\n'
            f"按六月实际合成数据，{poorest['platform']}渠道的汇总 ROAS 最低，为 {ratio(poorest['revenue'],poorest['spend'])}。"
            '历史上该渠道部分活动存在投放效率偏低问题，建议复核人群、素材和归因口径。'
            '其他渠道也需按后续月份重新评估，不以六月结论直接决定未来预算。'
            '平台名称仅用于合成 Demo，不能解读为真实平台优劣比较。\n')

    promo = header('促销策略','内部活动评估规则','2026-08-01','计划窗口：2026-08-15 ～ 2026-08-21；基线：2026-08-08 ～ 2026-08-14','虚构演示企业促销与商品毛利管理')
    promo += ('## 活动窗口与可比基线\n\n计划促销窗口为 2026-08-15 至 2026-08-21；对照基线为 2026-08-08 至 2026-08-14。'
              '两个窗口各为 7 天，使用相同 SKU、有效成交状态和成交日期口径。本文为活动前规则，不预写活动后的销量结果。\n\n'
              '## 折扣与商品毛利\n\n折扣金额指整条明细优惠，不是每件商品重复扣减。活动前应核验数量乘单价减优惠后的成交金额非负。'
              '商品毛利＝明细成交金额－数量乘单位成本快照。广告、物流、人工、平台抽佣、税费和仓储并未全部纳入，因此不能称为净利润。\n\n'
              '## 促销效果与因果边界\n\n促销期间销量上涨只能作为经营观察，不能仅凭前后变化断言因果。'
              '不能只凭前后销量差异就声称促销导致销量增长，或证明促销有效。'
              '活动评估需结合同期趋势、渠道投放、库存、商品曝光和其他促销因素；条件不足时明确存在未验证因素。\n\n'
              '## 活动复盘要求\n\n复盘应分别列出有效销量、折后成交金额、优惠支出、商品毛利和库存变化。'
              '只使用实际可得指标，缺失商品曝光、同期对照或历史库存时注明缺口；区分观察、假设与因果结论。\n')
    return dict(zip(FILENAMES,[monthly,sop,ads,promo]))


def query_historical_metrics():
    import mysql.connector
    creds=json.loads((ROOT/'.data/ecommerce/ecommerce_ro.json').read_text(encoding='utf-8'))
    require(tuple(creds[k] for k in ('host','port','database','user'))==('localhost',3307,DATABASE,'ecommerce_ro'),'Unexpected document data source')
    results={}
    with mysql.connector.connect(**creds,autocommit=False) as conn:
        conn.start_transaction(readonly=True,consistent_snapshot=True)
        with conn.cursor(dictionary=True) as cur:
            for name,sql in QUERIES.items():
                cur.execute(sql)
                results[name]=cur.fetchall()
        conn.rollback()
    require(len(results['sku3'])==1 and bool(results['total'][0]['gmv']),'Missing June business data')
    return results


def main():
    metrics=query_historical_metrics()
    docs=render_documents(metrics)
    DOC_DIR.mkdir(parents=True,exist_ok=True)
    for name,text in docs.items():
        (DOC_DIR/name).write_text(text,encoding='utf-8')
    SOURCE_PATH.write_text(json.dumps({'database':DATABASE,'user':'ecommerce_ro','period':[START,END],
        'end_exclusive':True,'synthetic':True,'queries':QUERIES,'metrics':metrics},ensure_ascii=False,
        indent=2,default=json_value)+'\n',encoding='utf-8')
    print('Built four UTF-8 documents from real read-only June SQL; source evidence saved.')


if __name__=='__main__':
    main()
