"""Small acceptance-only source policy, not a new production citation system.

Unknown publication dates remain unknown. Search snippets are not independently
verified article facts, nor evidence of this synthetic company's performance.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import ipaddress
import re
import socket
from urllib.parse import urlsplit

QUERIES = [
    '2026年8月 中国美妆消费市场 化妆品零售 趋势 国家统计局',
    '2026年8月 美妆日用消费品 抖音 小红书 快手 内容电商 平台官方 趋势报告',
    '2026年8月 护肤品消费市场 公开行业报告 趋势',
]
QUALITY_DOMAINS = (
    'stats.gov.cn', 'gov.cn', 'xinhuanet.com', 'news.cn', 'people.com.cn',
    'cctv.com', 'reuters.com', 'stcn.com', 'cnstock.com', 'yicai.com',
    '21jingji.com', 'jiemian.com', 'cnr.cn', 'ce.cn', 'cbndata.com',
    'iresearch.com.cn', 'iimedia.cn', 'oceanengine.com', 'douyin.com',
    'kuaishou.com', 'xiaohongshu.com', 'cninfo.com.cn', 'hkexnews.hk', 'mintel.com',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def domain(url):
    if not isinstance(url, str):
        return None
    try:
        p = urlsplit(url)
        if p.scheme not in ('https', 'http') or not p.hostname or p.username or p.password:
            return None
        if p.port not in (None, 80, 443):
            return None
        host = p.hostname.lower().rstrip('.')
        if host == 'localhost' or '.' not in host:
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        return host
    except ValueError:
        return None


def public_target(url):
    host = domain(url)
    if not host:
        raise ValueError('Invalid public URL')
    addresses = socket.getaddrinfo(host, urlsplit(url).port or (443 if url.startswith('https:') else 80))
    trusted = any(host == d or host.endswith('.'+d) for d in QUALITY_DOMAINS)
    # This Windows resolver uses a local proxy's 198.18/15 fake-IP DNS mode.
    # Permit that mapping only for the reviewed public publisher allowlist;
    # still use the original HTTPS hostname, never send credentials to it.
    def permitted(address):
        ip = ipaddress.ip_address(address)
        return ip.is_global or (trusted and ip in ipaddress.ip_network('198.18.0.0/15'))
    if not addresses or not all(permitted(a[4][0]) for a in addresses):
        raise ValueError('Non-public target')
    return host


def publication_value(raw):
    value = raw.get('published_date') or raw.get('published_at')
    if isinstance(value, str) and value.strip():
        return value, 'tavily_field'
    # Some general-search results expose the page's dateline only inside the
    # snippet. Require a standalone dated *time* line; never infer from a URL,
    # a sales-period mention, copyright year, or vague relative date.
    content = raw.get('content')
    if isinstance(content, str):
        match = re.search(r'^\s*(?:#{1,6}\s*)?(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?\s+(\d{1,2}):(\d{2})(?:\s|$)', content, re.M)
        if match:
            try:
                value = datetime(*map(int, match.groups())).isoformat(timespec='minutes')
                return value, 'tavily_content_dateline'
            except ValueError:
                pass
    return None, 'not_provided'


def publication(raw):
    value, _ = publication_value(raw)
    if not isinstance(value, str) or not value.strip():
        return None, 'unknown'
    try:
        dt = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return value, 'unverified_date'
    date = dt.date().isoformat()
    if '2026-08-01' <= date <= '2026-08-31':
        return value, 'august_2026'
    return value, 'outside_august_2026'


def normalize_sources(payload):
    if not isinstance(payload, dict) or payload.get('error'):
        return {'status': 'unavailable', 'sources': [], 'error':
                payload.get('error') if isinstance(payload, dict) else {'type': 'InvalidResponse'}}
    sources = []
    for raw in payload.get('results', []):
        if not isinstance(raw, dict):
            continue
        host = domain(raw.get('url'))
        title, text = raw.get('title'), raw.get('content')
        if not host or not isinstance(title, str) or not title.strip() or not isinstance(text, str) or not text.strip():
            continue
        published, period = publication(raw)
        relevant = bool(re.search(r'美妆|化妆品|护肤|日用|家清|cosmetic|skincare|beauty|消费品', title+' '+text, re.I))
        trusted = any(host == d or host.endswith('.'+d) for d in QUALITY_DOMAINS)
        coverage = re.search(r'(20\d{2})年(\d{1,2})[—～~\-](\d{1,2})月', title)
        data_period = (f'{coverage[1]}-{int(coverage[2]):02d}..{coverage[1]}-{int(coverage[3]):02d}'
                       if coverage else None)
        sources.append({'title': title, 'url': raw['url'], 'published_date': published,
            'date_basis': publication_value(raw)[1], 'data_period_from_title': data_period,
            'period': period, 'domain': host, 'content': text,
            'score': raw.get('score'), 'relevant': relevant, 'priority_source': trusted,
            'selected': False, 'used_by_main': False,
            'selection_reason': 'pending'})
    # At most three sources; no forced representation of every platform.
    selected = sorted((s for s in sources if s['relevant'] and s['priority_source']),
                      key=lambda s: s['period'] != 'august_2026')[:3]
    for s in sources:
        s['selected'] = any(s is x for x in selected)
        s['selection_reason'] = ('public_background_only' if s['selected'] and s['period'] != 'august_2026'
            else 'same_month_public_context' if s['selected'] else 'outside_minimal_quality_or_relevance_selection')
    return {'status': 'ok' if selected else 'empty', 'sources': sources,
            'publication_in_august': any(s['period'] == 'august_2026' for s in selected),
            # Publication during August alone is not proof that the article's
            # statistics cover August. This minimal policy doesn't certify that.
            'same_period_evidence': False,
            'limitation': '公开材料仅为背景；发布日期不等于统计覆盖期。未核实同期口径时，不能解释八月内部经营变化。'}


def format_network(result):
    if result['status'] == 'unavailable':
        return '公开信息来源当前不可用。未补造任何市场数据。'
    selected = [s for s in result['sources'] if s['selected']]
    if not selected:
        return '未找到足够公开资料。未编造来源。'
    lines = []
    if not result['same_period_evidence']:
        lines.append('未找到足够同期公开证据。以下仅列检索到的非同期或日期未确认背景，不能作为八月业绩变化证据。')
    for i, s in enumerate(selected, 1):
        s['used_by_main'] = True
        lines.append(f"[N{i}] {s['title']}\nURL：{s['url']}\n来源：{s['domain']}；发布时间：{s['published_date'] or '未提供，不能确认同期'}；"
                     f"标题所示统计期：{s.get('data_period_from_title') or '未确认'}\n"
                     f"Tavily 片段（非企业事实）：{s['content'][:500]}")
    lines.append(result['limitation'])
    return '\n'.join(lines)
