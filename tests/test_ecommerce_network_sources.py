"""Offline contract checks for the Phase D acceptance-only source policy."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('phase_d_sources',Path(__file__).parent/'integration/ecommerce_network_sources.py')
sources=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sources)


class NetworkSourcePolicyTests(unittest.TestCase):
    def payload(self,**changes):
        source={'title':'化妆品公开报告','url':'https://www.stats.gov.cn/report',
                'content':'消费品零售趋势片段。','published_date':'2026-08-15'}
        source.update(changes)
        return {'results':[source]}

    def test_retains_actual_fields(self):
        r=sources.normalize_sources(self.payload())
        s=r['sources'][0]
        self.assertEqual(s['domain'],'www.stats.gov.cn')
        self.assertEqual(s['content'],'消费品零售趋势片段。')
        self.assertEqual(s['period'],'august_2026')

    def test_unknown_date_not_guessed_from_url_or_snippet(self):
        r=sources.normalize_sources(self.payload(published_date=None,content='2026年8月市场增长'))
        self.assertEqual(r['sources'][0]['period'],'unknown')
        self.assertIn('未找到足够同期公开证据',sources.format_network(r))

    def test_future_news_not_august_evidence(self):
        r=sources.normalize_sources(self.payload(published_date='2026-09-15'))
        self.assertFalse(r['same_period_evidence'])
        self.assertIn('不能作为八月业绩变化证据',sources.format_network(r))

    def test_rfc_date(self):
        self.assertEqual(sources.publication({'published_date':'Mon, 10 Aug 2026 12:00:00 GMT'})[1],'august_2026')

    def test_dateline_in_search_content_is_preserved(self):
        result=sources.normalize_sources(self.payload(published_date=None,
            content='美妆市场\n###### 2026年09月16日 19:56   媒体\n消费品报道'))
        source=result['sources'][0]
        self.assertEqual(source['published_date'],'2026-09-16T19:56')
        self.assertEqual(source['date_basis'],'tavily_content_dateline')
        self.assertEqual(source['period'],'outside_august_2026')

    def test_august_publication_not_august_statistics(self):
        result=sources.normalize_sources(self.payload(title='2026年1—7月份消费品零售',
            published_date=None,content='2026/08/17 15:00\n消费品统计'))
        self.assertEqual(result['sources'][0]['data_period_from_title'],'2026-01..2026-07')
        self.assertEqual(result['sources'][0]['period'],'august_2026')
        self.assertFalse(result['same_period_evidence'])

    def test_unparseable_date(self):
        self.assertEqual(sources.publication({'published_date':'yesterday'})[1],'unverified_date')

    def test_error_distinct_from_empty(self):
        failure=sources.normalize_sources({'results':[],'error':{'type':'TimeoutError'}})
        empty=sources.normalize_sources({'results':[]})
        self.assertEqual(failure['status'],'unavailable')
        self.assertEqual(empty['status'],'empty')

    def test_rejects_private_and_credential_urls(self):
        for url in ('http://localhost/x','http://127.0.0.1/x','http://10.0.0.2/x','file:///x',
                    'https://key:secret@example.org/x','https://example.org:3307/x'):
            self.assertIsNone(sources.domain(url))

    def test_irrelevant_or_low_quality_not_forced_into_answer(self):
        for changes in ({'url':'https://seo.example.org/x'}, {'title':'Weather','content':'Sunny today'}):
            result=sources.normalize_sources(self.payload(**changes))
            self.assertEqual(result['status'],'empty')
            self.assertNotIn('[N1]',sources.format_network(result))

    def test_network_numbering_not_internal_citation(self):
        text=sources.format_network(sources.normalize_sources(self.payload()))
        self.assertIn('[N1]',text)
        self.assertIn('https://www.stats.gov.cn/report',text)
        self.assertNotIn('[C1]',text)

    def test_missing_required_fields_not_valid_source(self):
        for changes in ({'title':''},{'content':''},{'url':None}):
            self.assertEqual(sources.normalize_sources(self.payload(**changes))['sources'],[])

    def test_proxy_fake_ip_only_for_reviewed_public_hosts(self):
        resolved=[(2,1,6,'',('198.18.0.9',443))]
        with patch.object(sources.socket,'getaddrinfo',return_value=resolved):
            self.assertEqual(sources.public_target('https://www.stats.gov.cn/report'),'www.stats.gov.cn')
            with self.assertRaises(ValueError):
                sources.public_target('https://unreviewed.example.org/report')

    def test_trusted_name_cannot_resolve_to_local_service(self):
        for ip in ('127.0.0.1','10.0.0.2','169.254.169.254'):
            with patch.object(sources.socket,'getaddrinfo',return_value=[(2,1,6,'',(ip,443))]):
                with self.assertRaises(ValueError):
                    sources.public_target('https://www.stats.gov.cn/report')
