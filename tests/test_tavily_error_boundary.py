"""Offline tests for the existing search tool's bounded failure contract."""
import asyncio
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tavily.errors import (
    BadRequestError, ForbiddenError, InvalidAPIKeyError,
    TimeoutError as TavilyTimeoutError, UsageLimitExceededError,
)

from app.tools import tavily_tool


class TavilyErrorBoundaryTests(unittest.TestCase):
    def invoke(self, *, result=None, error=None):
        with (patch.object(tavily_tool.tavily_client, 'search',
                           return_value=result, side_effect=error) as search,
              patch.object(tavily_tool.monitor, 'report_tool')):
            output = tavily_tool.internet_search.invoke({'query': 'public market trends'})
            self.assertEqual(search.call_count, 1)
            return output

    def check_error(self, error):
        output = self.invoke(error=error)
        self.assertEqual(output['results'], [])
        self.assertIn('error', output)
        self.assertNotIn('SECRET', str(output))
        return output

    def test_success_preserves_source_fields(self):
        value = {'results': [{'title': 'Report', 'url': 'https://example.org/report',
                 'content': 'Public context', 'published_date': '2026-08-10'}]}
        self.assertEqual(self.invoke(result=value), value)

    def test_empty_is_not_failure(self):
        self.assertEqual(self.invoke(result={'results': []}), {'results': []})

    def test_sdk_timeout(self):
        self.check_error(TavilyTimeoutError(60))

    def test_transport_timeout(self):
        self.check_error(Timeout('SECRET'))

    def test_connection_error(self):
        self.check_error(ConnectionError('SECRET'))

    def test_rate_limit(self):
        self.check_error(UsageLimitExceededError('SECRET'))

    def test_authentication_error(self):
        self.check_error(InvalidAPIKeyError('SECRET'))

    def test_forbidden_error(self):
        self.check_error(ForbiddenError('SECRET'))

    def test_bad_request(self):
        self.check_error(BadRequestError('SECRET'))

    def test_http_error(self):
        self.check_error(HTTPError('SECRET'))

    def test_invalid_json(self):
        self.assertEqual(self.check_error(ValueError('SECRET'))['error']['type'], 'InvalidResponse')

    def test_invalid_shape(self):
        for value in (None, [], {}, {'results': None}, {'results': 'bad'}):
            with self.subTest(value=value):
                self.assertEqual(self.invoke(result=value)['error']['type'], 'InvalidResponse')

    def test_programming_error_not_hidden(self):
        with self.assertRaises(TypeError):
            self.invoke(error=TypeError('programming error'))

    def test_compiled_graph_receives_failure_tool_message(self):
        builder = StateGraph(MessagesState)
        builder.add_node('tools', ToolNode([tavily_tool.internet_search]))
        builder.add_edge(START, 'tools')
        builder.add_edge('tools', END)
        message = AIMessage(content='', tool_calls=[{'id': 'timeout',
            'name': 'internet_search', 'args': {'query': 'public trends'}}])
        with (patch.object(tavily_tool.tavily_client, 'search', side_effect=TavilyTimeoutError(60)) as search,
              patch.object(tavily_tool.monitor, 'report_tool')):
            state = asyncio.run(builder.compile().ainvoke({'messages': [message]}))
        self.assertEqual(search.call_count, 1)
        self.assertIn('error', state['messages'][-1].content)
        self.assertEqual(state['messages'][-1].tool_call_id, 'timeout')
