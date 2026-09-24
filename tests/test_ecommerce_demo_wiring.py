"""Minimal offline checks for the Demo API and persistent profile."""
import contextlib
import io
import os
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from local_rag_test_support import OfflineDocumentTestCase
from test_main_agent_local_rag import run_fixture_coroutine
from app.api.context import get_thread_context
from scripts.start_ecommerce_demo import configure

with contextlib.redirect_stdout(io.StringIO()):
    from app.api import server


class EcommerceDemoWiringTests(OfflineDocumentTestCase):
    def test_kb_list_is_session_scoped_and_restores_context(self):
        kb = self.storage.create_knowledge_base('Demo SOP')
        with patch.object(server, 'LocalRAGStorage', return_value=self.storage):
            own = run_fixture_coroutine(server.list_knowledge_bases('session-A'))
            foreign = run_fixture_coroutine(server.list_knowledge_bases('session-B'))
        self.assertEqual(own['knowledge_bases'][0]['knowledge_base_id'], kb.knowledge_base_id)
        self.assertEqual(own['knowledge_bases'][0]['document_count'], 0)
        self.assertEqual(foreign['knowledge_bases'], [])
        self.assertEqual(get_thread_context(), 'session-A')

    def test_kb_list_rejects_invalid_thread_before_storage(self):
        with patch.object(server, 'LocalRAGStorage') as storage:
            with self.assertRaises(HTTPException):
                run_fixture_coroutine(server.list_knowledge_bases('../escape'))
            storage.assert_not_called()

    def test_api_passes_selected_id_to_runtime(self):
        kb = self.storage.create_knowledge_base('Demo SOP')
        runtime = AsyncMock()

        async def run():
            response = await server.run_task(server.TaskRequest(
                query='库存风险', thread_id='session-A', knowledge_base_id=kb.knowledge_base_id))
            await server.active_tasks['session-A']
            return response

        with (patch.object(server, 'validate_selected_knowledge_base', return_value=kb.knowledge_base_id),
              patch.object(server, 'run_deep_agent', runtime)):
            self.assertEqual(run_fixture_coroutine(run())['status'], 'started')
        runtime.assert_awaited_once_with('库存风险', 'session-A', knowledge_base_id=kb.knowledge_base_id)

    def test_profile_overrides_only_database_environment(self):
        profile = self.base / 'demo.env'
        profile.write_text('MYSQL_HOST=localhost\nMYSQL_PORT=3307\nMYSQL_USER=ecommerce_ro\n'
                           'MYSQL_PASSWORD=fixture\nMYSQL_DATABASE=insight_ecommerce_db\n'
                           'OPENAI_API_KEY=must-not-override\n', encoding='utf-8')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'original'}):
            configure(profile)
            self.assertEqual(os.environ['MYSQL_DATABASE'], 'insight_ecommerce_db')
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'original')

    def test_profile_rejects_wrong_database_account(self):
        profile = self.base / 'demo.env'
        profile.write_text('MYSQL_HOST=localhost\nMYSQL_PORT=3307\nMYSQL_USER=root\n'
                           'MYSQL_PASSWORD=fixture\nMYSQL_DATABASE=insight_ecommerce_db\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            configure(profile)
