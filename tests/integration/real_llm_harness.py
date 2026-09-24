"""Test-only isolation, observation and budgets. Import has no service effects."""
import asyncio
import contextlib
from contextvars import ContextVar
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import re
from threading import RLock
import time
from unittest.mock import patch
from uuid import uuid4


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def public_value(value):
    """Never stringify runtime/client objects (their repr includes state/secrets)."""
    if isinstance(value, dict):
        return {str(k): public_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'model_dump'):
        return public_value(value.model_dump(mode='json'))
    # LangGraph Command is not a Pydantic model. Only retain its public update.
    if type(value).__name__ == 'Command':
        return {'update': public_value(value.update)}
    return {'unserialized_type': type(value).__name__}


class Redactor:
    def __init__(self, secrets=()):
        self.secrets = sorted({s for s in secrets if isinstance(s, str) and s}, key=len, reverse=True)

    def text(self, value):
        for secret in self.secrets:
            value = value.replace(secret, '[REDACTED]')
        return re.sub(r'(?i)(Bearer\s+)[^\s\"\']+', r'\1[REDACTED]', value)

    def __call__(self, value):
        value = public_value(value)
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self(v) for v in value]
        if isinstance(value, dict):
            return {k: '[REDACTED]' if re.search(r'(?i)password|api[_-]?key|authorization|secret', k)
                    else self(v) for k, v in value.items()}
        return value

    @contextlib.contextmanager
    def logs(self):
        """Sanitize records before existing logging handlers see them."""
        factory = logging.getLogRecordFactory()
        def safe_record(*args, **kwargs):
            record = factory(*args, **kwargs)
            record.msg, record.args = self.text(record.getMessage()), ()
            if record.exc_info:
                record.msg += '\n' + self.text(logging.Formatter().formatException(record.exc_info))
                record.exc_info = None
                record.exc_text = None
            return record
        logging.setLogRecordFactory(safe_record)
        try:
            yield
        finally:
            logging.setLogRecordFactory(factory)


class AcceptanceBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    llm_requests: int = 8
    tool_calls: int = 12
    seconds: float = 180


class Budget:
    def __init__(self, limits=Limits(), clock=time.monotonic):
        self.limits, self.clock = limits, clock
        self.started = clock()
        self.counts = {'llm_requests': 0, 'tool_calls': 0}
        self.failure = None
        self.on_exceeded = None
        self.lock = RLock()

    def fail(self, reason):
        with self.lock:
            if self.failure is None:
                self.failure = {'category': 'acceptance_budget_exceeded', 'reason': reason,
                                'timestamp': timestamp()}
                if self.on_exceeded:
                    self.on_exceeded()
            raise AcceptanceBudgetExceeded('acceptance_budget_exceeded: ' + self.failure['reason'])

    def check(self):
        with self.lock:
            if self.failure:
                self.fail(self.failure['reason'])
            if self.clock() - self.started >= self.limits.seconds:
                self.fail('case wall-clock deadline reached')

    def consume(self, kind):
        with self.lock:
            self.check()
            if self.counts[kind] >= getattr(self.limits, kind):
                self.fail(kind + ' limit reached')
            self.counts[kind] += 1


KB_CASES = {'knowledge', 'knowledge_selection', 'no_knowledge', 'dual', 'causality',
            'network_failure', *(f'core_{i}' for i in range(1, 6))}


class CaseState:
    """One case owns one UUID session and separate ingestion/runtime directories."""
    def __init__(self, root, name):
        self.name = name
        self.session = 'phase-e-' + uuid4().hex
        self.base = Path(root).resolve() / self.session
        self.base.mkdir(parents=True, exist_ok=False)
        self.runtime_root = self.base / 'runtime'
        self.ingestion_root = self.base / 'ingestion'
        self.store_root = self.base / 'store'
        self.upload_dir.mkdir(parents=True)
        self.kb_id = None
        self.requires_kb = name in KB_CASES
        self.trace = []
        self.messages = []

    @property
    def upload_dir(self):
        return self.runtime_root / 'updated' / ('session_' + self.session)

    @property
    def ingestion_dir(self):
        return self.ingestion_root / ('session_' + self.session)

    def preflight(self, *, db, user, provider, validate_kb=None):
        names = sorted(p.name for p in self.upload_dir.iterdir())
        output_dir = self.runtime_root / 'output' / ('session_' + self.session)
        errors = []
        if names:
            errors.append('Unexpected uploaded files')
        if output_dir.exists() and any(output_dir.iterdir()):
            errors.append('Unexpected prior output state')
        if self.trace or self.messages:
            errors.append('Nonempty prior trace/message state')
        if self.requires_kb:
            if not self.kb_id or validate_kb is None:
                errors.append('Required case-local KB missing')
            else:
                try:
                    validate_kb(self.kb_id)
                except Exception:
                    errors.append('Case-local KB validation failed')
        elif self.kb_id is not None:
            errors.append('Unexpected selected KB')
        if (db, user) != ('insight_ecommerce_db', 'ecommerce_ro'):
            errors.append('Unexpected DB target/account')
        if provider != 'https://api.deepseek.com':
            errors.append('Unexpected LLM provider')
        result = {'case_name': self.name, 'session_id': self.session,
                  'selected_knowledge_base_id': self.kb_id,
                  'uploaded_file_count': len(names), 'uploaded_file_names': names,
                  'db_target': db, 'db_user': user, 'llm_provider': provider,
                  # This is an expectation, NOT tool removal or a forced route.
                  'network': 'enabled' if self.name == 'network' or self.name.startswith('core_')
                  else 'enabled_with_injected_timeout' if self.name == 'network_failure' else 'not required',
                  'passed': not errors, 'errors': errors}
        return result


class ToolTrace:
    """Observe run/arun; ContextVars preserve delegation through thread pools."""
    def __init__(self, events, budget, redactor=None, save=lambda: None):
        self.events, self.budget = events, budget
        self.redact = redactor or Redactor()
        self.save = save
        self.agent = ContextVar('phase_e_agent_' + uuid4().hex, default='Main Agent')
        self.parent = ContextVar('phase_e_parent_' + uuid4().hex, default=None)
        self.active = ContextVar('phase_e_active_' + uuid4().hex, default=None)
        self.lock = RLock()

    def begin(self, tool, inputs, kwargs):
        raw_id = kwargs.get('tool_call_id')
        runtime = inputs.get('runtime') if isinstance(inputs, dict) else None
        raw_id = raw_id or getattr(runtime, 'tool_call_id', None)
        call_id = raw_id or 'harness-' + uuid4().hex
        arguments = {k: v for k, v in inputs.items()
                     if k not in {'runtime', 'tool_runtime', 'callbacks', 'config'}} if isinstance(inputs, dict) else inputs
        event = {'timestamp': timestamp(), 'started_at': timestamp(),
                 'agent_name': self.agent.get(), 'tool_name': tool.name, 'name': tool.name,
                 'call_id': call_id, 'call_id_origin': 'model' if raw_id else 'harness',
                 'parent_call_id': self.parent.get(), 'arguments': self.redact(arguments),
                 'input': self.redact(arguments), 'result_summary': None, 'status': 'pending',
                 'success': None}
        if tool.name == 'execute_sql_query':
            query = inputs.get('query', '')
            event['sql'] = {'original': self.redact(query),
                            'is_select': bool(re.match(r'^\s*SELECT\b', query, re.I)),
                            'returned_rows': None, 'database_user': None, 'database_name': None}
        if tool.name == 'search_local_knowledge_base':
            event['rag'] = {'query': event['arguments'].get('query'),
                            'knowledge_base_id': event['arguments'].get('knowledge_base_id'), 'citations': []}
        if tool.name == 'internet_search':
            event['network'] = {'query': event['arguments'].get('query'), 'source_urls': []}
        with self.lock:
            self.events.append(event)
        try:
            self.budget.consume('tool_calls')
            if tool.name not in {'task', 'write_todos', 'list_sql_tables', 'get_table_data',
                                 'execute_sql_query', 'search_local_knowledge_base', 'internet_search'}:
                raise RuntimeError('Unnecessary file/other capability blocked by acceptance scope')
        except BaseException as exc:
            self.finish(event, error=exc)
            raise
        return event

    def connection(self, kwargs):
        """Record actual configured connection target, never the password."""
        self.budget.check()
        event = self.active.get()
        if event is not None:
            event['database'] = {'user': kwargs.get('user'), 'name': kwargs.get('database')}
            if 'sql' in event:
                event['sql'].update(database_user=kwargs.get('user'), database_name=kwargs.get('database'))

    def finish(self, event, value=None, error=None):
        output = self.redact(value)
        content = output.get('content') if isinstance(output, dict) and 'content' in output else output
        failure = error is not None or (isinstance(output, dict) and
                  (output.get('status') == 'error' or bool(output.get('error'))))
        if isinstance(content, str) and content.startswith(('拒绝执行', '查询出现异常', '表不存在或不允许访问')):
            failure = True
        if isinstance(content, str):
            try:
                decoded = json.loads(content)
            except (ValueError, TypeError):
                decoded = None
            if isinstance(decoded, dict) and decoded.get('error'):
                failure = True
        with self.lock:
            event.update(finished_at=timestamp(), success=not failure,
                         status='failure' if failure else 'success', output=output)
            event['result_summary'] = self.redact(str(content)[:2000]) if content is not None else None
            if error:
                event['error'] = self.redact(type(error).__name__ + ': ' + str(error))
            if 'sql' in event and not failure and isinstance(content, str):
                event['sql']['returned_rows'] = len(list(csv.DictReader(io.StringIO(content))))
            if 'rag' in event and isinstance(output, dict):
                event['rag']['citations'] = [e['citation'] for e in output.get('artifact', []) or []
                                             if isinstance(e, dict) and 'citation' in e]
            if 'network' in event:
                result = output
                if isinstance(content, str):
                    try: result = json.loads(content)
                    except ValueError: result = {}
                if isinstance(result, dict):
                    event['network']['source_urls'] = [x['url'] for x in result.get('results', [])
                                                       if isinstance(x, dict) and 'url' in x]
            self.save()

    @contextlib.contextmanager
    def scope(self, event):
        active = self.active.set(event)
        parent = self.parent.set(event['call_id'])
        agent = self.agent.set(event['arguments'].get('subagent_type', self.agent.get())
                               if event['name'] == 'task' and isinstance(event['arguments'], dict)
                               else self.agent.get())
        try:
            yield
        finally:
            self.agent.reset(agent)
            self.parent.reset(parent)
            self.active.reset(active)

    @contextlib.contextmanager
    def install(self):
        from langchain_core.tools import BaseTool
        original_run, original_arun = BaseTool.run, BaseTool.arun
        collector = self

        def run(tool, tool_input, *args, **kwargs):
            event = collector.begin(tool, tool_input, kwargs)
            with collector.scope(event):
                try:
                    value = original_run(tool, tool_input, *args, **kwargs)
                except BaseException as exc:
                    collector.finish(event, error=exc)
                    raise
                collector.finish(event, value)
                return value

        async def arun(tool, tool_input, *args, **kwargs):
            event = collector.begin(tool, tool_input, kwargs)
            with collector.scope(event):
                try:
                    value = await original_arun(tool, tool_input, *args, **kwargs)
                except BaseException as exc:
                    collector.finish(event, error=exc)
                    raise
                collector.finish(event, value)
                return value

        with patch.object(BaseTool, 'run', new=run), patch.object(BaseTool, 'arun', new=arun):
            yield self


def outcome_dimensions(case):
    """Unknown semantics/source correctness stays null until evidence review."""
    manual = case.get('manual_review', {})
    legacy_invalid = bool(manual.get('harness_limitations'))
    preflight = case.get('preflight', {})
    errors = case.get('errors', [])
    routes = [e.get('input', {}).get('subagent_type') for e in case.get('tools', []) if e['name'] == 'task']
    database = case['name'] == 'database'
    routed_correctly = routes == ['数据库查询助手'] if database else None
    return {'harness_valid': False if legacy_invalid else preflight.get('passed'),
            'planning_pass': False if database and routed_correctly is False else None,
            'routing_pass': routed_correctly,
            'tool_selection_pass': False if any('capability blocked' in (e.get('error') or '')
                                                for e in case.get('tools', [])) else None,
            'tool_execution_pass': None if legacy_invalid else
                (all(e.get('success') is True for e in case.get('tools', [])) if case.get('tools') else None),
            'business_semantics_pass': False if manual.get('database_worker', {}).get('metric_contract_issue') else None,
            'source_isolation_pass': None,
            'final_answer_pass': False if errors or not case.get('final') else None}
