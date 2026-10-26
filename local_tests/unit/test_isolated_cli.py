"""Security and lifecycle contracts shared by optional native coding agents."""
from dataclasses import replace
import importlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from test_controller_backend import configuration
from harness.backends import NAMES, PROVIDERS, get
from harness.isolated_cli.credentials import Credential
from harness.isolated_cli.relay import image_count, open_request, prepare_request


@pytest.mark.parametrize('status', [200, 401, 302])
def test_host_transport_uses_proxy_preserves_errors_and_rejects_redirects(monkeypatch, status):
    sent = []
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            sent.append((self.path, self.headers['Authorization'],
                         self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            if status == 302:
                self.send_header('Location', 'http://other.invalid/credential-leak')
            self.end_headers()
            self.wfile.write(b'{"fixture":true}')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('http_proxy', f'http://127.0.0.1:{server.server_port}')
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('NO_PROXY', '')
    try:
        with open_request('http://upstream.invalid/v1/chat/completions', 'POST',
                          {'Authorization': 'Bearer fixture-key'}, b'{}', 5) as response:
            assert response.status == status
            assert response.read1(1024) == b'{"fixture":true}'
        assert sent == [('http://upstream.invalid/v1/chat/completions', 'Bearer fixture-key', b'{}')]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

@pytest.fixture(params=('kimi',))
def backend(request):
    name = request.param
    return name, get(name), importlib.import_module('harness.'+name+'_cli.spec')


def key(tmp_path):
    path = tmp_path/'provider.key'
    path.write_text('private-fixture-credential')
    path.chmod(0o600)
    return path


def test_optional_backend_preserves_existing_lazy_imports():
    script = '''
import importlib.abc,sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('harness.kimi_cli','harness.deepseek_cli')):
            raise ImportError('optional backend disabled')
sys.meta_path.insert(0,Reject())
from harness.backends import get
assert get('codex').HOME_KEY == 'codex_home'
assert get('claude').HOME_KEY == 'claude_home'
'''
    subprocess.run([sys.executable, '-c', script], check=True)


def test_provider_choices_and_unsupported_modes_fail_early(tmp_path, backend):
    BACKEND, cli, spec = backend
    base = replace(configuration(tmp_path), agent_cli=BACKEND, agent_effort='max')
    assert PROVIDERS[BACKEND] == spec.PROVIDERS
    for provider in spec.PROVIDERS:
        assert replace(base, agent_provider=provider).agent_provider == provider
    for updates in ({'agent_formal': True}, {'subagents': 1}, {'exploration_envs': 2}):
        with pytest.raises(ValueError, match='currently supports code formal'):
            replace(base, **updates)
    with pytest.raises(ValueError, match='agent_provider'):
        replace(base, agent_provider='claude-login')


def test_api_key_modes_use_private_files_not_configuration(tmp_path, backend):
    BACKEND, cli, spec = backend
    path = key(tmp_path)
    for provider in (p for p in spec.PROVIDERS if p not in getattr(spec, 'ACCOUNTS', (spec.ACCOUNT,))):
        credential = Credential(spec, provider, key_file=path,
            base_url='http://127.0.0.1:8123/v1' if provider == 'custom' else None)
        assert credential.headers()['Authorization'] == 'Bearer '+path.read_text()
        assert path.read_text() not in json.dumps(credential.record())
    path.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        Credential(spec, 'openrouter', key_file=path)


@pytest.mark.parametrize('url', ['https://user:secret@host/v1', 'file:///tmp/key',
                                      'https://host/v1?api_key=secret', 'https://host/v1#secret'])
def test_endpoint_cannot_embed_credentials(tmp_path, url, backend):
    _, _, spec = backend
    with pytest.raises(ValueError, match='without embedded credentials'):
        Credential(spec, 'custom', key_file=key(tmp_path), base_url=url)


def test_account_ignores_default_api_key_and_requires_private_auth_home(tmp_path, backend):
    BACKEND, cli, spec = backend
    auth = tmp_path/'auth'
    auth.mkdir(mode=0o700)
    credential = Credential(spec, spec.ACCOUNT, auth_home=auth, key_file=tmp_path/'unused.key')
    assert credential.key_file is None and credential.record()['key_file'] is None
    auth.chmod(0o755)
    with pytest.raises(ValueError, match='0700'):
        Credential(spec, spec.ACCOUNT, auth_home=auth)


def test_relay_keeps_images_tools_and_reasoning_history_and_enforces_model(backend):
    _, _, spec = backend
    messages = [{'role': 'tool', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}]},
                {'role': 'assistant', 'content': '', 'reasoning_content': 'native model state'}]
    incoming = {'model': spec.MODEL, 'messages': messages, 'tools': [{'type': 'function', 'function': {'name': 'robot'}}],
                'reasoning_effort': 'low'}
    forwarded = prepare_request(incoming, spec.MODEL, 'max', openrouter=True)
    assert forwarded['messages'] is messages and forwarded['tools'] == incoming['tools']
    assert forwarded['reasoning'] == {'effort': 'max'} and 'reasoning_effort' not in forwarded
    assert forwarded['provider']['require_parameters'] is True
    assert image_count(forwarded['messages']) == 1
    direct = prepare_request(incoming, spec.MODEL, 'max')
    assert direct['reasoning_effort'] == 'max' and direct['thinking'] == {'type': 'enabled'}
    assert incoming['reasoning_effort'] == 'low'
    with pytest.raises(ValueError, match='Only the configured'):
        prepare_request({**incoming, 'model': 'different-model'}, spec.MODEL, 'max')


def test_vendor_configuration_contains_only_placeholder_credentials(tmp_path, backend):
    BACKEND, cli, spec = backend
    root = tmp_path/'home'
    root.mkdir()
    config = replace(configuration(tmp_path/'session'), agent_cli=BACKEND, agent_effort='max')
    spec.write_home(config, root, spec.MODEL, tmp_path/'mcp.sock')
    text = '\n'.join(p.read_text() for p in root.rglob('*') if p.is_file())
    assert 'rax-session-transport' in text and 'rax-model.invalid' in text
    assert 'max' in text and ('image_in' in text or 'supportsImage' in text)
    assert 'private-fixture-credential' not in text


def test_capture_native_identity_errors_and_usage(tmp_path, backend):
    BACKEND, cli, spec = backend
    event = {'type': 'session.resume_hint', 'session_id': 'native-id'} if BACKEND == 'kimi' else {
        'type': 'result', 'sessionId': 'native-id', 'usage': {'inputTokens': 5}}
    cli.capture_event(tmp_path, json.dumps(event))
    assert cli.latest_session(tmp_path) == 'native-id'
    cli.capture_event(tmp_path, json.dumps({'type': 'error', 'message': 'native error', 'error': 'native error'}))
    assert cli.last_error(tmp_path) == 'native error'
    cli.capture_event(tmp_path, 'not json')
    cli.capture_event(tmp_path, '[]')
