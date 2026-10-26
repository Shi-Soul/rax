"""Qoder credentials, account routing, native events and process isolation."""
from dataclasses import replace
import base64
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from harness import agent_host
from harness.qoder_cli import auth_proxy, auto_research as qoder, spec
from scripts.validate_native_cli import fixture_workspace
from test_controller_backend import configuration


def private_file(path, text):
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_auth_is_required_before_workspace_creation(tmp_path):
    config = replace(configuration(tmp_path/'trial'), agent_cli='qoder', agent_effort='max')
    with pytest.raises(ValueError, match='agent-auth-home'):
        qoder.deploy(config)
    assert not (config.root/'agent-workspace').exists()
    auth = tmp_path/'auth'
    auth.mkdir(mode=0o700)
    token = private_file(auth/'qoder-pat', 'private-pat')
    assert qoder.token_file(auth) == token
    token.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        qoder.token_file(auth)


def test_browser_auth_store_requires_private_credential_files(tmp_path):
    auth = tmp_path/'auth'
    auth.mkdir(mode=0o700)
    store = auth/'.auth'
    store.mkdir(mode=0o700)
    private_file(store/'user', 'encrypted-browser-credential')
    machine = private_file(store/'machine_id', 'fixture-machine-id')
    assert qoder.token_file(auth) is None
    machine.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        qoder.token_file(auth)


@pytest.mark.parametrize('url,method', [
    ('https://openapi.qoder.com.cn/api/v1/jobToken/exchange', 'POST'),
    ('https://openapi.qoder.com.cn/api/v1/deviceToken/refresh', 'POST'),
    ('https://openapi.qoder.com.cn/api/v3/user/status', 'GET'),
    ('https://gateway.qoder.com.cn/api/v2/model/list?Encode=1', 'GET'),
    ('https://gateway.qoder.com.cn/algo/api/v2/model/list?Encode=1', 'GET'),
    ('https://gateway.qoder.com.cn/algo/api/v5/service/region/endpoints', 'GET'),
    ('https://openapi.qoder.com.cn/api/v1/organizations/org-123/tags', 'GET'),
    ('https://openapi.qoder.com.cn/sash/api/v1/organizations/org-123/mcp-governance/policy', 'GET'),
    ('https://openapi.qoder.com.cn/api/v1/qcs/config/resolve', 'POST'),
])
def test_account_routes(url, method):
    assert auth_proxy.target(url, method).scheme == 'https'


@pytest.mark.parametrize('url,method', [
    ('http://openapi.qoder.com.cn/api/v1/userinfo', 'GET'),
    ('https://openapi.qoder.com.cn.evil.invalid/api/v1/userinfo', 'GET'),
    ('https://secret@openapi.qoder.com.cn/api/v1/userinfo', 'GET'),
    ('https://openapi.qoder.com.cn:8443/api/v1/userinfo', 'GET'),
    ('https://openapi.qoder.com.cn/api/v1/userinfo#fragment', 'GET'),
    ('https://gateway.qoder.com.cn/api/v2/service/ask', 'POST'),
    ('https://openapi.qoder.com.cn/api/v2/config/updateDataPolicy', 'POST'),
    ('https://openapi.qoder.com.cn/api/v1/userinfo', 'DELETE'),
    ('https://openapi.qoder.sh/api/v1/userinfo', 'GET'),
])
def test_account_connection_cannot_make_model_calls_or_arbitrary_requests(url, method):
    with pytest.raises(ValueError, match='authentication and account metadata'):
        auth_proxy.target(url, method)


def test_pat_is_inserted_only_by_host_proxy(tmp_path, monkeypatch):
    sent = []
    def upstream(url, method, headers, body, timeout):
        assert urlsplit(url).hostname == 'openapi.qoder.com.cn'
        sent.append(json.loads(body))
        return SimpleNamespace(status=200, headers={'Content-Type': 'application/json'},
                               read=lambda size: b'{"token":"issued-session-token"}', close=lambda: None)
    monkeypatch.setattr(auth_proxy, 'open_request', upstream)
    token = private_file(tmp_path/'qoder-pat', 'operator-private-pat')
    path = tmp_path/'auth.sock'
    with auth_proxy.serve(path, token):
        connection = http.client.HTTPConnection('localhost')
        connection.sock = socket.socket(socket.AF_UNIX)
        connection.sock.connect(str(path))
        connection.request('POST', '/qoder', json.dumps({
            'url': 'https://openapi.qoder.com.cn/api/v1/jobToken/exchange', 'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'personal_token': 'rax-session-transport'})}))
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b'{"token":"issued-session-token"}'
        connection.close()
    assert sent == [{'personal_token': 'operator-private-pat'}]
    assert not path.exists()


def test_failed_result_is_not_treated_as_success(tmp_path):
    qoder.capture_event(tmp_path, json.dumps({'type': 'result', 'subtype': 'success',
        'is_error': True, 'result': 'Not logged in', 'session_id': 'session-1',
        'usage': {'input_tokens': 0}}))
    assert qoder.latest_session(tmp_path) == 'session-1'
    assert qoder.last_error(tmp_path) == 'Not logged in'
    qoder.capture_event(tmp_path, 'not-json')
    assert spec.arguments('continue', 'session-1')[-4:] == ['--resume', 'session-1', '--print', 'continue']


def test_builtin_inference_preserves_encoded_body_and_stream(tmp_path, monkeypatch):
    url = 'https://gateway.qoder.com.cn/algo/api/v2/service/pro/sse/agent_chat_generation'
    with pytest.raises(ValueError):
        auth_proxy.target(url, 'POST')
    assert auth_proxy.target(url, 'POST', inference=True).hostname == 'gateway.qoder.com.cn'
    encoded = b'\x1f\x8b\x00\xffencoded-vendor-request'
    stream = b'data: {"text":"READY"}\n\ndata: [DONE]\n\n'
    def upstream(target, method, headers, body, timeout):
        assert (target, method, body, timeout) == (url, 'POST', encoded, 900)
        response = io.BytesIO(stream)
        response.status = 200
        response.headers = {'Content-Type': 'text/event-stream', 'X-Request-Id': 'fixture-request'}
        return response
    monkeypatch.setattr(auth_proxy, 'open_request', upstream)
    path = tmp_path/'auth.sock'
    with auth_proxy.serve(path, None, inference=True):
        connection = http.client.HTTPConnection('localhost')
        connection.sock = socket.socket(socket.AF_UNIX)
        connection.sock.connect(str(path))
        connection.request('POST', '/qoder', json.dumps({'url': url, 'method': 'POST',
            'headers': {'Content-Type': 'application/octet-stream'}, 'bodyEncoding': 'base64',
            'body': base64.b64encode(encoded).decode()}))
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader('X-Request-Id') == 'fixture-request'
        assert response.read() == stream
        connection.close()


def test_real_qoder_login_reaches_account_relay(tmp_path, monkeypatch):
    binary = spec.default_binary()
    if not binary.is_file() or not shutil.which('bwrap') or not shutil.which('node'):
        pytest.skip('Pinned Qoder CLI, node and bubblewrap are required')
    sent = []
    def reject_login(url, method, headers, body, timeout):
        parsed = urlsplit(url)
        sent.append((parsed.hostname, method, parsed.path, json.loads(body)))
        return SimpleNamespace(status=401, headers={'Content-Type': 'application/json'},
                               read=lambda size: b'{"error":"invalid fixture credential"}', close=lambda: None)
    monkeypatch.setattr(auth_proxy, 'open_request', reject_login)
    config = replace(configuration(tmp_path/'trial'), agent_cli='qoder', agent_effort='max')
    auth = tmp_path/'auth'
    auth.mkdir(mode=0o700)
    private_file(auth/'qoder-pat', 'fixture-qoder-pat')
    key = private_file(tmp_path/'provider.key', 'fixture-model-key')
    args = SimpleNamespace(backend='qoder', binary=binary, provider='openrouter',
                           model=spec.MODEL, key_file=key, auth_home=auth, base_url=None)
    deployed = fixture_workspace(qoder, config, args)
    env = qoder.agent_environment(config, deployed, agent_host.socket_path(config.root))
    with qoder.relay(config, deployed):
        result = subprocess.run(qoder.agent_command(config, deployed, prompt='Reply READY'),
                                env=env, capture_output=True, text=True, timeout=45)
    assert sent, result.stderr[-2000:]
    assert sent[0][:3] == ('openapi.qoder.com.cn', 'POST', '/api/v1/jobToken/exchange')
    assert sent[0][3]['personal_token'] == 'fixture-qoder-pat'
    assert result.returncode != 0
    assert 'Network attempt failed at dns' not in result.stderr


def test_changed_qoder_bundle_is_rejected_before_execution(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    entry = tmp_path/'changed.mjs'
    entry.write_text('console.log("MUST_NOT_RUN")')
    transport = Path(qoder.__file__).with_name('transport.mjs')
    result = subprocess.run([node, '--import', str(transport), str(entry)],
                            env={**os.environ, 'RAX_QODER_ENTRY': str(entry)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'unsupported Qoder bundle' in result.stderr
    assert 'MUST_NOT_RUN' not in result.stdout


def test_embedded_data_url_does_not_require_a_network_relay(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    transport = Path(qoder.__file__).with_name('transport.mjs')
    script = '''
import assert from 'node:assert/strict';
const response = await fetch('data:application/octet-stream;base64,AGFzbQEAAAA=');
assert.equal(WebAssembly.validate(await response.arrayBuffer()), true);
'''
    subprocess.run([node, '--import', str(transport), '--input-type=module', '-e', script],
                   env={**os.environ, 'RAX_QODER_ENTRY': str(tmp_path/'unused-entry.mjs')},
                   capture_output=True, text=True, timeout=10, check=True)


def test_builtin_request_rejects_model_or_effort_changes(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required')
    audit = tmp_path/'request-audit.jsonl'
    script = '''
import assert from 'node:assert/strict';
const inspect = globalThis[Symbol.for('rax.qoder.audit')];
const body = {model_config: {display_name: 'DeepSeek-Flash', key: 'dfmodel'},
  parameters: {reasoning_effort: 'max', enable_thinking: true},
  messages: [{reasoning_content: 'PRIVATE_TEXT', content: [{type: 'image_url', image_url: {url: 'PRIVATE_IMAGE'}}]}]};
inspect(JSON.stringify(body));
body.parameters.reasoning_effort = 'low';
assert.throws(() => inspect(JSON.stringify(body)), /configured model or reasoning effort/);
body.parameters.reasoning_effort = 'max';
body.model_config.display_name = 'Auto';
assert.throws(() => inspect(JSON.stringify(body)), /configured model or reasoning effort/);
'''
    subprocess.run([node, '--import', str(Path(qoder.__file__).with_name('transport.mjs')),
                    '--input-type=module', '-e', script], env={**os.environ,
                        'RAX_QODER_ENTRY': str(tmp_path/'unused.mjs'), 'RAX_QODER_AUDIT': str(audit),
                        'RAX_QODER_MODEL': 'DeepSeek-Flash', 'RAX_QODER_EFFORT': 'max'},
                   capture_output=True, text=True, timeout=10, check=True)
    content = audit.read_text()
    assert 'PRIVATE_' not in content
    assert len(content.splitlines()) == 1
    assert json.loads(content)['image_blocks'] == 1


@pytest.mark.parametrize('browser_login,provider', [(False, 'openrouter'), (True, 'openrouter'), (True, spec.ACCOUNT)])
def test_real_qoder_binary_and_sandbox(tmp_path, browser_login, provider):
    binary = spec.default_binary()
    if not binary.is_file() or not shutil.which('bwrap') or not shutil.which('node'):
        pytest.skip('Pinned Qoder CLI, node and bubblewrap are required')
    config = replace(configuration(tmp_path/'trial'), agent_cli='qoder', agent_effort='max', agent_provider=provider)
    auth = tmp_path/'auth'
    auth.mkdir(mode=0o700)
    if browser_login:
        (auth/'.auth').mkdir(mode=0o700)
        private_file(auth/'.auth/user', 'encrypted-browser-credential')
        private_file(auth/'.auth/machine_id', 'fixture-machine-id')
    else:
        private_file(auth/'qoder-pat', 'private-qoder-pat')
    key = private_file(tmp_path/'provider.key', 'private-model-key')
    model = spec.model(provider)
    args = SimpleNamespace(backend='qoder', binary=binary, provider=provider,
                           model=model, key_file=None if provider == spec.ACCOUNT else key, auth_home=auth, base_url=None)
    deployed = fixture_workspace(qoder, config, args)
    sock = agent_host.socket_path(config.root)
    environment = qoder.agent_environment(config, deployed, sock)
    assert 'private-qoder-pat' not in json.dumps(environment)
    assert 'private-model-key' not in json.dumps(environment)
    assert ('QODERCN_PERSONAL_ACCESS_TOKEN' not in environment) == browser_login
    settings = json.loads((deployed['qoder_home']/'settings.json').read_text())
    selected = model if provider == spec.ACCOUNT else 'rax/'+model
    assert settings['model']['name'] == selected
    assert ('providers' not in settings) == (provider == spec.ACCOUNT)
    assert settings['model']['reasoningEffort'] == 'max'
    command = qoder.agent_command(config, deployed, prompt='unused')
    assert command[command.index('--mcp-config')+1] == str(deployed['qoder_home']/'mcp.json')
    assert command[command.index('--model')+1] == selected
    assert command[command.index('--reasoning-effort')+1] == 'max'
    # --help validates the actual vendor's command-line options without making a model call.
    result = subprocess.run([*command, '--help'], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-1000:]
    assert '--reasoning-effort' in result.stdout
    protected = [str(key), str(auth/'.auth/user' if browser_login else auth/'qoder-pat'),
                 str(agent_host.PROJECT/'README.md')]
    program = f'''
from pathlib import Path
import socket
for name in {protected!r}:
    assert not Path(name).is_file(), name
assert not Path('/dev/nvidia0').exists()
s = socket.socket(); s.settimeout(1)
assert s.connect_ex(('1.1.1.1', 443)) != 0
s.close()
try:
    Path({str(Path(spec.__file__))!r}).write_text('changed')
except OSError:
    pass
else:
    raise AssertionError('Backend source is writable')
Path('output/isolated.txt').write_text('OK')
'''
    if browser_login:
        program += f'''
store = Path({str(deployed['qoder_home']/'.auth')!r})
assert (store/'user').read_text() == 'encrypted-browser-credential'
(store/'user').write_text('refreshed-browser-credential')
'''
    check = qoder.sandbox(config, deployed, [str(agent_host.agent_python()), '-I', '-c', program])
    subprocess.run(check, env=environment, capture_output=True, text=True, timeout=15, check=True)
    assert (deployed['workspace']/'output/isolated.txt').read_text() == 'OK'
    if browser_login:
        assert (auth/'.auth/user').read_text() == 'refreshed-browser-credential'
