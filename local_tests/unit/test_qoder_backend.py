"""Qoder credentials, account routing, native events and process isolation."""
from dataclasses import replace
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from types import SimpleNamespace

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


@pytest.mark.parametrize('url,method', [
    ('https://openapi.qoder.sh/api/v1/jobToken/exchange', 'POST'),
    ('https://openapi.qoder.sh/api/v3/user/status', 'GET'),
    ('https://api2.qoder.sh/api/v2/model/list?Encode=1', 'GET'),
    ('https://openapi.qoder.sh/api/v1/organizations/org-123/tags', 'GET'),
])
def test_account_routes(url, method):
    assert auth_proxy.target(url, method).scheme == 'https'


@pytest.mark.parametrize('url,method', [
    ('http://openapi.qoder.sh/api/v1/userinfo', 'GET'),
    ('https://openapi.qoder.sh.evil.invalid/api/v1/userinfo', 'GET'),
    ('https://secret@openapi.qoder.sh/api/v1/userinfo', 'GET'),
    ('https://openapi.qoder.sh:8443/api/v1/userinfo', 'GET'),
    ('https://openapi.qoder.sh/api/v1/userinfo#fragment', 'GET'),
    ('https://api2.qoder.sh/api/v2/service/ask', 'POST'),
    ('https://openapi.qoder.sh/api/v2/config/updateDataPolicy', 'POST'),
    ('https://openapi.qoder.sh/api/v1/userinfo', 'DELETE'),
])
def test_account_connection_cannot_make_model_calls_or_arbitrary_requests(url, method):
    with pytest.raises(ValueError, match='authentication and catalog'):
        auth_proxy.target(url, method)


def test_pat_is_inserted_only_by_host_proxy(tmp_path, monkeypatch):
    sent = []
    class Upstream:
        def __init__(self, host, timeout):
            assert host == 'openapi.qoder.sh'
        def request(self, method, path, body, headers):
            sent.append(json.loads(body))
        def getresponse(self):
            return SimpleNamespace(status=200, getheader=lambda *args: 'application/json',
                                   read=lambda size: b'{"token":"issued-session-token"}')
        def close(self):
            pass
    monkeypatch.setattr(auth_proxy.http.client, 'HTTPSConnection', Upstream)
    token = private_file(tmp_path/'qoder-pat', 'operator-private-pat')
    path = tmp_path/'auth.sock'
    with auth_proxy.serve(path, token):
        connection = http.client.HTTPConnection('localhost')
        connection.sock = socket.socket(socket.AF_UNIX)
        connection.sock.connect(str(path))
        connection.request('POST', '/qoder', json.dumps({
            'url': 'https://openapi.qoder.sh/api/v1/jobToken/exchange', 'method': 'POST',
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


def test_real_qoder_login_reaches_account_relay(tmp_path, monkeypatch):
    binary = spec.default_binary()
    if not binary.is_file() or not shutil.which('bwrap') or not shutil.which('node'):
        pytest.skip('Pinned Qoder CLI, node and bubblewrap are required')
    sent = []
    class RejectLogin:
        def __init__(self, host, timeout):
            self.host = host
        def request(self, method, path, body, headers):
            sent.append((self.host, method, path, json.loads(body)))
        def getresponse(self):
            return SimpleNamespace(status=401, getheader=lambda *args: 'application/json',
                                   read=lambda size: b'{"error":"invalid fixture credential"}')
        def close(self):
            pass
    monkeypatch.setattr(auth_proxy.http.client, 'HTTPSConnection', RejectLogin)
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
    assert sent[0][:3] == ('openapi.qoder.sh', 'POST', '/api/v1/jobToken/exchange')
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


def test_real_qoder_binary_and_sandbox(tmp_path):
    binary = spec.default_binary()
    if not binary.is_file() or not shutil.which('bwrap') or not shutil.which('node'):
        pytest.skip('Pinned Qoder CLI, node and bubblewrap are required')
    config = replace(configuration(tmp_path/'trial'), agent_cli='qoder', agent_effort='max')
    auth = tmp_path/'auth'
    auth.mkdir(mode=0o700)
    private_file(auth/'qoder-pat', 'private-qoder-pat')
    key = private_file(tmp_path/'provider.key', 'private-model-key')
    args = SimpleNamespace(backend='qoder', binary=binary, provider='openrouter',
                           model=spec.MODEL, key_file=key, auth_home=auth, base_url=None)
    deployed = fixture_workspace(qoder, config, args)
    sock = agent_host.socket_path(config.root)
    environment = qoder.agent_environment(config, deployed, sock)
    assert 'private-qoder-pat' not in json.dumps(environment)
    assert 'private-model-key' not in json.dumps(environment)
    settings = json.loads((deployed['qoder_home']/'settings.json').read_text())
    assert settings['model']['name'] == 'rax/'+spec.MODEL
    assert settings['model']['reasoningEffort'] == 'max'
    command = qoder.agent_command(config, deployed, prompt='unused')
    assert command[command.index('--mcp-config')+1] == str(deployed['qoder_home']/'mcp.json')
    assert command[command.index('--model')+1] == 'rax/'+spec.MODEL
    assert command[command.index('--reasoning-effort')+1] == 'max'
    # --help validates the actual vendor's command-line options without making a model call.
    result = subprocess.run([*command, '--help'], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-1000:]
    assert '--reasoning-effort' in result.stdout
    protected = [str(key), str(auth/'qoder-pat'), str(agent_host.PROJECT/'README.md')]
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
    check = qoder.sandbox(config, deployed, [str(agent_host.agent_python()), '-I', '-c', program])
    subprocess.run(check, env=environment, capture_output=True, text=True, timeout=15, check=True)
    assert (deployed['workspace']/'output/isolated.txt').read_text() == 'OK'
