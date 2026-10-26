"""Agent frontend permissions, tool contracts, and fixed Codex launcher config."""
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
from dataclasses import replace

import pytest

from test_controller_backend import configuration, supervisor
from services.controller.frontend import ResearchFrontend
from harness import agent_host
from harness.codex_cli.auto_research import CALL_SECONDS, MODEL, deploy, startup_prompt
from bench import STEP, TASK


def data(reply):
    return json.loads(reply['content'][0]['text'])


@pytest.fixture
def frontend(supervisor, tmp_path):
    f = ResearchFrontend(supervisor[0], tmp_path/'agent')
    yield f
    f.close()


def test_frontend_soft_guidance_and_native_images(supervisor, frontend):
    s, backend = supervisor
    f = frontend
    initial = {t['name'] for t in f.tools()}
    assert STEP in initial and 'register' in initial
    assert 'gemini_generate' in initial
    assert not {'operate', 'enter_recorder', 'exit_recorder', 'discard_recorder'} & initial
    assert not {'shell_exec', 'shell_poll', 'shell_stop', 'read_file', 'view_image'} & initial
    assert not any(name.startswith('research_') for name in initial)
    assert 'research_write_file' not in initial and 'train' not in initial
    assert 'private_request_formal_episode' not in initial
    assert not f.status()['manual_success_confirmed']
    bundle = f.workspace/'code/test'
    bundle.mkdir(parents=True)
    (bundle/'controller.py').write_text('def main(ctx): pass')
    first = f.call('start_episode', {})
    assert first['content'][1] == backend.image and 'structuredContent' not in first
    registered = data(f.call('register', {'source': 'code/test'}))
    assert registered['bundle'] in s.state['bundles'] and s.state['active']['mode'] == 'interactive'
    with pytest.raises(RuntimeError, match='rehearsal'):
        f.call('submit', {'bundle': registered['bundle']})
    assert s.state['active']['mode'] == 'interactive'
    backend.success = True
    assert data(f.call('evaluate', {}))['manual_success_confirmed']
    assert initial == {t['name'] for t in f.tools()}


def test_frontend_paths_and_native_recording_bypass_are_rejected(supervisor, tmp_path, frontend):
    s, backend = supervisor
    f = frontend
    for name in ('/etc/passwd', '../private/state.json', 'code/../../private/configuration.json'):
        with pytest.raises((ValueError, PermissionError)):
            f.source(name)
    (f.workspace/'link').symlink_to(tmp_path/'private')
    with pytest.raises(ValueError):
        f.call('register', {'source': 'link'})
    f.call('start_episode', {})
    backend.success = True
    f.call('evaluate', {})
    f.call(STEP, {'actions': [[0]*14]})
    assert backend.step == 1


def test_protocol_errors_do_not_disclose_private_paths(supervisor, tmp_path, frontend):
    f = frontend
    reply = f.handle({'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call',
                     'params': {'name': 'read_file', 'arguments': {'path': 'missing'}}})
    assert reply['id'] == 7 and reply['result']['isError']
    assert str(tmp_path) not in json.dumps(reply)


def test_public_result_preserves_execution_and_evaluation_diagnostics():
    source = {'error_type': 'RuntimeError', 'error': {'reason': 'Container exited early'},
              'evaluation_error': 'TimeoutError', 'evaluation_failure': {'reason': 'Evaluation timed out'},
              'container': 'private-container', 'workspace_path': '/private'}
    result = ResearchFrontend.public_result(source)
    assert result['error'] == source['error'] and result['evaluation_failure'] == source['evaluation_failure']
    assert 'container' not in result and 'workspace_path' not in result


def test_large_inline_image_through_frontend_stdio(frontend):
    import io
    import os
    from PIL import Image
    import base64
    from services.controller.gemini import GeminiRouter, MODEL
    raw = io.BytesIO()
    Image.frombytes('RGB', (768, 768), os.urandom(768*768*3)).save(raw, format='PNG')
    block = {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(raw.getvalue()).decode()}
    captured = []
    def send(payload):
        captured.append(payload)
        return {'model': MODEL, 'choices': [{'message': {'content': 'seen'}}],
                'usage': {'total_tokens': 10}}
    frontend.api.gemini = GeminiRouter('host-only-test-key', send=send)
    request = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': 'gemini_generate', 'arguments': {
            'messages': [{'role': 'user', 'content': [block]}]}}}) + '\n'
    assert len(request) > 2*1024*1024
    output = io.StringIO()
    frontend.serve(io.StringIO(request), output)
    result = json.loads(output.getvalue())['result']
    assert not result.get('isError'), result
    assert data(result)['usage']['total_tokens'] == 10
    assert captured[0]['messages'][0]['content'][0]['image_url']['url'].endswith(block['data'])


def test_frontend_oversized_wire_request_closes_stream(frontend, monkeypatch):
    import io
    monkeypatch.setattr('services.controller.frontend.MAX_REQUEST_BYTES', 16)
    output = io.StringIO()
    frontend.serve(io.StringIO('x'*17 + '\n' + '{}\n'), output)
    assert len(output.getvalue().splitlines()) == 1
    assert json.loads(output.getvalue())['error']['message'] == 'Request exceeds wire limit'


@pytest.fixture
def demo_cache(tmp_path, monkeypatch):
    """Deterministic video decoder; real JPEG export, no OpenCV/network dependency."""
    import sys
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    cache = tmp_path/'runtime/reference-demos/website'
    cache.mkdir(parents=True)
    for task in (TASK,):
        (cache/f'{task}.mp4').write_bytes(b'\x00\x00\x00\x18ftypisom')
    class Capture:
        def __init__(self, _):
            self.index = 0
        def isOpened(self):
            return True
        def read(self):
            self.index += 1
            return (True, np.full((8, 12, 3), self.index*50, dtype=np.uint8)) if self.index <= 3 else (False, None)
        def release(self):
            pass
    def write(path, frame, _):
        Image.fromarray(frame).save(path, format='JPEG')
        return True
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(VideoCapture=Capture, imwrite=write, IMWRITE_JPEG_QUALITY=1))
    import urllib.request
    def no_network(*a, **k):
        pytest.fail('Cached demonstration must not require network')
    monkeypatch.setattr(urllib.request, 'urlopen', no_network)
    return cache


def test_deploy_copies_skills_rubric_and_confines_the_agent(tmp_path, monkeypatch, demo_cache):
    kind, task = 'none', TASK
    provider = tmp_path/'provider'
    provider.mkdir()
    (provider/'openrouter.config.toml').write_text('model_provider="fixture"\n[model_providers.fixture]\nname="Fixture"\nbase_url="http://127.0.0.1:1"\nwire_api="responses"\n[rax]\nkey_file="model-key"\n')
    config = replace(configuration(tmp_path), task=task, demonstration_context=kind)
    from services.controller.demonstrations import provision_demonstration
    monkeypatch.setattr('harness.codex_cli.auto_research.provision_demonstration',
        lambda config, workspace, _, **kw: provision_demonstration(config, workspace, demo_cache.parent, **kw))
    key = provider/'model-key'
    key.write_text('sk-fixture\n')
    key.chmod(0o600)
    deployed = deploy(config, provider_home=provider, codex='/usr/bin/true')
    settings = tomllib.loads((deployed['codex_home']/'config.toml').read_text())
    assert set(settings['mcp_servers']) == {'auto_research'}
    assert settings['web_search'] == 'disabled' and settings['approval_policy'] == 'never'
    # Codex's own sandbox confines the agent's commands: they write the workspace and the run
    # tree's top, read the run tree's parts and Codex's project configuration, and see neither
    # the repository nor the sessions.
    profile = settings['permissions'][settings['default_permissions']]
    filesystem = profile['filesystem']
    assert profile['extends'] == ':workspace' and 'sandbox_mode' not in settings
    workspace, published = deployed['workspace'].resolve(), (config.root/'published').resolve()
    assert filesystem[str(workspace)] == 'write' and filesystem[str(workspace/'.codex')] == 'read'
    assert filesystem[str(published)] == 'write' and filesystem[str(published/'runs')] == 'read'
    assert any(filesystem.get(str(path.resolve())) == 'deny' for path in agent_host.repositories())
    assert filesystem[str(config.root.resolve().parent)] == 'deny'
    assert settings['model_reasoning_effort'] == config.agent_effort == 'xhigh'
    # Otherwise Codex's own defaults (compaction included).
    assert not {'model_auto_compact_token_limit', 'model_auto_compact_token_limit_scope'} & set(settings)
    assert settings['model_providers']['session']['base_url'] == 'http://127.0.0.1:1'
    assert 'sk-fixture' not in (deployed['codex_home']/'config.toml').read_text()
    assert settings['mcp_servers']['auto_research']['tool_timeout_sec'] == CALL_SECONDS + 300  # 20 days.
    assert not (deployed['codex_home']/'auth.json').exists()
    assert (deployed['workspace']/'.agents/skills/autonomous-control/SKILL.md').is_file()
    assert not (deployed['workspace']/'.agents/skills/policies').exists()  # No policy skill or package.
    from services.controller.storage import AGENT_TEMPLATE
    from services.mcp_contract import Contract
    from services.workspace import Setup, install_skills, render
    from services.workspace.recipes import SKILL_LAYERS
    demonstrations = kind != 'none'
    images = {'none': 0, 'terminal_state': 1}.get(kind, 3)
    setup = Setup(Contract.from_config(config), task=task, demonstrations=demonstrations, demonstration_kind=kind,
                  demonstration_images=images)
    installed = tmp_path/'installed'
    install_skills(installed/'.agents/skills', [layer.format(**setup.axes()) for layer in SKILL_LAYERS])
    assert (deployed['workspace']/'.agents/skills/in-context-action-learning').exists() == demonstrations
    for name in ('robot-interface', 'object-localization', 'experience-memory', 'autonomous-control',
                 'manual-robot-control', 'gemini'):
        relative = Path('.agents/skills')/name
        assert (deployed['workspace']/relative/'SKILL.md').is_file()
        for source in (installed/relative).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                copied = deployed['workspace']/source.relative_to(installed)
                if source.name.endswith('.formal-episode.md'):
                    assert not copied.exists()  # Formal-copy variant only.
                    continue
                if source.suffix == '.md':
                    assert copied.read_text() == render(source.read_text(), setup)
                else:
                    assert copied.read_bytes() == source.read_bytes()
    from services.controller.storage import RUNTIME_SOURCE, runtime_digest
    copied_api = deployed['workspace']/'api/runtime.py'
    assert copied_api.read_bytes() == RUNTIME_SOURCE.read_bytes()
    original_digest = runtime_digest()
    copied_api.write_text('# Agent can edit its own copy, not the trusted runner SDK.\n')
    assert runtime_digest() == original_digest
    assert not (deployed['workspace']/'services').exists()
    assert 'https://' not in (deployed['workspace']/'TASK.md').read_text()
    assert 'Python' in (deployed['workspace']/'AGENTS.md').read_text()
    context = json.loads((config.root/'demonstration-context.json').read_text())
    assert context['kind'] == kind
    target = deployed['workspace']/'runtime/demonstrations'
    assert not list(target.rglob('*.mp4'))  # Only pixels, never video/data/actions.
    assert context['manifest_path'] is None and not list(target.iterdir())
    for name in ('AGENTS.md', 'START_PROMPT.md'):
        assert (deployed['workspace']/name).read_text() == render((AGENT_TEMPLATE/name).read_text(), setup)
    assert startup_prompt(deployed['workspace']) == render((AGENT_TEMPLATE/'START_PROMPT.md').read_text(), setup).strip()
    assert (deployed['configuration'].stat().st_mode & 0o777) == 0o600
    with pytest.raises(FileExistsError):
        deploy(config, provider_home=provider, codex='/usr/bin/true')


def test_a_key_profile_logs_in_with_its_own_key_and_openrouter_key_stays_for_gemini(tmp_path, monkeypatch, demo_cache):
    provider = tmp_path/'provider'
    (provider/'secrets').mkdir(parents=True)
    for name, key in (('openai_key', 'sk-profile-fixture'), ('openrouter_key', 'sk-or-fixture')):
        (provider/'secrets'/name).write_text(key+'\n')
        (provider/'secrets'/name).chmod(0o600)
    (provider/'native.config.toml').write_text('model_provider="openai"\n[model_providers.openai]\nname="OpenAI"\n'
        'base_url="https://api.openai.com/v1"\nwire_api="responses"\n[rax]\nkey_file="secrets/openai_key"\nnative_models=true\n')
    config = configuration(tmp_path)
    from services.controller.demonstrations import provision_demonstration
    monkeypatch.setattr('harness.codex_cli.auto_research.provision_demonstration',
        lambda config, workspace, _, **kw: provision_demonstration(config, workspace, demo_cache.parent, **kw))
    deployed = deploy(config, provider_home=provider, profile='native', key_file=provider/'secrets/openrouter_key',
                      codex='/usr/bin/true')
    assert json.loads(deployed['login'].read_text()) == {
        'profile': 'native', 'model': MODEL.removeprefix('openai/'), 'key_file': str((provider/'secrets/openai_key').resolve()),
        'provider': {'name': 'OpenAI', 'base_url': 'https://api.openai.com/v1', 'wire_api': 'responses'}}
    written = (deployed['codex_home']/'config.toml').read_text()
    assert 'sk-profile-fixture' not in written and 'sk-or-fixture' not in written
    # Codex itself reads the profile's key; Gemini keeps the OpenRouter key.
    from harness.codex_cli.auto_research import agent_environment
    assert agent_environment(config, deployed, tmp_path/'mcp.sock')['RAX_MODEL_KEY'] == 'sk-profile-fixture'


def test_chatgpt_profile_stores_only_the_login_path(tmp_path, monkeypatch, demo_cache):
    provider = tmp_path/'provider'
    provider.mkdir()
    (provider/'auth.json').write_text(json.dumps({'auth_mode': 'chatgpt',
        'tokens': {'access_token': 'h.e30.s', 'refresh_token': 'r', 'account_id': 'acct-fixture'}}))
    (provider/'openrouter_key').write_text('sk-or-fixture\n')
    (provider/'openrouter_key').chmod(0o600)
    (provider/'login.config.toml').write_text('model_provider="chatgpt"\n[model_providers.chatgpt]\nname="ChatGPT"\n'
        'base_url="https://chatgpt.com/backend-api/codex"\nwire_api="responses"\n[rax]\nchatgpt_auth="auth.json"\nnative_models=true\n')
    config = configuration(tmp_path)
    from services.controller.demonstrations import provision_demonstration
    monkeypatch.setattr('harness.codex_cli.auto_research.provision_demonstration',
        lambda config, workspace, _, **kw: provision_demonstration(config, workspace, demo_cache.parent, **kw))
    deployed = deploy(config, provider_home=provider, profile='login', key_file=provider/'openrouter_key',
                      codex='/usr/bin/true')
    login = str((provider/'auth.json').resolve())
    assert json.loads(deployed['login'].read_text()) == {
        'profile': 'login', 'model': MODEL.removeprefix('openai/'), 'chatgpt_auth': login}
    # Codex refreshes the shared login through the session's link; no token is copied.
    auth = deployed['codex_home']/'auth.json'
    assert auth.is_symlink() and str(auth.readlink()) == login
    settings = (deployed['codex_home']/'config.toml').read_text()
    assert 'model_providers' not in tomllib.loads(settings) and 'h.e30.s' not in settings


def test_profile_model_is_the_sessions_only_model(tmp_path, monkeypatch, demo_cache):
    """A profile's [rax] model is the model in the session's Codex configuration."""
    provider = tmp_path/'provider'
    login = tmp_path/'elsewhere/auth.json'
    login.parent.mkdir()
    login.write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {'access_token': 'h.e30.s', 'refresh_token': 'r',
                                                                    'account_id': 'acct-fixture'}}))
    provider.mkdir()
    # An absolute login path: an account logged in under its own CODEX_HOME.
    (provider/'other.config.toml').write_text('model_provider="chatgpt"\n[model_providers.chatgpt]\nname="ChatGPT"\n'
        'base_url="https://chatgpt.com/backend-api/codex"\nwire_api="responses"\n[rax]\n'
        f'chatgpt_auth="{login}"\nnative_models=true\nmodel="openai/gpt-6.1-sol"\n')
    from services.controller.demonstrations import provision_demonstration
    monkeypatch.setattr('harness.codex_cli.auto_research.provision_demonstration',
        lambda config, workspace, _, **kw: provision_demonstration(config, workspace, demo_cache.parent, **kw))
    deployed = deploy(configuration(tmp_path), provider_home=provider, profile='other', codex='/usr/bin/true')
    stored = json.loads(deployed['login'].read_text())
    assert stored['model'] == 'gpt-6.1-sol' and stored['chatgpt_auth'] == str(login.resolve())
    assert tomllib.loads((deployed['codex_home']/'config.toml').read_text())['model'] == 'gpt-6.1-sol'
    (provider/'bad.config.toml').write_text((provider/'other.config.toml').read_text().replace(
        'model="openai/gpt-6.1-sol"', 'model="bad model"'))
    with pytest.raises(ValueError, match="Invalid model in provider profile 'bad'"):
        deploy(configuration(tmp_path/'second'), provider_home=provider, profile='bad', codex='/usr/bin/true')


def deploy_profile(tmp_path, monkeypatch, demo_cache, profile, *, name='test', **kwargs):
    """deploy() with a provider profile's TOML text (or none); returns the session's stored login."""
    from services.controller.demonstrations import provision_demonstration
    monkeypatch.setattr('harness.codex_cli.auto_research.provision_demonstration',
        lambda config, workspace, _, **kw: provision_demonstration(config, workspace, demo_cache.parent, **kw))
    provider = tmp_path/'provider'
    provider.mkdir(parents=True, exist_ok=True)
    if profile is not None:
        (provider/f'{name}.config.toml').write_text(profile)
    deployed = deploy(configuration(tmp_path/'session'), provider_home=provider, profile=name, codex='/usr/bin/true',
                      **kwargs)
    return json.loads(deployed['login'].read_text())


def private_key(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text+'\n')
    path.chmod(0o600)
    return path


def openrouter_key(tmp_path):
    return private_key(tmp_path/'openrouter_key', 'sk-or-fixture')


EXAMPLE = ('model_provider="example"\n[model_providers.example]\nname="Example"\n'
           'base_url="https://api.example.com/v1"\nwire_api="responses"\nenv_key="EXAMPLE_API_KEY"\n')
OPENROUTER = ('model_provider="openrouter"\n[model_providers.openrouter]\nname="OpenRouter"\n'
              'base_url="https://openrouter.ai/api/v1"\nwire_api="responses"\n')


def test_another_provider_uses_its_own_key_file_and_never_the_openrouter_key(tmp_path, monkeypatch, demo_cache):
    key = private_key(tmp_path/'provider/secrets/example_key', 'sk-example-fixture')
    login = deploy_profile(tmp_path, monkeypatch, demo_cache, EXAMPLE+'[rax]\nkey_file="secrets/example_key"\n',
                           key_file=openrouter_key(tmp_path))
    assert login['key_file'] == str(key.resolve())
    # Codex reads the key from the variable the session sets, not the operator's environment.
    assert login['provider'] == {'name': 'Example', 'base_url': 'https://api.example.com/v1', 'wire_api': 'responses'}


def test_another_provider_without_its_own_key_file_is_refused(tmp_path, monkeypatch, demo_cache):
    """The operator's OpenRouter key (kept for Gemini) is sent only to OpenRouter."""
    with pytest.raises(ValueError, match=r"Provider profile 'test' needs its own key file \(\[rax\] key_file\); "
                                         'the OpenRouter key is sent only to OpenRouter'):
        deploy_profile(tmp_path, monkeypatch, demo_cache, EXAMPLE, key_file=openrouter_key(tmp_path))


def test_the_openrouter_profile_uses_the_openrouter_key_file(tmp_path, monkeypatch, demo_cache):
    key = openrouter_key(tmp_path)
    login = deploy_profile(tmp_path/'file', monkeypatch, demo_cache, OPENROUTER, key_file=key)
    assert login['key_file'] == str(key.resolve())
    # Identified by its endpoint under any profile name.
    renamed = OPENROUTER.replace('openrouter"', 'router"').replace('providers.openrouter', 'providers.router')
    login = deploy_profile(tmp_path/'named', monkeypatch, demo_cache, renamed, name='mine', key_file=key)
    assert login['provider']['base_url'] == 'https://openrouter.ai/api/v1' and login['key_file'] == str(key.resolve())
    with pytest.raises(ValueError, match='The OpenRouter profile needs a key file: --key-file or '
                                         '~/.codex/secrets/openrouter_api_key'):
        deploy_profile(tmp_path/'none', monkeypatch, demo_cache, OPENROUTER, key_file=None)


def test_the_openrouter_profile_needs_no_provider_file(tmp_path, monkeypatch, demo_cache):
    """A fresh host with only an OpenRouter key: the default profile is built in; any other
    profile without a provider table is still refused by name."""
    from harness.codex_cli.auto_research import MODEL
    key = openrouter_key(tmp_path)
    login = deploy_profile(tmp_path, monkeypatch, demo_cache, None, name='openrouter', key_file=key)
    assert login == {'profile': 'openrouter', 'model': MODEL, 'key_file': str(key.resolve()),
                     'provider': {'name': 'OpenRouter', 'base_url': 'https://openrouter.ai/api/v1',
                                  'wire_api': 'responses'}}
    with pytest.raises(ValueError, match="Provider profile 'other' needs model_provider"):
        deploy_profile(tmp_path/'other', monkeypatch, demo_cache, None, name='other', key_file=key)


def test_the_launcher_key_file_is_optional(tmp_path, monkeypatch):
    from harness.codex_cli import auto_research
    monkeypatch.setenv('HOME', str(tmp_path))
    assert auto_research.openrouter_key_file() is None  # Gemini then reads OPENROUTER_API_KEY.
    default = tmp_path/'.codex/secrets/openrouter_api_key'
    default.parent.mkdir(parents=True)
    default.write_text('sk-or-fixture')
    assert auto_research.openrouter_key_file() == default
    assert auto_research.openrouter_key_file(tmp_path/'chosen') == tmp_path/'chosen'
    from harness.claude_cli.login import Credential
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-env-fixture')
    assert Credential('openrouter', key_file=None).secret() == 'sk-or-env-fixture'


def test_demo_context_validation_and_legacy_default(tmp_path):
    from dataclasses import asdict
    from services.controller.config import Configuration
    config = configuration(tmp_path)
    with pytest.raises(ValueError, match='demonstration_context'):
        replace(config, demonstration_context='video_url')
    legacy = asdict(config)
    legacy.pop('demonstration_context')
    assert Configuration.from_dict(legacy).demonstration_context == 'none'


def test_status_returns_only_trusted_demo_descriptor(supervisor, frontend):
    context = {'kind':'completion_sequence', 'image_count':3, 'root':'runtime/demonstrations',
        'manifest_path':'runtime/demonstrations/manifest.json'}
    (supervisor[0].config.root/'demonstration-context.json').write_text(json.dumps(context))
    assert data(frontend.call('status', {}))['demonstration_context'] == context


def test_startup_prompt_uses_deployed_draft_and_preserves_explicit_override(tmp_path):
    prompt = tmp_path/'START_PROMPT.md'
    prompt.write_text('Custom draft for this deployment.\n')
    assert startup_prompt(tmp_path) == 'Custom draft for this deployment.'
    prompt.unlink()
    assert startup_prompt(tmp_path, 'Diagnostic only; do not start an episode.') == 'Diagnostic only; do not start an episode.'


def fake_launch(tmp_path, monkeypatch, events, *, stopped=False):
    """The native Codex launcher with the frontend and the agent CLI replaced by recorders."""
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    config.root.mkdir(parents=True)
    deployed = {'workspace': tmp_path, 'codex_home': tmp_path, 'configuration': tmp_path/'config.json'}

    class Process:
        returncode = None
        terminated = False
        pid = os.getpid()

        def __init__(self, command, **kwargs):
            self.args = command
            self.name = 'agent' if command[0] == 'fake-agent' else 'server'
            events.append(self.name+' started')
            if self.name == 'agent':
                assert kwargs['stdin'] is subprocess.DEVNULL  # Codex exec would wait on an open stdin.
            self.stderr = iter([b'working\n', b'ERROR: Reconnecting... 1/2\n',
                                b'ERROR: Selected model is at capacity. Please try a different model.\n'])

        def wait(self, timeout=None):
            if self.name == 'agent' and not self.terminated:
                assert timeout is None  # A session has no wall-clock budget.
                assert (config.root/auto_research.PID_FILE).is_file()  # For register and rehearse to freeze it.
                events.append('unbounded wait')
                if stopped:
                    raise KeyboardInterrupt  # The operator's stop.
            self.returncode = 1 if self.name == 'agent' else 0
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            events.append(self.name+' terminated')

    monkeypatch.setattr(auto_research, 'check_login', lambda deployed: {})
    monkeypatch.setattr(auto_research, 'write_home', lambda *a: None)
    monkeypatch.setattr(auto_research, 'start_frontend', lambda *a, **k: Process(['frontend']))
    monkeypatch.setattr(auto_research, 'agent_command', lambda *a, prompt, **k: ['fake-agent', prompt])
    monkeypatch.setattr(auto_research, 'agent_environment', lambda *a: {})
    monkeypatch.setattr(auto_research.agent_host, 'start', lambda command, **k: Process(command, **k))
    return auto_research, config, deployed


@pytest.mark.parametrize('stopped', [False, True])
def test_launcher_waits_without_a_deadline_and_preserves_cleanup(tmp_path, monkeypatch, stopped):
    events = []
    auto_research, config, deployed = fake_launch(tmp_path, monkeypatch, events, stopped=stopped)
    if stopped:
        with pytest.raises(KeyboardInterrupt):
            auto_research.launch(config, deployed, prompt='test')
    else:
        assert auto_research.launch(config, deployed, prompt='test').returncode == 1
        # Codex's last ERROR line, the provider's message, is kept whole.
        assert deployed['agent_error'] == 'ERROR: Selected model is at capacity. Please try a different model.'
    assert 'unbounded wait' in events
    assert ('agent terminated' in events) == stopped  # A running agent is stopped; an exited one is not.
    assert 'server terminated' in events and not (config.root/auto_research.PID_FILE).exists()


def test_one_frontend_serves_every_agent_run_of_a_launch(tmp_path, monkeypatch):
    events = []
    auto_research, config, deployed = fake_launch(tmp_path, monkeypatch, events)
    def agent_runs(run):
        assert run('first') == 1
        events.append('between runs')
        return run('second')
    assert auto_research.launch(config, deployed, prompt='first', agent_runs=agent_runs).returncode == 1
    assert events.count('server started') == 1 and events.count('agent started') == 2
    # The frontend, owner of the open episodes, is still up between the runs; cleanup stops it
    # once, after the last run.
    between = events.index('between runs')
    assert 'server terminated' not in events[:between] and events[between+1] == 'agent started'
    assert events.index('server terminated') > events.index('agent started', between)


@pytest.mark.parametrize('options,interactive,prompt', [([], True, 'Draft startup.'),
    (['--non-interactive'], False, 'Draft startup.'), (['--exec-prompt', 'Diagnostic.'], False, 'Diagnostic.')])
def test_launcher_mode_and_durable_lifecycle(tmp_path, monkeypatch, options, interactive, prompt):
    from types import SimpleNamespace
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    source = tmp_path/'input.json'
    source.write_text('{}')
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'START_PROMPT.md').write_text('Draft startup.\n')
    monkeypatch.setattr(auto_research.Configuration, 'from_dict', lambda _: config)
    monkeypatch.setattr(auto_research.Configuration, '__post_init__', lambda _: None)  # Local fixture storage.
    monkeypatch.setattr(auto_research, 'deploy', lambda *a, **k: {'workspace':workspace, 'codex_home':workspace})
    monkeypatch.setattr(auto_research.signal, 'signal', lambda *a: None)
    calls = []
    def launch(*args, **kwargs):
        calls.append(kwargs)
        record = json.loads((config.root/'agent-lifecycle.json').read_text())
        assert record['status'] == 'running' and not record.get('finished_at')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(auto_research, 'launch', launch)
    monkeypatch.setattr(auto_research.sys, 'argv', ['auto-research', '--config', str(source), *options])
    with pytest.raises(SystemExit) as exit:
        auto_research.main()
    assert exit.value.code == 0
    assert calls[0]['interactive'] is interactive and calls[0]['prompt'] == prompt
    path = config.root/'agent-lifecycle.json'
    record = json.loads(path.read_text())
    assert record['status'] == 'exited' and record['finished_at'] >= record['started_at']
    with pytest.raises(FileExistsError):
        auto_research.main()
    assert json.loads(path.read_text()) == record


def test_each_agent_turn_is_told_of_episodes_refunded_since_the_last(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    source = tmp_path/'input.json'
    source.write_text('{}')
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'START_PROMPT.md').write_text('Draft startup.\n')
    pool = config.root/'exploration'
    pool.mkdir(parents=True)
    rows = [{'episode': 1, 'env_id': 0, 'seed': 0, 'status': 'closed'},
            {'episode': 2, 'env_id': 1, 'seed': 1, 'status': 'active'}]  # Left live by the stopped frontend.
    (pool/'state.json').write_text(json.dumps({'schema': 1, 'env_count': 2, 'seeds': [0, 1, 2, 3],
                                               'episodes': rows, 'sealed': False}))
    monkeypatch.setattr(auto_research.Configuration, 'from_dict', lambda _: config)
    monkeypatch.setattr(auto_research.Configuration, '__post_init__', lambda _: None)  # Local fixture storage.
    monkeypatch.setattr(auto_research, 'deploy', lambda *a, **k: {'workspace': workspace, 'codex_home': workspace})
    monkeypatch.setattr(auto_research.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(auto_research, 'latest_session', lambda home: 'thread')
    monkeypatch.setattr(auto_research, 'MAX_RELAUNCHES', 1)
    prompts = []
    def launch(*args, prompt, agent_runs, **kwargs):
        def run(prompt):
            prompts.append(prompt)
            state = json.loads((pool/'state.json').read_text())
            state['episodes'].append({'episode': 3, 'env_id': 0, 'seed': 2, 'status': 'error',
                                      'error': 'RuntimeError: worker exited (-9)'})  # Fails during the first turn.
            (pool/'state.json').write_text(json.dumps(state))
            return 0
        return SimpleNamespace(returncode=agent_runs(run))
    monkeypatch.setattr(auto_research, 'launch', launch)
    monkeypatch.setattr(auto_research.sys, 'argv', ['auto-research', '--config', str(source), '--non-interactive'])
    with pytest.raises(SystemExit) as exit:
        auto_research.main()
    assert exit.value.code == 0 and len(prompts) == 2
    first, second = prompts
    assert first.startswith('Draft startup.\n\nHarness note:') and 'episode 2 (interrupted: The session stopped while this episode was live)' in first
    assert 'episode 3' not in first
    assert second.startswith(auto_research.CONTINUE_PROMPT)
    assert 'episode 3 (error: RuntimeError: worker exited (-9))' in second and 'episode 2 ' not in second


def test_a_codex_error_resumes_the_conversation_against_the_same_frontend(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    source = tmp_path/'input.json'
    source.write_text('{}')
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'START_PROMPT.md').write_text('Draft startup.\n')
    monkeypatch.setattr(auto_research.Configuration, 'from_dict', lambda _: config)
    monkeypatch.setattr(auto_research.Configuration, '__post_init__', lambda _: None)  # Local fixture storage.
    monkeypatch.setattr(auto_research, 'deploy', lambda *a, **k: {'workspace': workspace, 'codex_home': workspace})
    monkeypatch.setattr(auto_research.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(auto_research, 'latest_session', lambda home: 'thread')
    monkeypatch.setattr(auto_research, 'MAX_RELAUNCHES', 0)
    monkeypatch.setattr(auto_research, 'ERROR_PAUSE_SECONDS', 0)
    error = 'ERROR: Selected model is at capacity. Please try a different model.'
    launches, prompts = [], []
    def launch(config, deployed, *, prompt, agent_runs, **kwargs):
        launches.append(prompt)
        def run(prompt):
            prompts.append((prompt, deployed.get('resume_session')))
            deployed['agent_error'] = error if len(prompts) == 1 else None
            return 1 if len(prompts) == 1 else 0
        return SimpleNamespace(returncode=agent_runs(run))
    monkeypatch.setattr(auto_research, 'launch', launch)
    monkeypatch.setattr(auto_research.sys, 'argv', ['auto-research', '--config', str(source), '--non-interactive'])
    with pytest.raises(SystemExit) as exit:
        auto_research.main()
    assert exit.value.code == 0 and launches == ['Draft startup.']  # One launch: one frontend.
    assert prompts == [('Draft startup.', None), (auto_research.ERROR_PROMPT.format(error=error), 'thread')]
    record = json.loads((config.root/'agent-lifecycle.json').read_text())
    assert [(e['returncode'], e['error']) for e in record['agent_errors']] == [(1, error)]


def agent_formal_launch(tmp_path, monkeypatch, *options):
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    source = tmp_path/'input.json'
    source.write_text('{}')
    monkeypatch.setattr(auto_research.Configuration, 'from_dict', lambda _: config)
    monkeypatch.setattr(auto_research.Configuration, '__post_init__', lambda _: None)  # Local fixture storage.
    monkeypatch.setattr(auto_research.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(auto_research, 'redeploy', lambda c: {'workspace': tmp_path/'workspace'})
    monkeypatch.setattr(auto_research, 'launch', lambda *a, **k: pytest.fail('the exploring agent relaunched'))
    monkeypatch.setattr(auto_research.sys, 'argv', ['auto-research', '--config', str(source), '--agent-formal', *options])
    return auto_research, config


def test_agent_formal_flag_plays_live_copies_on_a_closed_session(tmp_path, monkeypatch):
    from services.agent_formal import checkpoint
    auto_research, config = agent_formal_launch(tmp_path, monkeypatch, '--resume')
    config.root.mkdir(parents=True)
    (config.root/'configuration.json').write_text('{}')  # Deployed when it was first launched.
    (config.root/'exploration-complete.json').write_text('{}')
    (config.root/'agent-lifecycle.json').write_text(json.dumps(
        {'pid': 1, 'started_at': 0, 'finished_at': 1, 'status': 'exited', 'formal': 'completed'}))
    calls = []
    monkeypatch.setattr(checkpoint, 'after_exploration', lambda c, d, **k: calls.append(k) or
                        {'status': 'completed', 'agent_formal': {'status': 'completed'}})
    with pytest.raises(SystemExit) as exit:
        auto_research.main()
    assert exit.value.code == 0 and [k['live_copies'] for k in calls] == [True]
    record = json.loads((config.root/'agent-lifecycle.json').read_text())
    assert record['status'] == 'exited' and record['agent_formal'] == 'completed' and len(record['attempts']) == 1


@pytest.mark.parametrize('options', [(), ('--resume',)])
def test_agent_formal_flag_needs_a_closed_session(tmp_path, monkeypatch, options):
    auto_research, config = agent_formal_launch(tmp_path, monkeypatch, *options)
    with pytest.raises(SystemExit, match='exploration has closed'):
        auto_research.main()


def test_invalid_lifecycle_arguments_name_the_field(supervisor, tmp_path):
    from services.controller.frontend import ResearchFrontend
    s, _ = supervisor
    f = ResearchFrontend(s, tmp_path/'workspace')
    try:
        with pytest.raises(ValueError, match='unexpected: env_id'):
            f.call('start_episode', {'env_id': 0})
        with pytest.raises(ValueError, match='missing: source'):
            f.call('register', {})
    finally:
        f.api.revoke()


def test_lifecycle_refusals_reach_the_session_audit_whole(supervisor, frontend):
    """A refused lifecycle call (here: no exploration episode left) is an mcp_error in the
    session's audit, as robot tool errors are, so the operator's error tally sees it."""
    s, _ = supervisor
    s.state['exploration_started'] = len(s.config.exploration_seeds)
    reply = frontend.handle({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
                             'params': {'name': 'start_episode', 'arguments': {}}})
    assert reply['result']['isError']
    events = [json.loads(line) for line in (s.config.root/'mcp.jsonl').read_text().splitlines()]
    [error] = [event for event in events if event['event'] == 'mcp_error']
    assert {k: error[k] for k in ('operation', 'stage', 'type', 'reason')} == {
        'operation': 'start_episode', 'stage': 'lifecycle', 'type': 'RuntimeError',
        'reason': 'No exploration episode remains; call status for the next step'}
    assert data(reply['result'])['error'] == {k: error[k] for k in data(reply['result'])['error']}


@pytest.mark.parametrize('exits', [True, False])
def test_a_frontend_that_fails_to_start_reports_its_own_error(tmp_path, monkeypatch, exits):
    """The launcher passes the frontend's error through whole: how it ended (its exit code, or no
    socket in time, judged before a hung one is ended) and its traceback, not a pointer to its log."""
    from harness.codex_cli import auto_research
    config = configuration(tmp_path)
    config.root.mkdir(parents=True)
    deployed = {'configuration': tmp_path/'config.json', 'workspace': tmp_path}
    ended, sessions = [], []
    artifact_root = tmp_path/'separate-storage'
    monkeypatch.setenv('RAX_ARTIFACT_ROOT', str(artifact_root))
    monkeypatch.setenv('RAX_PRIVATE_FIXTURE', 'must-not-be-forwarded')

    class Frontend:
        returncode = None

        def __init__(self, args, *, stdout, **kwargs):
            assert kwargs['env']['RAX_ARTIFACT_ROOT'] == str(artifact_root)
            assert 'RAX_PRIVATE_FIXTURE' not in kwargs['env']
            sessions.append(kwargs.get('start_new_session'))
            stdout.write(b"Traceback (most recent call last):\n  File \"frontend.py\"\n"
                         b"ValueError: A trial's saved configuration cannot change\n")
            if exits:
                self.returncode = 1

        def poll(self):
            return self.returncode

        def terminate(self):
            ended.append('terminated')
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(auto_research.subprocess, 'Popen', Frontend)
    (config.root/'frontend.log').write_bytes(b'RuntimeError: the earlier launch\'s simulator error\n')
    with auto_research.open_frontend_log(config.root) as log, pytest.raises(RuntimeError) as raised:
        auto_research.start_frontend(config, deployed, tmp_path/'mcp.sock', log, timeout=0.2)
    how = 'exited with code 1' if exits else 'opened no socket within 0.2 s'
    assert str(raised.value) == (f"Host MCP startup failed: the frontend {how}; frontend.log ends:\n"
                                 "Traceback (most recent call last):\n  File \"frontend.py\"\n"
                                 "ValueError: A trial's saved configuration cannot change")
    assert ended == ([] if exits else ['terminated'])
    assert sessions == [True]  # Its own session: a terminal's Ctrl+C reaches only the launcher.
    # Each launch appends under its own header: the earlier launch's error stays readable.
    log = (config.root/'frontend.log').read_text().splitlines()
    assert log[0] == "RuntimeError: the earlier launch's simulator error" and log[1] == ''
    assert re.fullmatch(rf'===== launch \S+ \(launcher pid {os.getpid()}\) =====', log[2])
    assert log[3:] == ['Traceback (most recent call last):', '  File "frontend.py"',
                       "ValueError: A trial's saved configuration cannot change"]


def test_a_lifecycle_deadline_is_a_timeout_of_that_call(supervisor, frontend, monkeypatch):
    """The startup or evaluation wall limit (ExecutionDeadline, a BaseException) answers the
    call that hit it with a timeout error; the server keeps serving."""
    from services.controller.sandbox import ExecutionDeadline
    s, backend = supervisor
    def stuck(*, formal, seed):
        raise ExecutionDeadline('Execution wall-time budget exhausted')
    monkeypatch.setattr(backend, 'start', stuck)
    call = lambda name: frontend.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                                         'params': {'name': name, 'arguments': {}}})
    reply = call('start_episode')
    assert reply['result']['isError']
    error = json.loads(reply['result']['content'][0]['text'])['error']
    assert (error['type'], error['operation'], error['stage']) == ('TimeoutError', 'start_episode', 'lifecycle')
    from services.benchmarks import for_config
    startup = for_config(s.config).startup_seconds
    assert error['reason'] == f'The environment did not start within its {startup:g}s startup limit'
    assert s.state['active'] is None
    status = call('status')
    assert 'error' not in status and not status['result'].get('isError')


def test_a_failed_start_is_kept_in_the_operator_record(supervisor, frontend, monkeypatch):
    """The agent's reply is not the only record: the result row, its published result.json
    and an mcp_error event in mcp.jsonl all keep the whole error."""
    s, backend = supervisor
    def broken(*, formal, seed):
        raise RuntimeError('simulator process exited with code -11; its log: CUDA error: out of memory')
    monkeypatch.setattr(backend, 'start', broken)
    reply = frontend.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                             'params': {'name': 'start_episode', 'arguments': {}}})
    reason = 'simulator process exited with code -11; its log: CUDA error: out of memory'
    assert json.loads(reply['result']['content'][0]['text'])['error']['reason'] == reason
    row = s.state['results'][-1]
    assert row['start_failure'] == {'type': 'RuntimeError', 'operation': 'start_episode',
                                    'stage': 'episode_start', 'reason': reason}
    [published] = [json.loads(p.read_text()) for p in (s.config.root/'published/runs').glob('interactive-*/result.json')]
    events = [json.loads(line) for line in (s.config.root/'mcp.jsonl').read_text().splitlines()]
    errors = [e for e in events if e['event'] == 'mcp_error']
    assert [(e['tool'], e['stage'], e['reason']) for e in errors] == [('start_episode', 'lifecycle', reason)]
    assert published['start_failure']['reason'] == reason
