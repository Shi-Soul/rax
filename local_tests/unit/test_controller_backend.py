"""Portable controller contracts; native/GPU and bundle sandbox checks are separate."""
import base64
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from bench import BENCH, CAMERA, OBSERVE, STEP, TASK
from services.controller.config import Configuration, Limits
from services.controller.gemini import GeminiRouter, MODEL, load_key
from services.controller.sandbox import BundleSandbox, ExecutionDeadline, deadline
from services.controller.storage import snapshot, verify
from services.controller.supervisor import Supervisor, formal_outcome


def configuration(tmp_path):
    limits = Limits(60, 512, 1, 32, 32, 4 * 1024 * 1024)
    # tmp_path is under the tests' artifact root (conftest), as production session roots are.
    return Configuration(tmp_path / "private",
        TASK, (1, 2, 3), 4, 0, "GPU-aaaa",
        limits, limits, limits, formal_eval_seed=1)


def test_host_only_key_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-test-key")
    assert load_key() == "environment-test-key"
    key = tmp_path / "key"
    key.write_text("file-test-key\n")
    key.chmod(0o600)
    assert load_key(key) == "file-test-key"
    key.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        load_key(key)
    key.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(key)
    with pytest.raises(OSError):
        load_key(link)


def test_legacy_training_allowance_is_ignored_on_config_load(tmp_path):
    raw = asdict(configuration(tmp_path))
    raw['training_seconds'] = 3600
    config = Configuration.from_dict(raw)
    assert 'training_seconds' not in asdict(config)
    assert asdict(config.training) == raw['training']


def test_agent_gpu_fields_of_old_configurations_are_retired_on_load(tmp_path):
    raw = asdict(configuration(tmp_path))
    raw.update(controller_gpu='GPU-bbbb', training_gpu='GPU-bbbb')
    config = Configuration.from_dict(raw)
    assert config.sim_gpu == 'GPU-aaaa'
    assert 'controller_gpu' not in asdict(config) and 'training_gpu' not in asdict(config)
    with pytest.raises(TypeError, match='controller_gpu'):  # New configurations cannot name one.
        replace(config, controller_gpu='GPU-bbbb')
    with pytest.raises(ValueError, match='explicit GPU UUID'):
        replace(config, sim_gpu='0')


def test_bundle_runners_get_no_gpu(supervisor, tmp_path):
    s, _ = supervisor
    created = []
    class CaptureRunner(Runner):
        def __init__(self, limits, **kwargs):
            created.append(kwargs)
            super().__init__(limits, **kwargs)
    s.runner_factory = CaptureRunner
    ident, _ = s.register(source(tmp_path))
    assert s.run(ident, formal=False)['reason'] == 'exit'
    assert created and all(kwargs == {} for kwargs in created)


def test_exited_simulator_cleanup_does_not_evaluate(supervisor):
    s, backend = supervisor
    gateway, _ = s.start_interactive()
    gateway.simulator_exited = 'simulator process exited with code -11'
    result = s.finish_interactive()
    assert result['simulator_exited'] == 'simulator process exited with code -11'
    assert backend.evaluations == 0 and backend.closed


def test_a_backend_with_a_native_record_evaluates_after_its_simulator_exits(supervisor):
    s, backend = supervisor
    backend.evaluates_from_record = True  # A backend whose verdict is a file the simulator wrote.
    gateway, _ = s.start_interactive()
    gateway.simulator_exited = 'simulator process exited with code 0; its log: {"event": "episode_wall_timeout"}'
    result = s.finish_interactive()
    assert result['evaluation'] == {'task_complete': False} and backend.evaluations == 1
    assert result['simulator_exited'].startswith('simulator process exited with code 0')
    def unreadable():
        raise RuntimeError('Native evaluation identity mismatch')
    backend.evaluate = unreadable
    gateway, _ = s.start_interactive()
    gateway.simulator_exited = 'simulator process exited with code -11'
    result = s.finish_interactive()  # A crash with no matching record is recorded, not raised.
    assert result['evaluation_error'] == 'RuntimeError: Native evaluation identity mismatch'
    assert 'evaluation' not in result and backend.closed


def test_episode_restart_does_not_require_storage_symlinks(supervisor, monkeypatch):
    s, backend = supervisor
    readlink = os.readlink
    def unsupported_link(path, **kwargs):
        if Path(path).name == 'autonomous_controller':
            raise FileNotFoundError(2, 'No such file or directory', str(path))
        return readlink(path, **kwargs)
    monkeypatch.setattr(os, 'readlink', unsupported_link)
    for _ in range(2):
        s.start_interactive()
        result = s.finish_interactive()
        assert result['recording_errors'] == []
        assert 'start_failure' not in result
    assert len(backend.starts) == 2


@pytest.mark.parametrize('formal', [False, True])
def test_a_stop_mid_episode_is_recorded_as_the_stop(supervisor, tmp_path, formal):
    """SIGTERM raises SystemExit in the session service; the episode it cut short is recorded as
    stopped, and a controller stopped mid-call is not evaluated."""
    s, backend = supervisor
    def stopped(bundle, job):
        job['gateway'].handle(job['token'], packet(STEP))
        raise SystemExit(0)
    Runner.hook = stopped
    ident, _ = s.register(source(tmp_path))
    with pytest.raises(SystemExit):
        s._run_episode(ident, formal=formal)
    result = s.state['results'][-1]
    assert result['reason'] == 'session_stopped' and result['session_stopped'] == {'stage': 'controller_execution'}
    assert result['task_complete'] is None and 'evaluation_failure' not in result and backend.evaluations == 0
    assert s.state['active'] is None and backend.closed
    saved = json.loads((s.config.root/'state.json').read_text())['results'][-1]
    assert saved['reason'] == 'session_stopped'


def test_a_stop_during_evaluation_is_recorded_as_the_stop(supervisor, tmp_path, monkeypatch):
    s, backend = supervisor
    def evaluate():
        raise SystemExit(0)
    monkeypatch.setattr(backend, 'evaluate', evaluate)
    ident, _ = s.register(source(tmp_path))
    with pytest.raises(SystemExit):
        s._run_episode(ident, formal=False)
    result = s.state['results'][-1]
    assert result['reason'] == 'exit' and result['session_stopped'] == {'stage': 'evaluation'}
    assert result['task_complete'] is None and s.state['active'] is None


def packet(name, **arguments):
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}}


def value(response):
    return json.loads(response["result"]["content"][0]["text"])


class Backend:
    def __init__(self):
        self.starts = []
        self.closed = False
        self.evaluations = 0
        self.step = 0
        self.success = False
        buf = io.BytesIO()
        Image.new("RGB", (4, 3), (42, 16, 8)).save(buf, format="PNG")
        self.image = {"type": "image", "mimeType": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}

    def tools(self):
        return [{"name": s} for s in (STEP, OBSERVE, "private_request_formal_episode")]

    def start(self, *, formal, seed):
        self.starts.append((formal, seed))
        return {}, []

    def call(self, name, arguments):
        if name == STEP:
            self.step += 1
        return {"episode_id": "episode", "step_id": self.step, "seed": 123,
                "metadata": {"private": True}, "manifest_path": "/secret",
                "attachments": [{"kind": "rgb", "camera": CAMERA, "content_index": 1}]}, [self.image]

    def evaluate(self):
        self.evaluations += 1
        return {"task_complete": self.success}

    def close(self):
        self.closed = True


class Runner:
    hook = None

    def __init__(self, limits, **kwargs):
        self.limits = limits

    def preflight(self):
        pass

    def run(self, bundle, **kwargs):
        result = type(self).hook(bundle, kwargs) if type(self).hook else {"reason": "exit", "returncode": 0}
        output = kwargs['output']
        output.mkdir(exist_ok=True)
        (output/'files').mkdir(exist_ok=True)
        for name, content in [('stdout.log', ''), ('stderr.log', ''), ('logs.json', '{}')]:
            if not (output/name).exists():
                (output/name).write_text(content)
        return result


@pytest.fixture
def supervisor(tmp_path):
    backend = Backend()
    supervisor = Supervisor(configuration(tmp_path), lambda: backend, runner_factory=Runner)
    yield supervisor, backend
    Runner.hook = None
    supervisor.close()


@pytest.fixture
def unlocked_supervisor(supervisor):
    # Fixture for post-success mechanics; admission is tested separately below.
    s, _ = supervisor
    s.state["interactive_success"] = {"episode": 0, "at": 0}
    return supervisor


def source(tmp_path, entry="controller.py"):
    directory = tmp_path / entry.replace(".py", "")
    directory.mkdir()
    (directory / entry).write_text("def main(ctx):\n    pass\n")
    return directory


def test_mcp_discovery_and_no_lifecycle_tools(unlocked_supervisor):
    s, _ = unlocked_supervisor
    gateway, token = s.start_interactive()
    result = gateway.handle(token, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in result["result"]["tools"]}
    assert "gemini_generate" in names and OBSERVE in names
    assert "private_request_formal_episode" not in names
    for name in ("private_request_formal_episode", "private_request_human_setup", "private_request_human_evaluation"):
        assert "error" in gateway.handle(token, packet(name))


def test_formal_one_shot_held_out_and_evaluated_on_crash(supervisor, tmp_path):
    s, backend = supervisor
    s.start_interactive()
    backend.success = True
    s.finish_interactive()
    ident, manifest = s.register(source(tmp_path, "controller.py"))
    s.run(ident, formal=False)
    Runner.hook = lambda bundle, job: {"reason": "exit", "returncode": 1}
    result = s.run(ident, formal=True)
    assert backend.starts[-1] == (True, 4)
    assert backend.closed and backend.evaluations == 3
    assert result["returncode"] == 1 and result["task_complete"] is True
    with pytest.raises(RuntimeError, match="no retry"):
        s.run(ident, formal=True)
    s.close()
    reopened = Supervisor(s.config, lambda: Backend(), runner_factory=Runner)
    try:
        with pytest.raises(RuntimeError, match="no retry"):
            reopened.start_interactive()
    finally:
        reopened.close()


def test_rehearsal_resets_and_uses_exploration_budget(supervisor, tmp_path):
    s, backend = supervisor
    s.start_interactive()
    backend.success = True
    s.finish_interactive()
    ident, _ = s.register(source(tmp_path, "controller.py"))
    s.run(ident, formal=False)
    assert backend.starts == [(False, 1), (False, 2)]
    assert s.state["exploration_started"] == 2 and not s.state["formal_reserved"]


def test_shutdown_failure_retains_evaluation_and_discloses_infrastructure(supervisor, tmp_path, monkeypatch):
    from services.controller.supervisor import formal_outcome
    s, backend = supervisor
    backend.success = True
    bundle, _ = s.register(source(tmp_path))
    def failed_close():
        raise RuntimeError('Cannot close /private/native/process')
    monkeypatch.setattr(backend, 'close', failed_close)
    with pytest.raises(RuntimeError, match='Cannot close'):
        s.run(bundle, formal=False)
    result = s.state['results'][-1]
    assert result['task_complete'] is True and formal_outcome(result) == 'success'
    assert s.state['active'] is None
    error = result['infrastructure_errors'][0]
    assert error['stage'] == 'environment_shutdown' and error['type'] == 'RuntimeError'
    assert '/private' not in error['reason']
    saved = json.loads(s.state_path.read_text())['results'][-1]
    assert saved['infrastructure_errors'] == result['infrastructure_errors']


def test_programming_available_without_manual_success(supervisor, tmp_path):
    s, backend = supervisor
    gateway, token = s.start_interactive()
    assert "gemini_generate" in {tool["name"] for tool in gateway.definitions()}
    # No configured provider, but no admission check or hidden API schema.
    assert "error" in gateway.handle(token, packet("gemini_generate", messages=[]))
    assert s.budget()["calls_used"] == 0
    assert "result" in gateway.handle(token, packet(STEP))
    s.finish_interactive()
    bundle, _ = s.register(source(tmp_path, "controller.py"))  # Registration only freezes files.
    assert s.run(bundle, formal=False)['returncode'] == 0
    assert not s.state['interactive_success']


def test_exact_clean_rehearsal_required_and_persisted(supervisor, tmp_path):
    s, backend = supervisor
    s.start_interactive()
    backend.success = True
    s.finish_interactive()
    src = source(tmp_path, "controller.py")
    bundle, manifest = s.register(src)
    with pytest.raises(RuntimeError, match="clean rehearsal"):
        s.run(bundle, formal=True)
    assert not s.state["formal_reserved"] and len(backend.starts) == 1
    # A clean run qualifies without task success: formal scores give partial credit.
    backend.success = False
    s.run(bundle, formal=False)
    assert s.state["results"][-1]["task_complete"] is False
    assert s._has_qualifying_rehearsal(bundle, manifest)
    (src / "controller.py").write_text("def main(ctx):\n    print('changed')\n")
    changed, _ = s.register(src)
    with pytest.raises(RuntimeError, match="clean rehearsal"):
        s.run(changed, formal=True)
    s.close()
    reopened = Supervisor(s.config, lambda: Backend(), runner_factory=Runner)
    try:
        assert reopened.run(bundle, formal=True)["returncode"] == 0
    finally:
        reopened.close()


@pytest.mark.parametrize("reason,code", [("timeout", None), ("exit", 1), ("episode_ended", 1), ("execution_error", None)])
def test_rehearsal_errors_do_not_qualify_despite_positive_evaluation(supervisor, tmp_path, reason, code):
    s, backend = supervisor
    s.start_interactive()
    backend.success = True
    s.finish_interactive()
    bundle, _ = s.register(source(tmp_path, "controller.py"))
    Runner.hook = lambda *_: {"reason": reason, "returncode": code}
    s.run(bundle, formal=False)
    with pytest.raises(RuntimeError, match="clean rehearsal"):
        s.run(bundle, formal=True)
    assert not s.state["formal_reserved"]


def test_repeated_exploration_layouts_are_accepted_but_formal_stays_held_out(tmp_path):
    config = configuration(tmp_path)
    assert replace(config, exploration_seeds=(1, 2, 1, 2)).exploration_seeds == (1, 2, 1, 2)
    with pytest.raises(ValueError, match="1..1000"):
        replace(config, exploration_seeds=(1,) * 1001)


def test_native_terminal_success_qualifies_but_simulator_exit_does_not(supervisor, tmp_path):
    s, backend = supervisor
    s.start_interactive()
    backend.success = True
    s.finish_interactive()
    bundle, manifest = s.register(source(tmp_path, "controller.py"))
    Runner.hook = lambda *_: {"reason": "episode_ended", "returncode": 0, "simulator_exited": "simulator process exited with code -11"}
    s.run(bundle, formal=False)
    assert not s._has_qualifying_rehearsal(bundle, manifest)
    Runner.hook = lambda *_: {"reason": "episode_ended", "returncode": 0, "simulator_exited": None}
    s.run(bundle, formal=False)
    assert s._has_qualifying_rehearsal(bundle, manifest)


def test_slow_exit_after_episode_end_is_evaluated_not_a_timeout(supervisor, tmp_path):
    s, backend = supervisor
    s.start_interactive()
    backend.success = True
    s.finish_interactive()
    bundle, manifest = s.register(source(tmp_path, "controller.py"))
    Runner.hook = lambda *_: {"reason": "finalization_timeout", "returncode": None}
    s.run(bundle, formal=False)
    result = s.state["results"][-1]
    assert result["reason"] == "episode_ended" and result["finalization_timeout"] is True
    assert result["task_complete"] is True and "timeout_stage" not in result
    assert s._has_qualifying_rehearsal(bundle, manifest)
    assert formal_outcome(result) == "success"


def test_gemini_author_prompt_images_usage_and_failure_reservation(unlocked_supervisor):
    s, backend = unlocked_supervisor
    captured = []
    def send(payload):
        captured.append(payload)
        return {"model": MODEL, "choices": [{"message": {"role": "assistant", "content": "custom response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10, "cost": 0.001}}
    s.gemini = GeminiRouter("secret-not-exposed", send=send)
    gateway, token = s.start_interactive()
    observed = gateway.handle(token, packet(OBSERVE))
    observation = value(observed)
    assert 'observation_id' not in observation
    before = s.budget()
    rejected = gateway.handle(token, packet('gemini_generate', messages=[{
        'role': 'user', 'content': [{'type': 'image', 'observation_id': 'removed', 'camera': CAMERA}]}]))
    assert 'error' in rejected and captured == []
    assert s.budget() == before
    assert "metadata" not in observation and "seed" not in observation and "manifest_path" not in observation
    messages = [{"role": "system", "content": "My controller's own prompt"}, {"role": "user", "content": [
        {"type": "text", "text": "Plan an action from this image"},
        next(block for block in observed['result']['content'] if block['type'] == 'image')]}]
    response = value(gateway.handle(token, packet("gemini_generate", messages=messages)))
    assert captured[0]["messages"][0] == messages[0]
    assert captured[0]["messages"][1]["content"][1]["image_url"]["url"].endswith(backend.image["data"])
    assert response["usage"]["total_tokens"] == 10 and response["cost_usd"] == 0.001
    assert response["budget"]["calls_used"] == 1 and response["budget"]["tokens_charged"] == 10
    def fail(payload):
        raise TimeoutError("secret provider diagnostic")
    s.gemini._send = fail
    error = gateway.handle(token, packet("gemini_generate", messages=messages))
    assert "secret" not in json.dumps(error)
    assert error["error"]["data"]["budget"]["unresolved_reserved_tokens"] > 0
    assert backend.step == 0


@pytest.mark.parametrize("extra", [{"model": "other"}, {"tools": [{"type": "web_search"}]}, {"api_key": "x"}, {"provider": {}}, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://localhost"}}]}]}])
def test_gemini_rejects_direct_access_and_overrides(extra):
    router = GeminiRouter("unused")
    args = {"messages": [{"role": "user", "content": "hello"}], **extra}
    with pytest.raises(ValueError):
        router.prepare(args)


PREVIEW_RUN = next((name for name in sorted(BENCH.robot_tools) if name.endswith('_preview_run')), None)


@pytest.mark.skipif(PREVIEW_RUN is None, reason=f'{BENCH.label} has no robot-only preview')
def test_preview_images_available_to_gemini(unlocked_supervisor):
    s, backend = unlocked_supervisor
    original_call = backend.call
    def call(name, arguments):
        observation, images = original_call(name, arguments)
        return ({"final_observation": observation} if name == PREVIEW_RUN else observation), images
    backend.call = call
    captured = []
    def send(payload):
        captured.append(payload)
        return {"model": MODEL, "choices": [{"message": {"content": "preview"}}],
                "usage": {"total_tokens": 10}}
    s.gemini = GeminiRouter("test", send=send)
    gateway, token = s.start_interactive()
    preview = gateway.handle(token, packet(PREVIEW_RUN))
    assert 'observation_id' not in value(preview)
    reply = gateway.handle(token, packet("gemini_generate", messages=[{"role": "user", "content": [
        next(block for block in preview['result']['content'] if block['type'] == 'image')]}]))
    assert "error" not in reply
    assert captured[0]["messages"][0]["content"][0]["image_url"]["url"].endswith(backend.image["data"])
    assert backend.step == 0


def test_snapshot_rejects_links_and_detects_mutation(tmp_path):
    src = source(tmp_path, "controller.py")
    (src / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError):
        snapshot(src, tmp_path / "bad", max_bytes=10000)
    (src / "escape").unlink()
    manifest = snapshot(src, tmp_path / "good", max_bytes=10000)
    verify(tmp_path / "good", manifest)
    os.chmod(tmp_path / "good/controller.py", 0o600)
    (tmp_path / "good/controller.py").write_text("changed")
    with pytest.raises(ValueError):
        verify(tmp_path / "good", manifest)


def test_bundle_command_has_no_host_execution_or_network(tmp_path):
    config = configuration(tmp_path)
    workspace = tmp_path / "agent-workspace"
    cmd = BundleSandbox(config.development).command(
        tmp_path / "bundle", "rehearsal", "controller.py", workspace=workspace, layout=tmp_path / "layout",
        writable=tmp_path / "output", sdk=tmp_path / "runtime.py")
    assert cmd[:6] == ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    assert "--share-net" not in cmd and "--clearenv" in cmd  # No network, no inherited environment.
    assert not any("docker.sock" in s or "nvidia" in s.lower() for s in cmd)  # Bundles run without a GPU.
    # The output directory is the only writable host path; the rest is read-only.
    assert [cmd[i + 1:i + 3] for i, s in enumerate(cmd) if s == "--bind"] == [[str(tmp_path / "output"), str(workspace / "output")]]
    assert [str(tmp_path / "bundle"), str(workspace / "code/controller")] in [cmd[i + 1:i + 3] for i, s in enumerate(cmd) if s == "--ro-bind"]
    assert cmd[-5:] == ["-u", str(workspace / "api/runtime.py"), "rehearsal", str(workspace / "code/controller"), "controller.py"]


@pytest.mark.parametrize('path', ['code', 'runtime/project', 'api/project', 'output/project',
    '/workspace/code/project', '../code/project', 'code/../api', 'code//project',
    'code/./project', 'code/project/', 'code/a,b', 'code/a\nb', 'code/a\\b', ''])
def test_registered_project_path_is_canonical(path):
    from services.controller.storage import project_directory
    with pytest.raises(ValueError):
        project_directory(path)


def test_registered_project_path_affects_bundle_hash(tmp_path):
    from services.controller.storage import project_directory
    assert str(project_directory('code/nested/reach')) == 'code/nested/reach'
    src = source(tmp_path, 'controller.py')
    first = snapshot(src, tmp_path/'first', max_bytes=10000,
                     workspace_path='code/nested/reach')
    second = snapshot(src, tmp_path/'second', max_bytes=10000,
                      workspace_path='code/other')
    assert first['workspace_path'] == 'code/nested/reach'
    assert first['sha256'] != second['sha256']


def test_deadline_is_not_caught_as_tool_error():
    import time
    with pytest.raises(ExecutionDeadline):
        with deadline(0.01):
            try:
                time.sleep(1)
            except Exception:
                pytest.fail("Deadline incorrectly swallowed")


def test_rehearsal_timeout_clamps_startup_and_returns_error_artifacts(supervisor, tmp_path, monkeypatch):
    import time
    from services.controller.frontend import ResearchFrontend
    s, backend = supervisor
    object.__setattr__(s.config, 'formal', replace(s.config.formal, wall_seconds=1))
    original = backend.start
    def slow_start(**kwargs):
        original(**kwargs)
        time.sleep(5)
    monkeypatch.setattr(backend, 'start', slow_start)
    ident, manifest = s.register(source(tmp_path))
    frontend = ResearchFrontend(s, tmp_path/'workspace')
    started = time.monotonic()
    reply = frontend.call('rehearse', {'bundle': ident})
    assert time.monotonic()-started < 3
    value = json.loads(reply['content'][0]['text'])
    assert reply['isError'] and value['reason'] == 'timeout'
    assert value['error_type'] == 'TimeoutError' and value['timeout_seconds'] == 1
    assert value['task_complete'] is None and value['artifacts']['mcp_trace']
    assert value['timeout_stage'] == 'environment_setup'
    assert value['timing']['environment_setup_seconds'] >= .9
    assert value['timing']['environment_step_seconds'] == 0
    assert backend.closed and s.backend is None and s.state['active'] is None
    assert not s._has_qualifying_rehearsal(ident, manifest)


@pytest.mark.parametrize('legacy_development_seconds', [1, 4000])
def test_rehearsal_uses_the_formal_episode_wall_limit(supervisor, tmp_path, legacy_development_seconds):
    s, _ = supervisor
    object.__setattr__(s.config, 'development', replace(s.config.development, wall_seconds=legacy_development_seconds))
    object.__setattr__(s.config, 'formal', replace(s.config.formal, wall_seconds=600))
    seen = []
    class CaptureRunner(Runner):
        def run(self, bundle, **kwargs):
            seen.append((self.limits.wall_seconds, kwargs['wall_seconds']))
            return super().run(bundle, **kwargs)
    s.runner_factory = CaptureRunner
    ident, _ = s.register(source(tmp_path))
    s.run(ident, formal=False)
    assert seen[0][0] == 600 and 0 < seen[0][1] <= 600


def test_episode_without_a_wall_limit_runs_to_its_end(supervisor, tmp_path):
    s, _ = supervisor
    object.__setattr__(s.config, 'formal', replace(s.config.formal, wall_seconds=None))
    seen = []
    class CaptureRunner(Runner):
        def run(self, bundle, **kwargs):
            seen.append((self.limits.wall_seconds, kwargs['wall_seconds']))
            return super().run(bundle, **kwargs)
    s.runner_factory = CaptureRunner
    ident, _ = s.register(source(tmp_path))
    result = s.run(ident, formal=False)
    assert seen == [(None, None)]
    assert result['reason'] == 'exit' and result['timing']['wall_limit_seconds'] is None


def test_stages_without_a_wall_limit_keep_their_own_limits(supervisor, tmp_path, monkeypatch):
    import time
    from services.controller import supervisor as module
    from services.controller.frontend import ResearchFrontend
    s, backend = supervisor
    object.__setattr__(s.config, 'formal', replace(s.config.formal, wall_seconds=None))
    real = module.for_config
    monkeypatch.setattr(module, 'for_config', lambda config: replace(real(config), startup_seconds=1))
    monkeypatch.setattr(module, 'EVALUATION_SECONDS', 1)
    original = backend.start
    def slow_start(**kwargs):
        original(**kwargs)
        time.sleep(5)
    monkeypatch.setattr(backend, 'start', slow_start)
    ident, _ = s.register(source(tmp_path))
    frontend = ResearchFrontend(s, tmp_path/'workspace')
    started = time.monotonic()
    value = json.loads(frontend.call('rehearse', {'bundle': ident})['content'][0]['text'])
    assert time.monotonic()-started < 3
    assert value['reason'] == 'timeout' and value['timeout_stage'] == 'environment_setup'
    assert value['timeout_seconds'] == 1 and value['timing']['wall_limit_seconds'] is None
    monkeypatch.setattr(backend, 'start', original)
    def slow_evaluation():
        time.sleep(5)
        return {'task_complete': True}
    monkeypatch.setattr(backend, 'evaluate', slow_evaluation)
    started = time.monotonic()
    value = json.loads(frontend.call('rehearse', {'bundle': ident})['content'][0]['text'])
    assert time.monotonic()-started < 3
    assert value['reason'] == 'timeout' and value['timeout_stage'] == 'evaluation'
    assert value['timeout_seconds'] == 1 and value['task_complete'] is None


def test_wall_limit_none_is_valid_but_zero_is_not():
    assert Limits(None, 512, 1, 32, 32, 1024).wall_seconds is None
    with pytest.raises(ValueError, match='wall_seconds'):
        Limits(0, 512, 1, 32, 32, 1024)


def test_exhausted_legacy_time_budget_does_not_block_rehearsals_after_restart(tmp_path):
    config = configuration(tmp_path)
    backend = Backend()
    s = Supervisor(config, lambda: backend, runner_factory=Runner)
    ident, _ = s.register(source(tmp_path))
    s.state['development_reserved_seconds'] = 999999
    s.state['configuration']['training_seconds'] = 3600
    s._save()
    s.close()
    restored = Supervisor(config, lambda: backend, runner_factory=Runner)
    try:
        assert 'development_reserved_seconds' not in restored.state
        assert 'training_seconds' not in restored.state['configuration']
        for _ in config.exploration_seeds:
            assert restored.run(ident, formal=False)['reason'] == 'exit'
        assert restored.state['exploration_started'] == len(config.exploration_seeds)
        with pytest.raises(RuntimeError, match='No exploration episode remains'):
            restored.run(ident, formal=False)
        assert 'development_reserved_seconds' not in json.loads(restored.state_path.read_text())
    finally:
        restored.close()


def test_evaluation_cannot_extend_rehearsal_deadline(supervisor, tmp_path, monkeypatch):
    import time
    s, backend = supervisor
    object.__setattr__(s.config, 'formal', replace(s.config.formal, wall_seconds=1))
    def slow_evaluation():
        time.sleep(5)
        return {'task_complete': True}
    monkeypatch.setattr(backend, 'evaluate', slow_evaluation)
    ident, manifest = s.register(source(tmp_path))
    result = s.run(ident, formal=False)
    assert result['reason'] == 'timeout' and result['task_complete'] is None
    assert result['timeout_stage'] == 'evaluation'
    assert result['timing']['evaluation_seconds'] >= .9
    assert backend.closed and not s._has_qualifying_rehearsal(ident, manifest)


@pytest.mark.parametrize('formal', [False, True])
@pytest.mark.parametrize('interrupt_motion', [False, True])
def test_episode_timings_partition_wall_time_and_persist(supervisor, tmp_path, monkeypatch, formal, interrupt_motion):
    import time
    s, backend = supervisor
    original = backend.call
    def call(name, arguments):
        if name == STEP:
            time.sleep(.015)
            if interrupt_motion:
                raise ExecutionDeadline()
        return original(name, arguments)
    monkeypatch.setattr(backend, 'call', call)
    def execution(bundle, job):
        time.sleep(.01)  # Local controller work, not MCP waiting.
        job['gateway'].handle(job['token'], packet(OBSERVE))
        job['gateway'].handle(job['token'], packet(STEP))
        return {'reason': 'exit', 'returncode': 0}
    Runner.hook = execution
    ident, _ = s.register(source(tmp_path))
    result = s._run_episode(ident, formal=formal)
    t = result['timing']
    assert t['schema'] == 'episode_wall_timing_v1'
    assert t['environment_setup_seconds'] > 0
    assert t['controller_processing_seconds'] >= .009
    assert t['environment_step_seconds'] >= .014
    assert t['environment_other_mcp_seconds'] > 0
    assert t['gemini_api_seconds'] == 0
    assert t['mcp_tools'][STEP]['calls'] == 1
    # Supervisor boundary observations must not be charged as controller calls.
    assert t['mcp_tools'][OBSERVE]['calls'] == 1
    components = [v for k, v in t.items() if k.endswith('_seconds') and k not in {'total_seconds', 'wall_limit_seconds'}]
    assert all(v >= 0 for v in components)
    assert sum(components) == pytest.approx(t['total_seconds'])
    saved = json.loads((s.config.root/'published/runs'/result['run_id']/'result.json').read_text())
    assert saved['timing'] == t
    if interrupt_motion:
        assert result['reason'] == 'timeout'
        assert result['timeout_stage'] == 'controller_execution'


PLANNER = next(iter(sorted(BENCH.previewable)), None)


@pytest.mark.skipif(PLANNER is None, reason=f'{BENCH.label} has no planning-only preview')
def test_gateway_timing_separates_preview_and_records_api_errors(supervisor, monkeypatch):
    s, _ = supervisor
    gateway, token = s.start_interactive()
    gateway.handle(token, packet(PLANNER, preview_only=True))
    gateway.handle(token, packet('gemini_generate', messages=[]))  # No provider configured.
    assert gateway.timings.calls[f'{PLANNER}_preview'] == 1
    assert PLANNER not in gateway.timings.calls
    assert gateway.timings.calls['gemini_generate'] == 1
    assert gateway.timings.seconds['gemini_generate'] > 0


def test_nested_deadline_does_not_restart_outer_budget():
    import time
    started = time.monotonic()
    with pytest.raises(ExecutionDeadline):
        with deadline(.15):
            with deadline(2):
                time.sleep(.08)
            time.sleep(.1)
    assert time.monotonic()-started < .22


def test_no_wall_limit_still_bounds_finalization():
    import signal
    import time
    from services.controller.sandbox import finalize_within, time_left, wall_end
    assert wall_end(None) is None and time_left(None) is None
    assert 4 < time_left(wall_end(None, 5)) <= 5 and 4 < time_left(wall_end(60, 5)) <= 5
    started = time.monotonic()
    with pytest.raises(ExecutionDeadline):
        with deadline(None):
            assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
            finalize_within(.1)
            time.sleep(2)
    assert time.monotonic()-started < .5 and signal.getitimer(signal.ITIMER_REAL)[0] == 0
    # Within a limit, finalization never extends it.
    started = time.monotonic()
    with pytest.raises(ExecutionDeadline):
        with deadline(.1):
            finalize_within(5)
            time.sleep(2)
    assert time.monotonic()-started < .5


def test_component_returns_locations_not_logs_or_images(unlocked_supervisor, tmp_path):
    s, _ = unlocked_supervisor
    s.start_interactive()
    def execution(bundle, job):
        out = job["output"]
        out.mkdir()
        (out / "files").mkdir()
        (out / "stdout.log").write_text("specific-debug-point")
        (out / "stderr.log").write_text("specific-error-point")
        (out / "logs.json").write_text('{"stdout":{"truncated":false}}')
        job["gateway"].handle(job["token"], packet(STEP))
        return {"reason": "exit", "returncode": 0}
    Runner.hook = execution
    s.finish_interactive()
    ident, _ = s.register(source(tmp_path))
    result = s.run(ident, formal=False)
    serialized = json.dumps(result)
    assert "specific-debug-point" not in serialized and "specific-error-point" not in serialized
    assert '"data"' not in serialized and not result["recording_errors"]
    def artifact(path):
        return s.published_root / Path(path).relative_to('runtime/autonomous_controller')
    paths = result["artifacts"]
    assert artifact(paths["stdout"]).read_text() == "specific-debug-point"
    assert artifact(paths["stderr"]).read_text() == "specific-error-point"
    assert (artifact(paths["code"]) / "controller.py").is_file()
    trace = [json.loads(line) for line in artifact(paths["mcp_trace"]).read_text().splitlines()]
    actions = [e for e in trace if e.get("tool") == STEP]
    assert actions[0]["call_id"] == actions[1]["call_id"]
    assert artifact(actions[1]["response_path"]).is_file()
    assert artifact(actions[1]["images"][0]["path"]).read_bytes().startswith(b"\x89PNG")
    assert result["start_step_id"] == 0 and result["end_step_id"] == 1


def test_frame_sequences_published_and_rebased_without_private_data(tmp_path):
    from services.controller.recording import RunRecording
    workspace = tmp_path / "native"
    original = Path("runtime/frames/sim/run/seq")
    folder = workspace / original
    folder.mkdir(parents=True)
    filename = f"frame_000000_{CAMERA}_rgb.png"
    (folder / filename).write_bytes(b"png-bytes")
    manifest = {"frequency_hz": 25, "frame_count": 1, "seed": 91,
                "transition": {"native_success": False, "native_score": 0},
                "frames": [{"step_id": 42, "states": [0] * 14,
                            "files": [{"kind": "rgb", "camera": CAMERA, "path": str(original / filename)}]}]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    recording = RunRecording(tmp_path / "published", "operate-" + "a" * 32)
    value = {"frame_sequence": {"frequency_hz": 25, "manifest_path": str(original / "manifest.json"),
             "frame_path_pattern": str(original / "frame_{frame_index:06d}_{camera}_{kind}.png")}}
    published = recording.sequence(value, workspace)
    relative = Path(published["manifest_path"]).relative_to("runtime/autonomous_controller")
    exported = json.loads((tmp_path / "published" / relative).read_text())
    assert "seed" not in exported and "native_score" not in json.dumps(exported)
    assert exported["frames"][0]["step_id"] == 42
    framepath = Path(exported["frames"][0]["files"][0]["path"]).relative_to("runtime/autonomous_controller")
    assert (tmp_path / "published" / framepath).read_bytes() == b"png-bytes"
    assert recording.sequence(value, workspace) == published


def test_frame_publication_rejects_escape_and_symlinks(tmp_path):
    from services.controller.recording import RunRecording
    recording = RunRecording(tmp_path / "published", "operate-" + "a" * 32)
    for path in ("/etc/passwd", "runtime/frames/../../operator/manifest.json"):
        with pytest.raises(ValueError):
            recording.sequence({"frame_sequence": {"manifest_path": path}}, tmp_path)
    folder = tmp_path / "runtime/frames/seq"
    folder.mkdir(parents=True)
    (folder / "manifest.json").symlink_to("/etc/passwd")
    with pytest.raises(OSError):
        recording.sequence({"frame_sequence": {"manifest_path": "runtime/frames/seq/manifest.json"}}, tmp_path)


def test_rehearsal_without_an_evaluation_does_not_qualify(supervisor, tmp_path):
    s, backend = supervisor
    s.start_interactive()
    s.finish_interactive()
    bundle, manifest = s.register(source(tmp_path, "controller.py"))
    Runner.hook = lambda *_: {"reason": "exit", "returncode": 0, "task_complete": None}
    s.run(bundle, formal=False)
    s.state["results"][-1]["task_complete"] = None  # Evaluation never decided.
    assert not s._has_qualifying_rehearsal(bundle, manifest)


def test_a_trial_saved_with_a_claude_account_choice_reopens(tmp_path):
    """Sessions saved while claude-login had an account pool carry agent_account in all three
    saved copies; loading drops it and the supervisor's no-change check still passes."""
    config = replace(configuration(tmp_path), agent_cli='claude', agent_provider='claude-login')
    Supervisor(config, lambda: Backend(), runner_factory=Runner).close()
    state_path = config.root/'state.json'
    state = json.loads(state_path.read_text())
    state['configuration']['agent_account'] = 'second'
    state_path.write_text(json.dumps(state))
    saved = {**json.loads(json.dumps(asdict(config), default=str)), 'agent_account': 'second'}
    loaded = Configuration.from_dict(saved)
    assert 'agent_account' not in asdict(loaded) and loaded.agent_provider == 'claude-login'
    reopened = Supervisor(loaded, lambda: Backend(), runner_factory=Runner)
    reopened._save()
    reopened.close()
    assert 'agent_account' not in json.loads(state_path.read_text())['configuration']
    with pytest.raises(ValueError, match="saved configuration cannot change"):
        Supervisor(replace(loaded, agent_provider='openrouter'), lambda: Backend(), runner_factory=Runner)


def test_recording_has_no_session_or_run_size_limit(supervisor):
    s, _ = supervisor
    s.start_interactive()
    run = s.config.development.artifact_bytes  # Once each run's cap; scaled, the session's.
    s.state["audit_bytes"] = 10**12
    image = base64.b64encode(b"x" * (run + 1)).decode()
    s.audit({"event": "mcp_result", "tool": "probe",
             "result": {"content": [{"type": "text", "text": "{}"}, {"type": "image", "data": image}]}})
    assert s.state["audit_bytes"] > 10**12 and not (s.gateway and s.gateway.terminal)
    call = f"{s._recording.calls:06d}"  # This probe's published response.
    published = s.published_root / "runs" / s._recording.run_id / "calls" / call / "image_01.png"
    assert published.stat().st_size == run + 1
    assert json.loads((s.config.root / "mcp.jsonl").read_text().splitlines()[-1])["tool"] == "probe"


def test_frame_mount_only_exposes_current_run_frames(tmp_path):
    from services.controller.recording import RunRecording
    record = RunRecording(tmp_path, 'rehearsal-'+'a'*32)
    runner = BundleSandbox(Limits(30, 128, 1, 32, 16, 10000))
    paths = dict(workspace=tmp_path/'agent-workspace', layout=tmp_path/'layout', writable=tmp_path/'output',
                 sdk=tmp_path/'runtime.py')
    command = runner.command(tmp_path/'bundle', 'rehearsal', 'controller.py', frames=record.frames, **paths)
    assert [str(record.frames), str(tmp_path/'agent-workspace'/record.logical/'frames')] in [
        command[i + 1:i + 3] for i, s in enumerate(command) if s == '--ro-bind']
    for directory in (record.root, tmp_path, tmp_path/'missing'):
        with pytest.raises(ValueError):
            runner.command(tmp_path/'bundle', 'rehearsal', 'controller.py', frames=directory, **paths)


def test_an_episode_left_open_by_a_killed_launcher_keeps_its_native_id(tmp_path):
    """A launcher killed mid-episode (kill -9, host reboot) leaves only state.json's active
    record; the next launcher records that episode as interrupted under its native ID."""
    config = configuration(tmp_path)
    s = Supervisor(config, lambda: Backend(), runner_factory=Runner)
    s.start_interactive()
    left = (config.root/'state.json').read_text()
    assert json.loads(left)['active'] == {'mode': 'interactive', 'episode': 1, 'episode_id': 'episode'}
    s.close()
    (config.root/'state.json').write_text(left)  # As the kill left it: no close ran.
    reopened = Supervisor(config, lambda: Backend(), runner_factory=Runner)
    try:
        [result] = reopened.state['results']
        assert result == {'mode': 'interactive', 'episode': 1, 'episode_id': 'episode',
                          'reason': 'supervisor_interrupted', 'task_complete': None}
        assert reopened.state['active'] is None
    finally:
        reopened.close()


def test_a_start_past_its_deadline_reports_the_deadline_not_its_cleanup(tmp_path, monkeypatch):
    """An environment that does not start within its limit (a heavy scene) is reported as
    exactly that, even when closing the half-started simulator then fails, as a simulator may with
    no episode yet; the deadline itself is a BaseException that no tool handler would report."""
    import time
    from services.controller import supervisor as module
    config = configuration(tmp_path)
    spec = module.for_config(config)
    monkeypatch.setattr(module, 'for_config', lambda _: replace(spec, startup_seconds=0.2))
    backend = Backend()
    backend.start = lambda **_: time.sleep(5)
    def close():
        raise RuntimeError('No active episode')
    backend.close = close
    s = Supervisor(config, lambda: backend, runner_factory=Runner)
    try:
        with pytest.raises(TimeoutError) as raised:
            s.start_interactive()
        assert str(raised.value) == ('The environment did not start within its 0.2s startup limit; '
                                     'closing it then failed: RuntimeError: No active episode')
        assert s.state['active'] is None and [r['episode'] for r in s.state['results']] == [1]
    finally:
        s.close()
