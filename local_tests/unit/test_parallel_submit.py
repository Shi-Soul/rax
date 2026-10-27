"""Parallel agent submission: one worker per layout, one shared Gemini ledger, no retry."""
import json
from pathlib import Path

import pytest

from test_bundle_sandbox import sandbox_runtime
from services.controller import formal_worker
from services.controller.frontend import ResearchFrontend
from services.controller.gemini import GeminiRouter, MODEL
from services.controller.supervisor import Supervisor
from test_controller_backend import Backend, Runner, source
from test_formal_batch import config
from bench import OBSERVE


def test_worker_bootstrap_reads_parent_state_through_replace_gap(tmp_path, monkeypatch):
    import errno
    import os
    import signal
    import sys
    from types import SimpleNamespace

    s, _, bundle = rehearsed(tmp_path)
    state_path = s.config.root/'state.json'
    original_read = Path.read_text
    reads, started = [], []

    def interrupted_read(path, *args, **kwargs):
        if path == state_path:
            reads.append(path)
            if len(reads) == 1:
                raise FileNotFoundError(errno.ENOENT, 'Replacement in progress', str(path))
        return original_read(path, *args, **kwargs)

    def run(parent, bundle_id, index, channel):
        started.append((parent.root, bundle_id, index))
        return {'reason': 'exit', 'returncode': 0, 'task_complete': False}

    monkeypatch.setattr(Path, 'read_text', interrupted_read)
    monkeypatch.setattr(formal_worker, 'ctypes', SimpleNamespace(
        CDLL=lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: 0)))
    monkeypatch.setattr(formal_worker, 'signal', SimpleNamespace(
        SIGTERM=signal.SIGTERM, SIG_IGN=signal.SIG_IGN, signal=lambda *args: None))
    monkeypatch.setattr(formal_worker, 'socket', SimpleNamespace(socket=lambda **kwargs: None))
    monkeypatch.setattr(formal_worker, 'Channel', lambda connection: None)
    monkeypatch.setattr(formal_worker, 'run_worker', run)
    monkeypatch.setattr(sys, 'argv', ['formal_worker', '--parent-root', str(s.config.root),
        '--bundle', bundle, '--index', '0', '--channel-fd', '1', '--parent-pid', str(os.getppid())])
    try:
        formal_worker.main()
        assert len(reads) == 2
        assert started == [(s.config.root, bundle, 0)]
    finally:
        s.close()


class Finished:
    """A worker that already ran to completion inside the factory."""
    pid = None

    def __init__(self, code=0):
        self.code = code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        pass


@pytest.fixture(autouse=True)
def reset_runner_hook():
    yield
    Runner.hook = None


def rehearsed(tmp_path, episodes=10, workers=3, gemini=None, factory=None):
    backend = Backend()
    c = config(tmp_path, formal_episodes=episodes, formal_workers=workers)
    s = Supervisor(c, lambda: backend, gemini, runner_factory=Runner, worker_factory=factory)
    bundle, _ = s.register(source(tmp_path, 'controller.py'))
    backend.success = True
    s.run(bundle, formal=False)
    return s, backend, bundle


def inline(s, backend, launched):
    def factory(root, bundle_id, index, connection):
        assert root == s.config.root
        launched.append(index)
        formal_worker.run_worker(s.config, bundle_id, index, formal_worker.Channel(connection),
                                 backend_factory=lambda: backend, runner_factory=Runner)
        return Finished()
    return factory


def test_parallel_batch_counts_every_layout_once_with_fixed_denominator(tmp_path):
    launched = []
    s, backend, bundle = rehearsed(tmp_path)
    s.worker_factory = inline(s, backend, launched)
    try:
        def run(_, job):
            index = len(backend.starts) - 2
            backend.success = index % 5 != 1
            return {'reason': 'exit', 'returncode': 1 if index % 5 == 0 else 0}
        Runner.hook = run
        result = s.run(bundle, formal=True)
        assert launched == list(range(10))
        assert backend.starts[1:] == [(True, i) for i in range(10)]
        assert result['completed_episodes'] == result['episode_count'] == 10
        assert (result['success_count'], result['unsuccessful_count'], result['error_count']) == (6, 2, 2)
        assert result['success_rate'] == .6 and result['status'] == 'completed'
        formal = [r for r in s.state['results'] if r['mode'] == 'formal']
        assert sorted(r['formal_episode_index'] for r in formal) == list(range(1, 11))
        assert {r['bundle'] for r in formal} == {bundle}
        assert sorted(r['layout_id'] for r in formal) == list(range(10))
        # Workers publish into the parent's agent-visible tree; their bundle copies are removed.
        for r in formal:
            assert (s.config.root/'published'/r['artifacts']['directory'].removeprefix('runtime/autonomous_controller/')).is_dir()
        assert not any((s.config.root/'formal-workers').glob('*/bundles'))
        report = json.loads((s.config.root/'published/formal_batch.json').read_text())
        assert [e['formal_episode_index'] for e in report['episodes']] == list(range(1, 11))
        assert 'formal_worker_pids' not in s.state and s.state['formal_batch']['active_episodes'] == []
        assert ResearchFrontend.public_result(result)['success_rate'] == .6
        with pytest.raises(RuntimeError, match='no retry'):
            s.run(bundle, formal=True)
    finally:
        s.close()


def test_workers_share_the_parent_gemini_ledger(tmp_path):
    sent = []
    def send(payload):
        sent.append(payload)
        return {'model': MODEL, 'choices': [{'message': {'role': 'assistant', 'content': 'OK'},
                'finish_reason': 'stop'}], 'usage': {'total_tokens': 10, 'cost': .002}}
    launched = []
    s, backend, bundle = rehearsed(tmp_path, episodes=4, workers=2, gemini=GeminiRouter('host-key', send=send))
    s.worker_factory = inline(s, backend, launched)
    try:
        def run(_, job):
            reply = job['gateway'].handle(job['token'], {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': 'gemini_generate', 'arguments': {
                    'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 64}}})
            value = json.loads(reply['result']['content'][0]['text'])
            assert value['message']['content'] == 'OK' and value['budget']['phase'] == 'formal'
            return {'reason': 'exit', 'returncode': 0}
        Runner.hook = run
        result = s.run(bundle, formal=True)
        assert len(sent) == 4 and result['success_count'] == 4
        assert s.state['usage']['formal']['calls'] == 4
        assert s.state['gemini_spend']['reserved'] == 0
        assert result['api_budget']['session_cost_reported_usd'] == pytest.approx(.008)
    finally:
        s.close()


def test_a_workers_http_error_releases_on_the_parent_ledger_and_logs_in_the_parent(tmp_path):
    import io
    import urllib.error
    body = b'{"error":{"message":"Insufficient credits","code":402}}'
    def send(payload):
        raise urllib.error.HTTPError('https://openrouter.ai/api/v1/chat/completions', 402, 'Payment Required',
                                     {}, io.BytesIO(body))
    launched = []
    s, backend, bundle = rehearsed(tmp_path, episodes=2, workers=2, gemini=GeminiRouter('host-key', send=send))
    s.worker_factory = inline(s, backend, launched)
    reasons = []
    try:
        def run(_, job):
            reply = job['gateway'].handle(job['token'], {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                'params': {'name': 'gemini_generate', 'arguments': {
                    'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 64}}})
            reasons.append(reply['error']['data']['reason'])
            return {'reason': 'exit', 'returncode': 0}
        Runner.hook = run
        s.run(bundle, formal=True)
        assert reasons == ['Provider returned HTTP 402 (Payment Required); it produced no completion, '
                           'so its billing reservation was released.'] * 2
        assert s.state['usage']['formal']['calls'] == 2 and s.state['gemini_spend']['reserved'] == 0
        records = [json.loads(line) for line in (s.config.root/'provider-errors.jsonl').read_text().splitlines()]
        assert [(r['status'], r['body'], r['worker_channel']) for r in records] == [(402, body.decode(), True)] * 2
    finally:
        s.close()


def test_completed_worker_evidence_is_saved_before_removing_recovery_pointer(tmp_path, monkeypatch):
    s, backend, bundle = rehearsed(tmp_path, episodes=2, workers=2)
    s.worker_factory = inline(s, backend, [])
    save = s._save
    def checked_save():
        batch = s.state.get('formal_batch', {})
        recorded = {r.get('formal_episode_index') for r in s.state['results']}
        for index in range(2):
            path = formal_worker.worker_root(s.config.root, index)/'state.json'
            if path.exists() and any(r['mode'] == 'formal' for r in json.loads(path.read_text())['results']):
                assert index+1 in batch.get('active_episodes', []) or index+1 in recorded
        save()
    monkeypatch.setattr(s, '_save', checked_save)
    try:
        assert s.run(bundle, formal=True)['success_count'] == 2
    finally:
        s.close()


def started_without_result(s, bundle, index):
    """Worker state as left by a worker killed after reserving its episode."""
    root = formal_worker.worker_root(s.config.root, index)
    root.mkdir(parents=True)
    active = {'mode': 'formal', 'bundle': 'imported', 'sha256': s.state['bundles'][bundle]['sha256'],
              'layout_id': s.config.formal_seed + index, 'eval_seed': s.config.formal_collection}
    (root/'state.json').write_text(json.dumps({'results': [], 'active': active}))


@pytest.mark.parametrize('failure', ['spawn', 'no_record'])
def test_systemic_worker_failure_interrupts_like_sequential(tmp_path, failure):
    launched = []
    s, backend, bundle = rehearsed(tmp_path, episodes=4, workers=2)
    good = inline(s, backend, launched)
    def factory(root, bundle_id, index, connection):
        if index == 1:
            launched.append(index)
            if failure == 'spawn':
                raise OSError('cannot spawn')
            return Finished(1)  # e.g. preflight/import failed before any episode started
        return good(root, bundle_id, index, connection)
    s.worker_factory = factory
    try:
        with pytest.raises((OSError, RuntimeError)):
            s.run(bundle, formal=True)
        batch = s.state['formal_batch']
        assert batch['status'] == 'interrupted' and batch['success_rate'] is None
        assert 2 not in launched and 3 not in launched  # Nothing launched after the failure.
        with pytest.raises(RuntimeError, match='no retry'):
            s.run(bundle, formal=True)
    finally:
        s.close()


def test_started_but_lost_episode_counts_as_error(tmp_path):
    launched = []
    s, backend, bundle = rehearsed(tmp_path, episodes=3, workers=2)
    good = inline(s, backend, launched)
    def factory(root, bundle_id, index, connection):
        if index == 1:
            started_without_result(s, bundle, index)
            return Finished(-9)
        return good(root, bundle_id, index, connection)
    s.worker_factory = factory
    try:
        result = s.run(bundle, formal=True)
        assert result['status'] == 'completed' and result['error_count'] == 1
        assert result['success_count'] == 2 and result['success_rate'] == pytest.approx(2/3)
        lost = next(r for r in s.state['results'] if r.get('formal_episode_index') == 2)
        assert lost['reason'] == 'worker_lost' and lost['formal_outcome'] == 'error'
    finally:
        s.close()


@pytest.mark.parametrize('success', [True, False])
@pytest.mark.parametrize('exit_code', [1, -9])
def test_post_result_worker_crash_preserves_task_outcome_and_reports_infra(tmp_path, success, exit_code):
    s, backend, bundle = rehearsed(tmp_path, episodes=2, workers=2)
    backend.success = success
    launched = []
    good = inline(s, backend, launched)
    def factory(root, bundle_id, index, connection):
        good(root, bundle_id, index, connection)
        return Finished(exit_code)  # Trusted episode result already durably saved.
    s.worker_factory = factory
    try:
        result = s.run(bundle, formal=True)
        assert launched == [0, 1]
        assert result['status'] == 'completed' and result['completed_episodes'] == 2
        assert result['success_rate'] == (1.0 if success else 0.0)
        assert result['error_count'] == 0 and result['infrastructure_error_count'] == 2
        assert ResearchFrontend.public_result(result)['infrastructure_error_count'] == 2
        report = json.loads((s.published_root/'formal_batch.json').read_text())
        for row in report['episodes']:
            assert row['task_complete'] is success
            assert row['formal_outcome'] == ('success' if success else 'unsuccessful')
            assert row['worker_returncode'] == exit_code
            error = row['infrastructure_errors'][0]
            assert error['stage'] == 'worker_exit' and error['episode_record_finalized'] is True
        assert report['infrastructure_error_count'] == 2
    finally:
        s.close()


def test_watchdog_leaves_workers_running_without_a_wall_limit(tmp_path, monkeypatch):
    from dataclasses import replace
    from services.controller import supervisor as module
    monkeypatch.setattr(module, 'WORKER_GRACE_SECONDS', -10**6)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    s, backend, bundle = rehearsed(tmp_path, episodes=2, workers=2)
    object.__setattr__(s.config, 'formal', replace(s.config.formal, wall_seconds=None))
    signals, polls = [], []
    class Long:
        pid = None
        def poll(self):
            polls.append(1)
            return None if len(polls) < 50 else -9
        def terminate(self):
            signals.append('TERM')
        def kill(self):
            signals.append('KILL')
        def wait(self, timeout=None):
            return self.poll()
    def factory(root, bundle_id, index, connection):
        started_without_result(s, bundle, index)
        return Long()
    s.worker_factory = factory
    try:
        s.run(bundle, formal=True)
        # A wall limit would have stopped these workers on their first poll.
        assert signals == [] and len(polls) >= 50
    finally:
        s.close()


def test_watchdog_terminates_once_then_kills(tmp_path, monkeypatch):
    from services.controller import supervisor as module
    monkeypatch.setattr(module, 'WORKER_GRACE_SECONDS', -10**6)
    monkeypatch.setattr(module, 'WORKER_KILL_SECONDS', -1)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    s, backend, bundle = rehearsed(tmp_path, episodes=2, workers=2)
    signals = []
    class Hung:
        pid = None
        code = None
        def poll(self):
            return self.code
        def terminate(self):
            signals.append('TERM')
        def kill(self):
            signals.append('KILL')
            self.code = -9
        def wait(self, timeout=None):
            return self.code
    def factory(root, bundle_id, index, connection):
        started_without_result(s, bundle, index)
        return Hung()
    s.worker_factory = factory
    try:
        result = s.run(bundle, formal=True)
        assert signals == ['TERM', 'TERM', 'KILL', 'KILL']
        assert result['error_count'] == 2 and result['success_rate'] == 0
    finally:
        s.close()


def test_restart_recovers_started_worker_evidence_without_retry(tmp_path):
    s, backend, bundle = rehearsed(tmp_path, episodes=4, workers=2)
    started_without_result(s, bundle, 2)
    s.state['formal_reserved'] = True
    s.state['formal_batch'] = {'bundle': bundle, 'sha256': s.state['bundles'][bundle]['sha256'],
        'status': 'running', 'episode_count': 4, 'completed_episodes': 0, 'success_count': 0,
        'unsuccessful_count': 0, 'error_count': 0, 'success_rate': None, 'active_episodes': [3]}
    s.state['formal_worker_pids'] = {'3': 999999999}
    s.state['configuration']['root'] = str(s.config.root)
    s._save()
    s.close()
    restarted = Supervisor(s.config, Backend, runner_factory=Runner)
    try:
        batch = restarted.state['formal_batch']
        assert batch['status'] == 'interrupted' and 'formal_worker_pids' not in restarted.state
        row = next(r for r in restarted.state['results'] if r.get('mode') == 'formal')
        assert row['formal_episode_index'] == 3 and row['formal_outcome'] == 'interrupted'
        with pytest.raises(RuntimeError, match='no retry'):
            restarted.run(bundle, formal=True)
    finally:
        restarted.close()


def finalized(s, bundle, index, **extra):
    return {'mode': 'formal', 'bundle': 'imported', 'sha256': s.state['bundles'][bundle]['sha256'],
            'layout_id': s.config.formal_seed + index, 'eval_seed': s.config.formal_collection,
            'reason': 'episode_ended', 'returncode': 0, 'task_complete': True, **extra}


def crashed(s, bundle, active):
    s.state['formal_reserved'] = True
    s.state['formal_batch'] = {'bundle': bundle, 'sha256': s.state['bundles'][bundle]['sha256'],
        'status': 'running', 'episode_count': 4, 'completed_episodes': 0, 'success_count': 0,
        'unsuccessful_count': 0, 'error_count': 0, 'success_rate': None, 'active_episodes': active}
    s.state['configuration']['root'] = str(s.config.root)
    s._save()
    s.close()


@pytest.mark.parametrize('stopped', [False, True])
def test_restart_keeps_a_finalized_worker_outcome(tmp_path, stopped):
    s, backend, bundle = rehearsed(tmp_path, episodes=4, workers=2)
    root = formal_worker.worker_root(s.config.root, 2)
    root.mkdir(parents=True)
    row = finalized(s, bundle, 2, **({'reason': 'session_stopped', 'task_complete': None,
                                      'session_stopped': {'stage': 'execution'}} if stopped else {}))
    (root/'state.json').write_text(json.dumps({'results': [row], 'active': None}))
    crashed(s, bundle, [3])
    restarted = Supervisor(s.config, Backend, runner_factory=Runner)
    try:
        batch = restarted.state['formal_batch']
        [recovered] = [r for r in restarted.state['results'] if r.get('mode') == 'formal']
        if stopped:  # The stop cut the episode short: it has no outcome and is not counted.
            assert recovered['formal_outcome'] == 'interrupted' and batch['completed_episodes'] == 0
        else:
            assert recovered['formal_outcome'] == 'success'
            assert (batch['completed_episodes'], batch['success_count']) == (1, 1)
        assert batch['status'] == 'interrupted' and 'active_episodes' not in batch
    finally:
        restarted.close()


def orphan(s, bundle, index, *, final, directory='formal-workers'):
    """A worker of the session still saving its episode when the supervisor restarts."""
    import subprocess
    import sys
    import textwrap
    import time
    root = formal_worker.worker_root(s.config.root, index, directory)
    root.mkdir(parents=True)
    active = {k: v for k, v in finalized(s, bundle, index).items() if k in ('mode', 'bundle', 'sha256', 'layout_id', 'eval_seed')}
    (root/'state.json').write_text(json.dumps({'results': [], 'active': active}))
    code = textwrap.dedent(f'''
        import signal, sys, time
        def stop(*_):
            time.sleep(1.5)
            open({str(root/'state.json')!r}, 'w').write({json.dumps(final)!r})
            sys.exit(0)
        signal.signal(signal.SIGTERM, stop)
        open({str(root/'ready')!r}, 'w').close()
        time.sleep(60)''')
    process = subprocess.Popen([sys.executable, '-c', code, 'services.controller.formal_worker',
                                '--parent-root', str(s.config.root)])
    while not (root/'ready').exists():
        time.sleep(0.05)
    return process


def test_restart_waits_for_an_orphaned_worker_and_keeps_its_record(tmp_path):
    s, backend, bundle = rehearsed(tmp_path, episodes=4, workers=2)
    process = orphan(s, bundle, 2, final={'results': [finalized(s, bundle, 2)], 'active': None})
    s.state['formal_worker_pids'] = {'3': process.pid}
    crashed(s, bundle, [3])
    try:
        restarted = Supervisor(s.config, Backend, runner_factory=Runner)
        try:
            assert process.poll() == 0  # Recovery read the worker's folder only after it exited.
            [recovered] = [r for r in restarted.state['results'] if r.get('mode') == 'formal']
            assert (recovered['formal_outcome'], recovered['reason']) == ('success', 'episode_ended')
            assert 'formal_worker_pids' not in restarted.state
        finally:
            restarted.close()
    finally:
        process.kill()
        process.wait()


def test_a_stopped_batch_with_workers_still_exiting_is_recovered_on_restart(tmp_path):
    s, backend, bundle = rehearsed(tmp_path, episodes=4, workers=2)
    process = orphan(s, bundle, 1, final={'results': [finalized(s, bundle, 1, task_complete=False)], 'active': None})
    s.state['formal_worker_pids'] = {'2': process.pid}
    crashed(s, bundle, [2])
    state = json.loads(s.state_path.read_text())
    state['formal_batch'].update(status='interrupted', finished=123.0)
    s.state_path.write_text(json.dumps(state))
    try:
        restarted = Supervisor(s.config, Backend, runner_factory=Runner)
        try:
            assert process.poll() == 0
            batch = restarted.state['formal_batch']
            assert (batch['status'], batch['finished']) == ('interrupted', 123.0) and 'active_episodes' not in batch
            assert (batch['completed_episodes'], batch['unsuccessful_count']) == (1, 1)
            [recovered] = [r for r in restarted.state['results'] if r.get('mode') == 'formal']
            assert (recovered['formal_episode_index'], recovered['formal_outcome']) == (2, 'unsuccessful')
        finally:
            restarted.close()
    finally:
        process.kill()
        process.wait()


def test_interrupted_parallel_batch_is_final_and_unresumable(tmp_path):
    launched = []
    s, backend, bundle = rehearsed(tmp_path, episodes=6, workers=2)
    good = inline(s, backend, launched)
    def factory(root, bundle_id, index, connection):
        if index == 3:
            raise KeyboardInterrupt()
        return good(root, bundle_id, index, connection)
    s.worker_factory = factory
    try:
        with pytest.raises(KeyboardInterrupt):
            s.run(bundle, formal=True)
        batch = s.state['formal_batch']
        assert batch['status'] == 'interrupted' and batch['success_rate'] is None
        assert batch['completed_episodes'] == 3 and launched == [0, 1, 2]
        s.state['configuration']['root'] = str(s.config.root)
        s._save()
    finally:
        s.close()
    restarted = Supervisor(s.config, Backend, runner_factory=Runner)
    try:
        with pytest.raises(RuntimeError, match='no retry'):
            restarted.run(bundle, formal=True)
    finally:
        restarted.close()


def test_worker_configuration_isolates_one_held_out_layout(tmp_path):
    parent = config(tmp_path, formal_episodes=50, formal_workers=4)
    child = formal_worker.worker_configuration(parent, 7)
    assert (child.formal_seed, child.formal_episodes, child.formal_workers) == (7, 1, 1)
    assert child.formal_collection == parent.formal_collection == 1
    assert child.root == parent.root/'formal-workers/episode-008'
    for count in (0, 17, True):
        with pytest.raises(ValueError, match='formal_workers'):
            config(tmp_path, formal_workers=count)


def test_four_process_workers_use_real_sandboxes_and_one_broker(sandbox_runtime):
    """Only robot feedback and the paid API are fake; process/sandbox plumbing is real."""
    from dataclasses import replace
    import os
    import subprocess
    import sys
    import threading
    from test_controller_backend import configuration
    root = sandbox_runtime
    c = replace(configuration(root), formal_seed=0, formal_eval_seed=1,
                formal_episodes=4, formal_workers=4)
    barrier = threading.Barrier(4, timeout=30)
    parallel = False
    def send(payload):
        if parallel:
            barrier.wait()  # Fails unless all four actual workers reach the broker concurrently.
        return {'model': MODEL, 'choices': [{'message': {'role': 'assistant', 'content': 'OK'},
                'finish_reason': 'stop'}], 'usage': {'total_tokens': 10, 'cost': .002}}
    backend = Backend()
    backend.success = True
    s = Supervisor(c, lambda: backend, GeminiRouter('fixture-only-key', send=send))
    project = source(root)
    (project/'controller.py').write_text('''import json
from pathlib import Path
def main(ctx):
    ctx.call('OBSERVE_TOOL')
    reply = ctx.call('gemini_generate', messages=[{'role': 'user', 'content': 'Hello'}], max_tokens=64)
    value = json.loads(reply['content'][0]['text'])
    assert value['message']['content'] == 'OK'
    assert not Path('/var/run/docker.sock').exists()
    Path('output/broker.json').write_text(json.dumps(value['budget']))
    print('BROKER_OK')
'''.replace('OBSERVE_TOOL', OBSERVE))
    children = []
    shim = '''import sys
sys.path.insert(0, sys.argv.pop(1))
from test_controller_backend import Backend
from services.controller import formal_worker
def backend(config):
    value = Backend()
    value.success = True
    return value
formal_worker.make_backend = backend
formal_worker.main()
'''
    def factory(parent_root, bundle, index, connection):
        with (root/f'worker-{index}.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, '-c', shim, str(Path(__file__).parent),
                '--parent-root', str(parent_root), '--bundle', bundle, '--index', str(index),
                '--channel-fd', str(connection.fileno()), '--parent-pid', str(os.getpid())],
                pass_fds=(connection.fileno(),), stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process
    try:
        bundle, _ = s.register(project)
        rehearsal = s.run(bundle, formal=False)
        assert rehearsal['task_complete'] and rehearsal['returncode'] == 0, rehearsal
        parallel = True
        s.worker_factory = factory
        result = s.run(bundle, formal=True)
        assert len(children) == 4 and all(p.returncode == 0 for p in children)
        assert result['success_count'] == result['completed_episodes'] == 4
        assert result['infrastructure_error_count'] == 0 and not barrier.broken
        assert s.state['usage']['formal']['calls'] == 4
        assert s.state['gemini_spend']['reserved'] == 0
        assert result['api_budget']['session_cost_reported_usd'] == pytest.approx(.010)
        for row in s.state['results']:
            run = s.published_root/row['artifacts']['directory'].removeprefix('runtime/autonomous_controller/')
            assert 'BROKER_OK' in (run/'stdout.log').read_text()
            assert (run/'exports/broker.json').is_file()
            assert row['artifacts']['final_images']
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        s.close()
