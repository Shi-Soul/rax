"""Formal batches reserve one immutable schedule; no retry or survivor-only average."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from services.controller.batch import import_rehearsed_bundle
from services.controller.frontend import ResearchFrontend
from services.controller.supervisor import Supervisor
from test_controller_backend import Backend, Runner, configuration, source


def config(tmp_path, **kwargs):
    return replace(configuration(tmp_path), **{'formal_seed': 0, 'formal_eval_seed': 1, 'formal_episodes': 50, **kwargs})


def test_fifty_formals_distinct_seeds_and_fixed_denominator(tmp_path):
    c = config(tmp_path)
    backend = Backend()
    s = Supervisor(c, lambda: backend, runner_factory=Runner)
    try:
        bundle, _ = s.register(source(tmp_path, 'controller.py'))
        with pytest.raises(RuntimeError, match='clean rehearsal'):
            s.run(bundle, formal=True)
        backend.success = True
        s.run(bundle, formal=False)
        def run(_, job):
            index = len(backend.starts)-2
            assert s.state['formal_reserved'] and job['mode'] == 'formal'
            backend.success = index % 5 != 1
            return {'reason': 'exit', 'returncode': 1 if index % 5 == 0 else 0}
        Runner.hook = run
        result = s.run(bundle, formal=True)
        assert backend.starts[1:] == [(True, i) for i in range(50)]
        assert result['completed_episodes'] == result['episode_count'] == 50
        assert result['success_count'] == 30 and result['error_count'] == 10
        assert result['unsuccessful_count'] == 10 and result['success_rate'] == .6
        report = json.loads((c.root/'published/formal_batch.json').read_text())
        assert len(report['episodes']) == 50 and report['success_rate'] == .6
        assert report['episodes'][0]['formal_outcome'] == 'error'
        assert all('layout_id' not in r and 'eval_seed' not in r for r in report['episodes'])
        assert result['api_budget']['session_cost_limit_usd'] is None
        assert [r['formal_episode_index'] for r in s.state['results'][1:]] == list(range(1,51))
        assert {r['eval_seed'] for r in s.state['results'][1:]} == {1}
        assert ResearchFrontend.public_result(result)['success_rate'] == .6
        with pytest.raises(RuntimeError, match='no retry'):
            s.run(bundle, formal=True)
        with pytest.raises(RuntimeError):
            s.start_interactive()
    finally:
        Runner.hook = None
        s.close()


def test_interrupted_batch_cannot_resume_or_report_final_rate(tmp_path):
    c = config(tmp_path)
    backend = Backend()
    s = Supervisor(c, lambda: backend, runner_factory=Runner)
    try:
        bundle, _ = s.register(source(tmp_path, 'controller.py'))
        backend.success = True
        s.run(bundle, formal=False)
        def interrupt(_, job):
            if len(backend.starts) == 3:
                raise KeyboardInterrupt()
            return {'reason': 'exit', 'returncode': 0}
        Runner.hook = interrupt
        with pytest.raises(KeyboardInterrupt):
            s.run(bundle, formal=True)
        assert s.state['formal_batch']['status'] == 'interrupted'
        assert s.state['formal_batch']['success_rate'] is None
        assert s.state['formal_batch']['completed_episodes'] == 1
    finally:
        Runner.hook = None
        s.close()
    restarted = Supervisor(c, lambda: Backend(), runner_factory=Runner)
    try:
        with pytest.raises(RuntimeError, match='no retry'):
            restarted.run(bundle, formal=True)
    finally:
        restarted.close()


def test_formal_episodes_are_bounded(tmp_path):
    value = configuration(tmp_path)
    assert replace(value, formal_seed=0, formal_episodes=50).formal_collection == 1
    for count in (0, 151, True):  # 150: the largest formal set
        with pytest.raises(ValueError, match='formal_episodes'):
            replace(value, formal_episodes=count)


def test_import_requires_real_rehearsal_and_identical_frozen_bundle(tmp_path):
    source_root = tmp_path/'original'
    original = Supervisor(configuration(source_root), Backend, runner_factory=Runner)
    target = Supervisor(config(tmp_path/'batch'), Backend, runner_factory=Runner)
    try:
        bundle, manifest = original.register(source(tmp_path, 'controller.py'))
        with pytest.raises(ValueError, match='no qualifying'):
            import_rehearsed_bundle(target, original.config.root, bundle)
        backend = Backend(); backend.success = True
        original.backend_factory = lambda: backend
        original.run(bundle, formal=False)
        imported = import_rehearsed_bundle(target, original.config.root, bundle)
        copied = target.state['bundles'][imported]
        assert copied['sha256'] == manifest['sha256']
        assert target._has_qualifying_rehearsal(imported, copied)
        assert not target.state['results']  # Imported evidence is not a new episode.
        (original.config.root/'bundles'/bundle/'controller.py').chmod(0o644)
        (original.config.root/'bundles'/bundle/'controller.py').write_text('tampered')
        with pytest.raises(ValueError, match='changed'):
            import_rehearsed_bundle(target, original.config.root, bundle)
    finally:
        original.close()
        target.close()


def test_bundle_import_reads_parent_state_through_replace_gap(tmp_path, monkeypatch):
    import errno

    source_root = tmp_path/'original'
    original = Supervisor(configuration(source_root), Backend, runner_factory=Runner)
    source_root = original.config.root
    target = Supervisor(config(tmp_path/'batch'), Backend, runner_factory=Runner)
    try:
        bundle, manifest = original.register(source(tmp_path, 'controller.py'))
        backend = Backend()
        backend.success = True
        original.backend_factory = lambda: backend
        original.run(bundle, formal=False)
        original_read = Path.read_text
        reads = []

        def interrupted_read(path, *args, **kwargs):
            if path == source_root/'state.json':
                reads.append(path)
                if len(reads) == 1:
                    raise FileNotFoundError(errno.ENOENT, 'Replacement in progress', str(path))
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', interrupted_read)
        imported = import_rehearsed_bundle(target, source_root, bundle)
        assert len(reads) == 2
        assert target.state['bundles'][imported]['sha256'] == manifest['sha256']
    finally:
        original.close()
        target.close()


def test_import_keeps_the_rehearsal_evidence_whole(tmp_path):
    """A rehearsal that reached the native end and then outlived the finalization window
    qualifies; its import must qualify too and must not claim task success."""
    source_root = tmp_path/'original'
    original = Supervisor(configuration(source_root), Backend, runner_factory=Runner)
    target = Supervisor(config(tmp_path/'batch'), Backend, runner_factory=Runner)
    try:
        bundle, manifest = original.register(source(tmp_path, 'controller.py'))
        backend = Backend(); backend.success = False
        original.backend_factory = lambda: backend
        Runner.hook = lambda *_: {'reason': 'finalization_timeout', 'returncode': None}
        original.run(bundle, formal=False)
        Runner.hook = None
        assert original._has_qualifying_rehearsal(bundle, manifest)
        imported = import_rehearsed_bundle(target, original.config.root, bundle)
        record = target.state['imported_rehearsals'][0]
        assert (record['task_complete'], record['returncode'], record['reason'], record['finalization_timeout']) \
            == (False, None, 'episode_ended', True)
        assert target._has_qualifying_rehearsal(imported, target.state['bundles'][imported])
    finally:
        Runner.hook = None
        original.close()
        target.close()
