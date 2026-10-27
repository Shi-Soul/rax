"""Operator-only post-hoc formal batch of an already rehearsed, frozen bundle.

Creates a new archive; never reopens or overwrites the source formal attempt.
No agent process, code edits, API credentials, or episode retries are involved.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import time

from .backend import make_backend
from .config import Configuration
from .errors import describe
from .storage import persist, verify
from .supervisor import Supervisor, qualifying_rehearsal

REHEARSAL_EVIDENCE = ('task_complete', 'returncode', 'reason', 'finalization_timeout', 'simulator_exited',
                      'evaluation_error', 'evaluation_failure', 'error_type', 'error', 'output_error',
                      'recording_errors', 'cleanup_error')


def validate_layouts(config):
    from services.benchmarks import for_config
    for_config(config).validate_layouts(config)


def import_rehearsed_bundle(supervisor, source, bundle_id):
    """Import evidence only after validating the source manifest and runtime."""
    from .storage import read_state
    state = read_state(source/'state.json')
    manifest = state['bundles'][bundle_id]
    source_bundle = (source/'bundles'/manifest['directory']).resolve()
    if not source_bundle.is_relative_to((source/'bundles').resolve()):
        raise ValueError('Invalid source bundle directory')
    verify(source_bundle, manifest)
    # Same evidence rule as Supervisor._has_qualifying_rehearsal, so a batch
    # session's own imported evidence can seed its parallel formal workers.
    evidence = next((r for r in state['results'] + state.get('imported_rehearsals', [])
                     if qualifying_rehearsal(r, bundle_id, manifest)), None)
    if evidence is None:
        raise ValueError('Source bundle has no qualifying rehearsal')
    original = Configuration.from_dict(state['configuration'])
    if original.task != supervisor.config.task:
        raise ValueError('Source task must match the batch')
    imported, copied = supervisor.register(source_bundle, workspace_path=manifest['workspace_path'])
    if copied['sha256'] != manifest['sha256']:
        raise ValueError('Imported bundle fingerprint differs from rehearsal')
    # The evidence fields qualifying_rehearsal reads are copied whole, so the import qualifies
    # exactly when the source rehearsal did (a finalization_timeout end included).
    supervisor.state['imported_rehearsals'] = [{
        'mode': 'rehearsal', 'bundle': imported, 'sha256': copied['sha256'],
        **{key: evidence.get(key) for key in REHEARSAL_EVIDENCE},
        'source_session': str(source), 'source_bundle': bundle_id,
        'source_episode_id': evidence.get('episode_id')}]
    supervisor._save()
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-session', type=Path, required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New SSD-backed auto-research session directory')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--eval-seed', type=int, default=1)
    parser.add_argument('--first-layout', type=int, default=0)
    parser.add_argument('--formal-wall-seconds', type=int,
                        help='Override the per-episode wall limit in this new evaluation only')
    args = parser.parse_args()
    source, output = args.source_session.resolve(), args.output.resolve()
    original = Configuration.from_dict(json.loads((source/'configuration.json').read_text()))
    config = replace(original, root=output, formal_episodes=args.episodes,
                     formal_eval_seed=args.eval_seed, formal_seed=args.first_layout,
                     demonstration_context='none', formal_workers=1)
    if args.formal_wall_seconds is not None:
        config = replace(config, formal=replace(config.formal, wall_seconds=args.formal_wall_seconds))
    validate_layouts(config)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    persist(output/'operator-input.json', json.loads(json.dumps(asdict(config), default=str)))
    lifecycle = {'kind': 'formal_batch', 'pid': os.getpid(), 'status': 'running',
                 'started_at': time.time(), 'source_session': str(source), 'source_bundle': args.bundle}
    persist(output/'agent-lifecycle.json', lifecycle)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    supervisor = None
    try:
        supervisor = Supervisor(config, lambda: make_backend(config))
        bundle = import_rehearsed_bundle(supervisor, source, args.bundle)
        print(json.dumps({'status': 'starting', 'output': str(output), 'bundle': bundle,
                          'episodes': config.formal_episodes, 'eval_seed': config.formal_collection,
                          'first_layout': config.formal_seed}), flush=True)
        result = supervisor.run(bundle, formal=True)
        persist(output/'formal-report.json', result)
        lifecycle.update(status='exited', returncode=0)
        print(json.dumps(result), flush=True)
    except KeyboardInterrupt:
        lifecycle['status'] = 'stopped'
        raise SystemExit(130)
    except Exception as exc:
        lifecycle.update(status='failed', error_type=type(exc).__name__, error=describe(exc))
        raise
    finally:
        try:
            if supervisor:
                supervisor.close()
        finally:
            lifecycle['finished_at'] = time.time()
            persist(output/'agent-lifecycle.json', lifecycle)


if __name__ == '__main__':
    main()
