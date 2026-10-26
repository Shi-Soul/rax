#!/usr/bin/env python3
"""Submit a validated native CLI trial, using the DeepSeek comparison settings."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from harness.backends import PROVIDERS, get
from harness.isolated_cli.credentials import Credential
from harness.model_serving.lifecycle import write_json
from harness.model_serving.slurm import script
from scripts.run_deepseek_experiment import allocated_gpu

SETTINGS = {'task': 'click_mouse', 'exploration_budget': 16, 'formal_episodes': 150,
    'formal_workers': 4, 'evaluation_mode': 'code_formal', 'exploration_envs': 1, 'subagents': 0,
    'observation_profile': 'rgbd', 'demonstration_context': 'terminal_state',
    'expert_demonstrations': 0, 'episode_reviews': False, 'external_perception': False, 'effort': 'max'}


def validate_report(report, backend, model):
    required = ('vision_wire', 'wire_max_effort', 'resume', 'native_file', 'native_shell', 'tool_call', 'tool_result')
    if not (report.get('status') == 'passed' and report.get('backend') == backend
            and report.get('provider') == 'openrouter' and report.get('model') == model
            and report.get('effort') == 'max' and all(report.get(k) is True for k in required)
            and report.get('boundary') and all(report['boundary'].values())):
        raise ValueError('Pass matching native CLI/model OpenRouter validation before submitting a GPU trial')


def submit(args):
    cli = get(args.backend)
    spec = importlib.import_module(cli.__name__.rsplit('.', 1)[0]+'.spec')
    model = args.model or spec.MODEL
    credential = Credential(spec, 'openrouter', key_file=args.key_file, auth_home=args.auth_home)
    if args.backend == 'qoder':
        cli.token_file(args.auth_home)
    validate_report(json.loads(args.validation.read_text()), args.backend, model)
    binary = cli.binary(args.binary)
    profile = json.loads(args.profile.read_text())
    resources = {'account': profile['account'], 'storage_group': profile.get('storage_group'), **profile['experiment']}
    if any(word in resources['partition'].lower() for word in ('requeue', 'preempt')):
        raise ValueError('Use a regular GPU partition')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = root/'trial.json'
    record = {'status': 'rendered', 'project': str(PROJECT),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=PROJECT, text=True).strip(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        'created_at': time.time(), 'agent_cli': args.backend, **credential.record(), 'model': model, **SETTINGS,
        'validation': str(args.validation.resolve()), 'binary': str(binary), 'resources': resources,
        'experiment_record': str(root/'experiment.json')}
    node = Path(shutil.which('node')).parent
    setup = ['cd '+shlex.quote(str(PROJECT)), 'export PYTHONDONTWRITEBYTECODE=1',
        'export PATH='+shlex.quote(str(node))+':"$PATH"', 'export TMPDIR=/tmp',
        'export RAX_RUNTIME_ROOT='+shlex.quote(str((PROJECT/'runtime').resolve())),
        'export RAX_ARTIFACT_ROOT="$RAX_RUNTIME_ROOT"', 'export DEXJOCO_RUNTIME_ROOT="$RAX_RUNTIME_ROOT"',
        'export OMP_NUM_THREADS=4', 'export OPENBLAS_NUM_THREADS=4',
        'export XDG_RUNTIME_DIR="/tmp/rax-native-${SLURM_JOB_ID}"',
        'mkdir -p "$XDG_RUNTIME_DIR"', 'chmod 700 "$XDG_RUNTIME_DIR"']
    batch = root/'experiment.sbatch'
    batch.write_text(script([PROJECT/'runtime/envs/host/bin/python', Path(__file__).resolve(),
        'run', '--manifest', manifest], name=f'rax-{args.backend}-click-mouse',
        output=root/'slurm-%j.log', setup=setup, **resources))
    subprocess.run(['bash', '-n', str(batch)], check=True)
    write_json(manifest, record)
    if args.submit:
        record.update(status='submitted', submitted_at=time.time(),
            job_id=subprocess.check_output(['sbatch', '--parsable', str(batch)], text=True).strip().split(';')[0])
        write_json(manifest, record)
    print(json.dumps({'manifest': str(manifest), **record}, indent=2))


def configure_command(record, gpu):
    return list(map(str, [sys.executable, PROJECT/'scripts/configure_auto_research.py',
        '--benchmark', 'dexjoco', '--task', record['task'], '--sim-gpu', gpu,
        '--episodes', record['exploration_budget'], '--formal-episodes', record['formal_episodes'],
        '--formal-workers', record['formal_workers'], '--agent-cli', record['agent_cli'],
        '--agent-provider', 'openrouter', '--agent-effort', 'max', '--exploration-envs', 1, '--subagents', 0,
        '--external-perception', 'off', '--observation-profile', 'rgbd',
        '--demonstration-context', 'terminal_state', '--expert-demonstrations', 0, '--episode-reviews', 'off']))


def run(args):
    record = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    outcome = Path(record['experiment_record'])
    if outcome.exists():
        raise FileExistsError('Trial already started; resume its existing session')
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    if current != record['commit']:
        raise ValueError('The trial checkout changed after submission; submit from a stable worktree')
    gpu = allocated_gpu()
    write_json(outcome, {'status': 'doctor', 'started_at': time.time(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    with (root/'doctor.log').open('w') as log:
        subprocess.run([sys.executable, PROJECT/'scripts/dexjoco_doctor.py', '--gpu', gpu,
            '--task', record['task'], '--output', root/'doctor'], cwd=PROJECT,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    output = subprocess.check_output(configure_command(record, gpu), cwd=PROJECT, text=True)
    config = Path(output.strip().splitlines()[-1])
    if not config.is_file():
        raise RuntimeError('Configuration was not created')
    result = {'status': 'running', 'config': str(config), 'session': str(config.parent),
        'agent_cli': record['agent_cli'], 'model': record['model'], 'provider': 'openrouter',
        **SETTINGS, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'created_at': time.time()}
    write_json(outcome, result)
    command = [sys.executable, '-m', 'harness.codex_cli.auto_research', '--config', str(config),
        '--non-interactive', '--agent-model', record['model'], '--agent-binary', record['binary'],
        '--key-file', record['key_file'], '--agent-key-file', record['key_file']]
    if record.get('auth_home'):
        command += ['--agent-auth-home', record['auth_home']]
    with (config.parent/'agent.log').open('a') as log:
        child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, lambda number, _: child.send_signal(number))
        code = child.wait()
    result.update(status='exited', returncode=code, finished_at=time.time())
    write_json(outcome, result)
    raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    launch = actions.add_parser('submit')
    launch.add_argument('--backend', choices=tuple(PROVIDERS), required=True)
    for name in ('root', 'profile', 'validation', 'key-file'):
        launch.add_argument('--'+name, type=Path, required=True)
    launch.add_argument('--binary', type=Path)
    launch.add_argument('--auth-home', type=Path, help='Qoder experiment directory containing qoder-pat')
    launch.add_argument('--model')
    launch.add_argument('--submit', action='store_true')
    worker = actions.add_parser('run')
    worker.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    submit(args) if args.action == 'submit' else run(args)


if __name__ == '__main__':
    main()
