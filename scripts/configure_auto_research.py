#!/usr/bin/env python3
"""Trusted operator: create a fresh auto-research configuration; no model/env launch."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from services.controller.config import AGENT_EFFORTS, EXPERT_CONTENTS, Configuration, Limits, FORMAL_EPISODES, MAX_EXPLORATION_EPISODES
from services.benchmarks import BENCHMARKS, DEFAULT, benchmark as benchmark_named
from services.controller.batch import validate_layouts
from services.demonstration_io import DEMONSTRATION_CONTEXTS
from harness.backends import NAMES


def exploration_seeds(args):
    """One layout per exploration episode, cycling through the collection when the
    episodes outnumber its layouts."""
    return benchmark_named(args.benchmark).exploration_seeds(
        args.task, args.exploration_collection, args.exploration_start_seed, args.episodes,
        formal_collection=args.formal_eval_seed, formal_seed=args.formal_seed, formal_episodes=args.formal_episodes)


def given(flag):
    return any(arg == flag or arg.startswith(flag+'=') for arg in sys.argv[1:])


def apply_benchmark(parser, args):
    """The benchmark's defaults for flags the operator did not pass (the same setup defaults
    the dashboard form loads); its fixed fields are an error when passed with another value."""
    bench = benchmark_named(args.benchmark)
    for name, value in {**bench.setup_defaults, **bench.cli_defaults}.items():
        if not given('--'+name.replace('_', '-')):
            setattr(args, name, value)
    for name, (value, reason) in bench.fixed.items():
        flag = '--'+name.replace('_', '-')
        if given(flag):
            current = getattr(args, name)
            if str('' if current is None else current) != str(value):
                parser.error(f'{flag} does not apply to {bench.label} sessions: {reason}')
        else:
            setattr(args, name, None if value == '' else value)


def arguments():
    # No abbreviated flags: apply_benchmark tells passed flags from defaults by their full name.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--task', required=True)
    parser.add_argument('--benchmark', choices=sorted(BENCHMARKS), default=DEFAULT,
                        help='Simulator benchmark')
    parser.add_argument('--task-rules', choices=('public', 'code-summary'), default='public',
                        help='TASK.md: the public task definition, or add rules summarized from the official task code')
    parser.add_argument('--agent-cli', choices=NAMES, default='codex',
                        help='Agent CLI backend')
    parser.add_argument('--agent-provider', choices=('openrouter', 'claude-login', 'anthropic', 'custom', 'deepseek-api', 'deepseek-account', 'moonshot-api', 'kimi-account', 'qoder-account'), default='openrouter',
                        help='Model provider for the selected native agent CLI')
    parser.add_argument('--agent-effort', choices=AGENT_EFFORTS, default='xhigh',
                        help='Model reasoning effort for the agent CLI')
    parser.add_argument('--external-perception', choices=('on', 'off'), default='on',
                        help='Expose the optional external Gemini perception tool')
    parser.add_argument('--sim-gpu', required=True, help='Simulator GPU ordinal or UUID (the agent has no GPU)')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--observation-profile', choices=('rgbd', 'rgb-only'), default='rgbd')
    parser.add_argument('--write-handoff', choices=('off', 'on'), default='off',
                        help='Install the transfer-handoff skill (for source trials of adaptation)')
    parser.add_argument('--adaptation-source', type=Path,
                        help='Finished auto-research trial whose agent continues in this environment')
    parser.add_argument('--adaptation-context', choices=('resume', 'compact', 'clear'), default='compact',
                        help='Adaptation: conversation carried from the source trial')
    parser.add_argument('--agent-formal', choices=('off', 'on'), default='off',
                        help='Evaluation mode: off = code alone (submit one rehearsed bundle); on = live copies of '
                             'the agent play the held-out layouts (no bundle tools in exploration)')
    parser.add_argument('--formal-context', choices=('resume', 'compact', 'clear'), default='compact',
                        help='Agent formal: conversation carried into each formal copy')
    parser.add_argument('--transfer-parts', default='memory,code,handoff',
                        help='Agent formal: comma-separated workspace folders carried into each copy')
    parser.add_argument('--transfer-mb', type=int, default=100, help='Cap on the carried folders (1..1024 MB)')
    parser.add_argument('--subagents', type=int, default=0, help='Concurrent exploration subagents (0..8); 0 = one per environment when envs > 1, else none')
    parser.add_argument('--exploration-envs', type=int, default=1, help='Concurrent exploration slots sharing the episode budget (1..8)')
    parser.add_argument('--demonstration-context', choices=DEMONSTRATION_CONTEXTS, default='terminal_state')
    parser.add_argument('--expert-demonstrations', type=int, default=0,
                        help='Complete expert episodes mounted read-only at runtime/experts (benchmarks that have them: 0 or 3)')
    parser.add_argument('--expert-content', choices=EXPERT_CONTENTS, default='full',
                        help='What the expert episodes hold: full (commanded references, measured motion, '
                             'object states, camera) or images (camera video, previews and depth only)')
    parser.add_argument('--formal-seed', type=int, default=0)
    parser.add_argument('--formal-episodes', type=int, default=FORMAL_EPISODES)
    parser.add_argument('--formal-workers', type=int, default=4, help='Concurrent formal episodes')
    parser.add_argument('--formal-eval-seed', type=int, default=1, help='Held-out layout collection')
    parser.add_argument('--exploration-collection', type=int, default=0,
                        help='Layout collection for exploration')
    parser.add_argument('--exploration-start-seed', type=int, default=1)
    parser.add_argument('--episode-reviews', choices=('off', 'on', 'detailed'), default='off',
                        help='Benchmarks with simulator ground truth: a review of each closed exploration '
                             'episode and rehearsal, written beside the episode for the agent; detailed adds the '
                             'ground-truth report and exact numbers. '
                             'Operator-side; never in the agent configuration')
    parser.add_argument('--recording-gib', type=int, default=10,
                        help="Size limit of a robot script run's output folder and of a submitted bundle "
                             '(recorded episode data is not capped)')
    return parser


def main():
    parser = arguments()
    args = parser.parse_args()
    if not 1 <= args.episodes <= MAX_EXPLORATION_EPISODES:
        parser.error(f'--episodes must be 1..{MAX_EXPLORATION_EPISODES}')
    if args.recording_gib < 1:
        parser.error('--recording-gib must be at least 1')
    apply_benchmark(parser, args)
    episode_seconds = benchmark_named(args.benchmark).task_timeouts(args.task)['episode_seconds']
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {}
    for line in devices.splitlines():
        index, identifier = map(str.strip, line.split(','))
        mapping[index] = mapping[identifier] = identifier
    parent = (PROJECT/'runtime').resolve()/'auto-research'
    root = parent/(time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'_'+uuid.uuid4().hex[:10])
    data_bytes = args.recording_gib*1024**3
    def limits(seconds, memory=8192):
        return Limits(seconds, memory, 4, 256, 1024, data_bytes)
    config = Configuration(root, args.task,
        exploration_seeds(args),
        args.formal_seed, args.exploration_collection, mapping[args.sim_gpu],
        limits(episode_seconds), limits(episode_seconds), limits(episode_seconds, 16384),
        demonstration_context=args.demonstration_context,
        expert_demonstrations=args.expert_demonstrations,
        expert_content=args.expert_content,
        formal_episodes=args.formal_episodes, formal_eval_seed=args.formal_eval_seed,
        formal_workers=args.formal_workers, observation_profile=args.observation_profile, exploration_envs=args.exploration_envs, subagents=args.subagents,
        agent_formal=args.agent_formal == 'on', formal_context=args.formal_context,
        write_handoff=args.write_handoff == 'on', adaptation_context=args.adaptation_context,
        adaptation_source=str(args.adaptation_source.resolve()) if args.adaptation_source else None,
        transfer_parts=tuple(p for p in args.transfer_parts.split(',') if p), transfer_mb=args.transfer_mb,
        task_rules=args.task_rules, agent_cli=args.agent_cli,
        agent_provider=args.agent_provider, agent_effort=args.agent_effort,
        external_perception=args.external_perception == 'on',
        benchmark=args.benchmark)
    validate_layouts(config)
    # Fetch only official visual media before any model/episode starts; the agent works offline. Deployment converts the selected format to local images.
    benchmark_named(args.benchmark).fetch_demonstration(PROJECT/'runtime/reference-demos', args.task,
                                                        args.demonstration_context)
    benchmark_named(args.benchmark).reference_mounts(config)  # The expert episodes are installed.
    root.mkdir(parents=True, mode=0o700)
    path = root/'operator-input.json'
    with path.open('x') as stream:
        json.dump(asdict(config), stream, default=str, indent=2)
    path.chmod(0o600)
    from services.episode_reviews.runner import set_settings
    set_settings(root, enabled=args.episode_reviews != 'off', detailed=args.episode_reviews == 'detailed')
    print(path)
    print(f'bash scripts/start_auto_research_agent.sh --config {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
