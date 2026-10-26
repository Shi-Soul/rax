"""Run the auto-research agent natively on the host: Codex (or Claude Code, harness/claude_cli)
as shipped, with its own login, in its own sandbox; the session's frontend serves the robot
tools (harness/agent_host.py)."""
from contextlib import contextmanager
from dataclasses import asdict
import argparse
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from urllib.parse import urlsplit

from harness import agent_host
from harness.codex_cli.configuration import subagent_settings, write_config
from services.controller.config import Configuration
from services.controller.demonstrations import provision_demonstration, task_demonstration_context
from services.controller.gemini import load_key
from services.controller.storage import persist
from services.benchmarks import for_config

PROJECT = agent_host.PROJECT
# Sessions have no wall-clock or model-call budget; only each episode's own limits end it.
# One robot tool call (an episode start, a rehearsal) may block this long, below the
# 2**31-1 ms (24.8-day) limit of the Claude CLI's Node timers.
CALL_SECONDS = 20*24*3600
MODEL = 'openai/gpt-6-astra'  # When the provider profile names none.
# A key-based provider profile hands Codex its key in this variable. As with native Codex, the
# agent's commands see it too (they have no internet).
KEY_VARIABLE = 'RAX_MODEL_KEY'
LOGIN_FILE = 'agent-login.json'  # The session's login (a path or a key file's path, never a secret).
PID_FILE = 'agent.pid'  # The running agent CLI, frozen by the frontend during register/rehearse.
PERMISSIONS = 'agent'


def populate_workspace(config, workspace, contract=None, *, formal_episode=False):
    """Template, skills, TASK.md and demonstrations for a new agent workspace.

    formal_episode: the workspace of an agent formal copy (services/agent_formal), whose
    instructions describe its one live episode instead of exploration. With agent formal
    on (live-copy mode) no text frames the work as an evaluation.
    """
    for name in ('code', 'memory', 'output', 'runtime'):
        (workspace/name).mkdir(parents=True, exist_ok=True)
    from services.mcp_contract import Contract
    from services.workspace import Setup, build, recipe_for
    if formal_episode:
        options = frozenset({'formal-episode'})
    else:
        options = frozenset(name for name, enabled in (('handoff', config.write_handoff),
                                                       ('agent-formal', config.agent_formal)) if enabled)
    # Demonstrations first: the instructions name the kind and number of images supplied.
    demonstration = provision_demonstration(config, workspace, PROJECT/'runtime/reference-demos',
                                            record=not formal_episode)
    benchmark = for_config(config)
    built = build(recipe_for(options),
                  Setup(contract or Contract.from_config(config), options, config.task, config.agent_cli,
                        demonstrations=config.demonstration_context != 'none',
                        demonstration_kind=demonstration['kind'],
                        demonstration_images=demonstration.get('image_count', 0),
                        carries_handoff='handoff' in config.transfer_parts,
                        expert_demonstrations=config.expert_demonstrations,
                        expert_content=config.expert_content), workspace)
    rubric = benchmark.task_context(config, demonstration, live_formal=formal_episode)
    text = rubric + task_demonstration_context(demonstration, live_formal=formal_episode, bundles=not config.agent_formal)
    (workspace/'TASK.md').write_text(text)
    if not formal_episode:
        refresh_reviews(config, workspace)
    return built


def refresh_reviews(config, workspace):
    """An exploring agent's instructions follow its benchmark's current settings, such as
    its episode reviews (at deploy and on every resume)."""
    for_config(config).refresh_workspace(config, workspace)


# The run tree's subdirectories the frontend writes during a session; the agent reads them.
PUBLISHED_PARTS = ('runs', 'episode_reviews')


def expose_published(root):
    """The session's run tree (frames, traces, reviews), seen by the agent at
    runtime/autonomous_controller. It stays outside the agent's writable workspace."""
    published = root/'published'
    published.mkdir(mode=0o700, exist_ok=True)
    for part in PUBLISHED_PARTS:
        (published/part).mkdir(mode=0o700, exist_ok=True)
    return published


def reference_sources(config):
    """The benchmark's read-only reference data for the agent (such as expert episodes): each
    source on the host and where the workspace shows it."""
    return [(Path(source).resolve(), PurePosixPath(destination).relative_to('/workspace'))
            for source, destination in for_config(config).reference_mounts(config)]


def expose_references(config, workspace):
    """Link the reference data into a workspace where its instructions name it (runtime/experts).
    The links' targets stay outside the workspace; the agent's sandbox reads them and never
    writes them (readable() in each launcher)."""
    for source, relative in reference_sources(config):
        link = workspace/relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():  # A resumed session keeps its links.
            link.symlink_to(source, target_is_directory=True)


def prepare_workspace(config, home_name, *, published_symlink=True):
    """The agent workspace (template, skills, TASK.md), its CLI home and the private
    configuration; shared by the Codex and Claude launchers."""
    root = config.root
    workspace, home = root/'agent-workspace', root/home_name
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    published = expose_published(root)
    persist(root/'workspace-manifest.json', populate_workspace(config, workspace))
    artifacts = workspace/'runtime/autonomous_controller'
    if published_symlink:
        artifacts.symlink_to(published, target_is_directory=True)
    else:
        artifacts.mkdir()  # The launcher mounts published/ here read-only.
    expose_references(config, workspace)
    private_config = root/'configuration.json'
    with private_config.open('x') as stream:
        json.dump(asdict(config), stream, default=str)
    private_config.chmod(0o600)
    return {'workspace': workspace, 'home': home, 'configuration': private_config}


def chatgpt_login(path):
    """A Codex ChatGPT login file (such as a profile's auth.json); its tokens stay in it."""
    auth = json.loads(Path(path).read_text())
    tokens = auth.get('tokens') or {}
    if auth.get('auth_mode') != 'chatgpt' or not tokens.get('access_token') or not tokens.get('refresh_token'):
        raise ValueError(f'{path} is not a Codex ChatGPT login')
    return auth


DEFAULT_OPENROUTER_KEY = Path('.codex/secrets/openrouter_api_key')  # Under the operator's home.
# The default 'openrouter' profile when no provider file defines one: OpenRouter's Responses API.
OPENROUTER_PROVIDER = {'name': 'OpenRouter', 'base_url': 'https://openrouter.ai/api/v1', 'wire_api': 'responses'}


def openrouter_key_file(path=None):
    """The OpenRouter key file (Gemini, and the model on the OpenRouter profile): the operator's
    --key-file, else ~/.codex/secrets/openrouter_api_key when it exists. None leaves the
    OPENROUTER_API_KEY environment variable as Gemini's only source (load_key(None))."""
    if path is not None:
        return Path(path)
    default = Path.home()/DEFAULT_OPENROUTER_KEY
    return default if default.is_file() else None


def openrouter_endpoint(name, provider):
    """Whether a Codex provider profile is OpenRouter, the only provider the OpenRouter key is sent to."""
    return name == 'openrouter' or urlsplit(str(provider.get('base_url', ''))).hostname == 'openrouter.ai'


def provider_login(provider_home, profile, key_file=None):
    """The session's login from an operator provider profile (<provider-home>/<profile>.config.toml).

    A ChatGPT profile ([rax] chatgpt_auth) shares that login file with every session
    using it: Codex writes refreshed tokens back through the session's link and re-reads the
    file before refreshing, so sessions never invalidate each other. Any other profile names
    its Responses provider and its key file ([rax] key_file); key_file (the operator's
    OpenRouter key) serves only an OpenRouter provider.
    """
    if not profile or Path(profile).name != profile:
        raise ValueError('Invalid provider profile')
    data, harness_settings = {}, {}
    for path in (provider_home/'config.toml', provider_home/f'{profile}.config.toml'):
        if path.exists():
            values = tomllib.loads(path.read_text())
            harness_settings.update(values.pop('rax', {}))  # Harness-only profile settings.
            for key, value in values.items():
                if key == 'model_providers':
                    data.setdefault(key, {}).update(value)
                else:
                    data[key] = value
    model = harness_settings.get('model') or data.get('model') or MODEL
    login = {'profile': profile}
    if harness_settings.get('chatgpt_auth'):
        auth = (provider_home/harness_settings['chatgpt_auth']).resolve()
        chatgpt_login(auth)
        login['chatgpt_auth'] = str(auth)
        model = model.removeprefix('openai/')  # ChatGPT names models without a vendor prefix.
    else:
        name = data.get('model_provider')
        provider = dict((data.get('model_providers') or {}).get(name) or {})
        if not provider and profile == 'openrouter':
            name, provider = 'openrouter', dict(OPENROUTER_PROVIDER)
        if not provider:
            raise ValueError(f'Provider profile {profile!r} needs model_provider and its [model_providers.<name>] '
                             f'table in {provider_home/"config.toml"} or {provider_home/(profile+".config.toml")}')
        if provider.get('wire_api', 'responses') != 'responses':
            raise ValueError(f'Provider profile {profile!r} names no Responses provider')
        if harness_settings.get('key_file'):
            key_file = provider_home/harness_settings['key_file']
        elif not openrouter_endpoint(name, provider):
            raise ValueError(f'Provider profile {profile!r} needs its own key file ([rax] key_file); '
                             'the OpenRouter key is sent only to OpenRouter')
        elif not key_file:
            raise ValueError('The OpenRouter profile needs a key file: --key-file or '
                             '~/.codex/secrets/openrouter_api_key')
        load_key(key_file)  # Fail before provisioning.
        for field in ('auth', 'env_key', 'env_key_instructions', 'experimental_bearer_token'):
            provider.pop(field, None)
        login.update(provider=provider, key_file=str(Path(key_file).resolve()))
        if harness_settings.get('native_models'):
            model = model.removeprefix('openai/')  # OpenAI's own API names models without one.
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}', model):
        raise ValueError(f'Invalid model in provider profile {profile!r}')
    login['model'] = model
    for key in ('service_tier',):  # Codex settings a profile may carry for its sessions.
        if key in data or key in harness_settings:
            login[key] = harness_settings.get(key, data.get(key))
    return login


def deploy(config, *, provider_home, profile='openrouter', key_file=None, codex=None):
    """Prepare a new session workspace without starting a model or simulator."""
    codex = codex_binary(codex)
    root = config.root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root/'state.json').exists() or (root/'agent-workspace').exists():
        raise FileExistsError('Use a fresh auto-research trial root')
    login = provider_login(provider_home, profile, key_file)
    prepared = prepare_workspace(config, 'codex-home')
    persist(root/LOGIN_FILE, login)
    deployed = {'workspace': prepared['workspace'], 'codex_home': prepared['home'], 'codex': codex,
                'configuration': prepared['configuration'], 'login': root/LOGIN_FILE, 'resume_session': None}
    write_home(config, deployed)
    if config.adaptation_source:
        from services.adaptation import import_source
        deployed['resume_session'] = import_source(config, deployed['workspace'], deployed['codex_home'])
    return deployed


def codex_binary(path=None):
    binary = Path(path or shutil.which('codex') or '').resolve()
    if not binary.is_file():
        raise RuntimeError('Codex binary is required')
    return binary


def readable(config, deployed):
    """Paths outside the workspace the agent reads: its run tree, or (agent formal) its
    episode's frames, and the benchmark's reference data. Codex re-exposes only writable paths
    inside hidden ones, so the run tree's top is writable and the parts the frontend writes are
    read-only within it; the reference data is carved out of the hidden paths instead."""
    references = [source for source, _ in reference_sources(config)]
    if 'frames' in deployed:
        return {'write': [deployed['frames']], 'read': [], 'references': references}
    published = config.root/'published'
    return {'write': [published], 'read': [published/part for part in PUBLISHED_PARTS], 'references': references}


def protected(workspace):
    """Workspace directories that configure Codex itself, kept read-only for the agent."""
    (workspace/'.codex').mkdir(exist_ok=True)
    return [workspace/'.codex']


def permissions(config, deployed, socket):
    """Codex's permission profile for the agent's commands: write only the workspace (and the
    robot socket's directory), hide the repository, the sessions and host sockets, and no
    internet: the managed proxy allows no domain, while local Unix sockets stay reachable."""
    workspace = Path(deployed['workspace']).resolve()
    exposed = readable(config, deployed)
    filesystem = {str(path): 'deny' for path in agent_host.hidden_paths(config.root, carved=True,
                                                                        kept=exposed['references'])}
    for path in (workspace, Path(socket).parent, *exposed['write']):
        filesystem[str(Path(path).resolve())] = 'write'
    # Project configuration there would configure Codex itself, the sandbox included.
    for path in (*exposed['read'], *exposed['references'], *protected(workspace)):
        filesystem[str(Path(path).resolve())] = 'read'
    return {'extends': ':workspace', 'filesystem': filesystem,
            'network': {'enabled': True, 'dangerously_allow_all_unix_sockets': True}}


def settings(config, deployed, socket):
    """The session's Codex configuration: Codex's own defaults, plus only the login's model,
    the session's effort and subagent cap, the robot tools and the sandbox."""
    login = json.loads(deployed['login'].read_text())
    value = {'model': login['model'], 'model_reasoning_effort': config.agent_effort}
    if login.get('provider'):
        value.update(model_provider='session', model_providers={'session': {
            **login['provider'], 'env_key': KEY_VARIABLE}})
    if login.get('service_tier'):
        value['service_tier'] = login['service_tier']
    workspace = str(Path(deployed['workspace']).resolve())
    value.update({
        # Unattended: nobody answers approvals; the sandbox is the boundary.
        'approval_policy': 'never', 'default_permissions': PERMISSIONS,
        'permissions': {PERMISSIONS: permissions(config, deployed, socket)},
        # Offline like the agent's commands: web search is internet access through the model.
        'web_search': 'disabled',
        'features': {'network_proxy': True},
        'projects': {workspace: {'trust_level': 'trusted'}},
        'mcp_servers': {'auto_research': {
            'command': str(agent_host.agent_python()), 'args': ['-I', str(agent_host.BRIDGE), str(socket)],
            'required': True, 'tool_timeout_sec': CALL_SECONDS+300, 'default_tools_approval_mode': 'approve'}}})
    from services.mcp_contract import Contract
    cap = deployed['subagents'] if 'subagents' in deployed else Contract.from_config(config).subagent_cap
    if cap is not None:
        agents = subagent_settings(cap)
        value['features'].update(agents['features'])
        if 'agents' in agents:
            value['agents'] = agents['agents']
    return value


def write_home(config, deployed, socket=None):
    """Write the session's Codex home for this launch: config.toml and the login link."""
    home = Path(deployed['codex_home'])
    socket = socket or agent_host.socket_path(config.root)
    write_config(home/'config.toml', settings(config, deployed, socket))
    login = json.loads(deployed['login'].read_text())
    auth = home/'auth.json'
    if login.get('chatgpt_auth'):
        if not (auth.is_symlink() and os.readlink(auth) == login['chatgpt_auth']):
            auth.unlink(missing_ok=True)
            auth.symlink_to(login['chatgpt_auth'])
    elif auth.is_symlink():
        auth.unlink()


def check_login(deployed):
    login = json.loads(deployed['login'].read_text())
    try:
        if login.get('chatgpt_auth'):
            chatgpt_login(login['chatgpt_auth'])
        else:
            load_key(login['key_file'])
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f'Agent login unusable: {exc}') from None
    return login


def agent_command(config, deployed, *, prompt, interactive=False, json_output=False):
    """The native Codex command line for one run of the agent."""
    command = [deployed['codex'], '--cd', deployed['workspace'],
               # One-call configuration overrides (session compaction); never permissions.
               *deployed.get('codex_overrides', ())]
    if not interactive:
        command += ['exec']
        if deployed.get('resume_session'):
            command += ['resume']
        command += ['--skip-git-repo-check']
        if json_output:
            command.append('--json')
    elif deployed.get('resume_session'):
        command.append('resume')
    if deployed.get('resume_session'):
        command.append(deployed['resume_session'])
    command.append(prompt)
    return [str(part) for part in command]


def agent_environment(config, deployed, socket):
    """Codex's process environment: its session home, the robot socket and the agent's
    Python; a key-based login's key for Codex itself."""
    workspace = Path(deployed['workspace']).resolve()
    environment = agent_host.environment(home=workspace, workspace=workspace, socket=socket,
                                         cpus=config.training.cpus, binary=deployed['codex'])
    environment['CODEX_HOME'] = str(Path(deployed['codex_home']).resolve())
    login = json.loads(deployed['login'].read_text())
    if login.get('key_file'):
        environment[KEY_VARIABLE] = load_key(login['key_file'])
    return environment


def session_files(home, session_id):
    """Transcript files that resume a Codex thread in a fresh home, relative to it."""
    for path in Path(home, 'sessions').rglob('rollout-*.jsonl'):
        try:
            with path.open() as stream:
                if (json.loads(stream.readline()).get('payload') or {}).get('id') == session_id:
                    return [path.relative_to(home)]
        except (OSError, ValueError):
            continue
    raise RuntimeError(f'Codex thread {session_id} not found in {home}')


HOME_KEY, HOME_NAME = 'codex_home', 'codex-home'
# Trusted launcher-written configuration a copied session needs (never credentials).
HOME_FILES = ('config.toml',)
# The frontend closes its simulator before exiting; a scene teardown alone may take over a minute.
SERVER_STOP_SECONDS = 120


STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


@contextmanager
def deferred_sigterm():
    """Hold stop signals (SIGTERM, SIGHUP) until the block ends, then deliver them. A repeated
    stop request must not abort cleanup half way and orphan the frontend and its simulator."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending = []
    previous = {number: signal.signal(number, lambda number, _: pending.append(number)) for number in STOP_SIGNALS}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    for number in dict.fromkeys(pending):
        signal.raise_signal(number)


def end_process(process, seconds):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def stop_services(agent, server):
    """Launcher cleanup: every step runs even if an earlier one fails."""
    steps = [lambda: agent_host.stop(agent)] if agent else []
    steps += [lambda: end_process(server, SERVER_STOP_SECONDS)] if server else []
    errors = []
    with deferred_sigterm():
        for step in steps:
            try:
                step()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]


def open_frontend_log(root):
    """The session's frontend.log, appended to with a header per launch: an earlier attempt's
    tracebacks and simulator errors stay readable after a resume."""
    log = (Path(root)/'frontend.log').open('ab')
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    log.write(f'\n===== launch {stamp} (launcher pid {os.getpid()}) =====\n'.encode())
    log.flush()
    log.launch_start = log.tell()  # Where this launch's output begins.
    return log


def start_frontend(config, deployed, socket, log, *, key_file=None, timeout=30):
    """The session's robot service: sole owner of its episodes, serving the CLI and the agent's Python.
    It runs in its own session: a terminal's Ctrl+C reaches only the launcher, whose cleanup then
    stops the frontend (and through it the simulators) in order."""
    socket.unlink(missing_ok=True)
    args = [sys.executable, '-m', 'services.controller.frontend', '--config', str(deployed['configuration']),
            '--workspace', str(deployed['workspace']), '--socket', str(socket),
            '--agent-pid-file', str(config.root/PID_FILE)]
    if key_file:  # gemini_generate stays brokered by the host frontend.
        args += ['--key-file', str(key_file)]
    environment = {k: v for k, v in os.environ.items() if not k.startswith('RAX_')}
    from services.storage_root import ENVIRONMENT, artifact_root
    environment[ENVIRONMENT] = str(artifact_root())
    environment['PYTHONPATH'] = str(PROJECT)
    server = subprocess.Popen(args, cwd=PROJECT, env=environment, stdout=log, stderr=log, start_new_session=True)
    deadline = time.monotonic()+timeout
    while not socket.exists():
        if server.poll() is not None or time.monotonic() >= deadline:
            failure = frontend_failure(server, log, timeout)  # Before ending a hung one.
            end_process(server, SERVER_STOP_SECONDS)
            raise failure
        time.sleep(.05)
    return server


def pass_stderr(stream, errors):
    """Pass the agent CLI's stderr through to the launcher's, keeping its ERROR lines whole
    (Codex prints the provider's error there before it exits)."""
    for line in stream:
        try:
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()
        except (AttributeError, OSError, ValueError):
            pass  # Keep draining: a full pipe would stall the agent.
        if line.startswith(b'ERROR'):
            errors.append(line.decode('utf-8', 'replace').strip())


def frontend_failure(server, log, waited):
    """The frontend's own startup error, whole: how it ended (its exit code, or still starting after
    ``waited`` seconds) and the end of what it wrote to ``log`` (an open frontend.log) in this
    launch, where it wrote its traceback."""
    code = server.poll()
    ended = f'exited with code {code}' if code is not None else f'opened no socket within {waited:g} s'
    log.flush()
    try:
        with open(log.name, 'rb') as stream:
            stream.seek(log.launch_start)
            tail = stream.read()[-4000:].decode('utf-8', 'replace').strip() or '(empty)'
    except OSError as exc:
        tail = f'(unreadable: {type(exc).__name__}: {exc})'
    return RuntimeError(f'Host MCP startup failed: the frontend {ended}; frontend.log ends:\n{tail}')


def launch(config, deployed, *, key_file=None, prompt, interactive=False, capture=False, agent_runs=None):
    """The frontend, then native Codex in the workspace.

    The frontend (the sole owner of the session's episodes) stays up for every agent run of this
    launch: `agent_runs(run)` may call `run(prompt)` again after a run ends (a finished turn, a
    provider error), and the open episodes keep running in between."""
    if os.getuid() == 0:
        raise RuntimeError('Run the launcher as a non-root operator')
    check_login(deployed)
    socket = agent_host.socket_path(config.root)
    write_home(config, deployed, socket)
    server = agent = None
    with open_frontend_log(config.root) as log:
        try:
            server = start_frontend(config, deployed, socket, log, key_file=key_file)
            def start(command, **streams):
                process = agent_host.start(command, cwd=deployed['workspace'], interactive=interactive,
                                           env=agent_environment(config, deployed, socket),
                                           stdin=None if interactive else subprocess.DEVNULL, **streams)
                agent_host.record(process, config.root/PID_FILE)
                return process
            if capture:
                command = agent_command(config, deployed, prompt=prompt, interactive=interactive, json_output=True)
                agent = start(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = agent.communicate()
                return subprocess.CompletedProcess(command, agent.returncode, stdout, stderr)
            def run(prompt):
                """One agent run against this launch's frontend; its last ERROR line is kept."""
                nonlocal agent
                command = agent_command(config, deployed, prompt=prompt, interactive=interactive)
                errors = []
                if interactive:
                    agent = start(command)
                else:
                    agent = start(command, stderr=subprocess.PIPE)
                    copier = threading.Thread(target=pass_stderr, args=(agent.stderr, errors), daemon=True)
                    copier.start()
                agent.wait()
                if not interactive:
                    copier.join(timeout=10)
                deployed['agent_error'] = errors[-1] if errors else None
                return agent.returncode
            returncode = agent_runs(run) if agent_runs else run(prompt)
            return subprocess.CompletedProcess(agent.args, returncode, None, None)
        finally:
            try:
                stop_services(agent, server)
            finally:
                (config.root/PID_FILE).unlink(missing_ok=True)


RESUME_PROMPT = ("This session was interrupted and has now been resumed in the same workspace. Reread "
                 "memory/PROGRESS.md, call status, and continue from the live state. Do not repeat finished "
                 "work, and do not restart an episode merely because the conversation resumed.")
CONTINUE_PROMPT = ("Your previous turn ended while exploration is still open; any background workers it left "
                   "running were stopped, and your open episodes kept running. Call status, reread "
                   "memory/PROGRESS.md, and continue from the live state until exploration is closed.")
ERROR_PROMPT = ("Your previous run ended on an error while exploration is still open ({error}); any background "
                "workers it left running were stopped, and your open episodes kept running. Call status, reread "
                "memory/PROGRESS.md, and continue from the live state.")
MAX_RELAUNCHES = 3
# A run that fails (a provider overload, a dropped stream) is resumed after a pause; a run
# that keeps failing quickly is given up after this many tries in a row.
MAX_ERROR_RELAUNCHES = 5
ERROR_PAUSE_SECONDS = 300
STEADY_SECONDS = 1800  # A run this long resets the count of failures in a row.


def run_agent(start, *, deployed, prompt, interactive, open_exploration, latest, lifecycle, save,
              last_error=lambda: None, sleep=time.sleep):
    """Run the agent; while exploration is open, resume the same conversation when a
    non-interactive run ends its turn (a bounded number of times) or fails (after a pause)."""
    failures = 0
    while True:
        started = time.monotonic()
        returncode = start(prompt)
        if interactive or not open_exploration():
            return returncode
        if returncode != 0:
            failures = 1 if time.monotonic()-started >= STEADY_SECONDS else failures+1
            error = last_error() or f'exit code {returncode}'
            lifecycle.setdefault('agent_errors', []).append({'time': time.time(), 'returncode': returncode,
                                                             'error': error})
            save()
            if failures > MAX_ERROR_RELAUNCHES:
                return returncode
            next_prompt = ERROR_PROMPT.format(error=error)
        else:
            failures = 0  # Failures count only in a row.
            relaunches = lifecycle.get('relaunches', 0)
            if relaunches >= MAX_RELAUNCHES:
                return returncode
            next_prompt = CONTINUE_PROMPT
        try:
            deployed['resume_session'] = latest()
        except (RuntimeError, OSError, TypeError):
            return returncode  # No conversation to continue.
        if returncode != 0:
            sleep(ERROR_PAUSE_SECONDS)
        else:
            lifecycle['relaunches'] = lifecycle.get('relaunches', 0)+1
            save()
        prompt = next_prompt


def latest_codex_thread(home):
    """Id of the newest top-level (user) Codex thread in a session's Codex home."""
    newest = None
    for path in Path(home, 'sessions').rglob('rollout-*.jsonl'):
        try:
            with path.open() as stream:
                meta = json.loads(stream.readline()).get('payload') or {}
        except (OSError, ValueError):
            continue
        if meta.get('thread_source', 'user') != 'user' or not meta.get('id'):
            continue  # Subagent threads are not resumed on their own.
        if newest is None or path.stat().st_mtime > newest[0]:
            newest = (path.stat().st_mtime, meta['id'])
    if newest is None:
        raise RuntimeError('No Codex conversation to resume in '+str(home))
    return newest[1]


latest_session = latest_codex_thread


def resumable_lifecycle(record):
    """Earlier attempts of a session, or a reason it cannot be resumed."""
    if not record.is_file():
        raise ValueError('Nothing to resume: this session was never launched')
    previous = json.loads(record.read_text())
    if previous.get('status') == 'prepared-only':
        raise ValueError('The session was only prepared; launch it instead')
    if not previous.get('finished_at'):
        # The caller holds launch.lock, so no launcher is alive: the previous one died
        # without cleanup (kill -9, host reboot). Its end is the session's last activity.
        logs = [path.stat().st_mtime for path in (record.parent/'frontend.log', record.parent/'state.json')
                if path.exists()]
        previous.update(status='crashed', finished_at=max(logs, default=previous.get('started_at', time.time())))
    return [*previous.pop('attempts', []), previous]


def native_session(root, home_name):
    """Refuse a session that ran in an agent container: its agent knows its workspace as
    /workspace, which a native agent does not have. Resume it with the code it started on."""
    if (root/'agent-storage.json').exists() or (root/'model-relay.json').exists() or not (root/LOGIN_FILE).exists():
        raise RuntimeError(f'{root} ran its agent in a container; resume it with the release it started on')
    for path in (root/'agent-workspace', root/home_name, root/'configuration.json'):
        if not path.exists():
            raise FileNotFoundError(f'Cannot resume: {path} is missing')


def redeploy(config, codex=None):
    """Reattach an existing Codex session workspace for a resumed launch."""
    root = config.root
    native_session(root, HOME_NAME)
    deployed = {'workspace': root/'agent-workspace', 'codex_home': root/HOME_NAME, 'codex': codex_binary(codex),
                'configuration': root/'configuration.json', 'login': root/LOGIN_FILE}
    expose_published(root)
    deployed['resume_session'] = latest_codex_thread(deployed['codex_home'])
    refresh_reviews(config, deployed['workspace'])
    return deployed


def startup_prompt(workspace, override=None):
    """Keep the default kickoff editable alongside the standing instructions."""
    if override:
        return override
    if (workspace/'ADAPTATION.md').is_file():
        from services.adaptation import PROMPT
        return PROMPT
    return (workspace/'START_PROMPT.md').read_text().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--provider-home', type=Path, default=Path(os.environ.get('CODEX_HOME', Path.home()/'.codex')))
    parser.add_argument('--profile', help='Provider profile; default: <provider-home>/experiment-profile, else openrouter')
    parser.add_argument('--key-file', type=Path,
                        help='OpenRouter key file for Gemini and the OpenRouter model profile; default: '
                             '~/.codex/secrets/openrouter_api_key when it exists, else OPENROUTER_API_KEY (Gemini only)')
    parser.add_argument('--prepare-only', action='store_true', help='Disposable deployment check; starts no model/episode')
    parser.add_argument('--exec-prompt', help='Run noninteractively instead of opening the terminal UI')
    parser.add_argument('--non-interactive', action='store_true', help='Run the default research prompt without a terminal UI')
    parser.add_argument('--formal-only', action='store_true',
                        help='Post-hoc agent formal of an imported agent (adaptation_source, agent_formal on): '
                             'import, freeze and run agent formal without an exploring agent')
    parser.add_argument('--agent-formal', action='store_true',
                        help='With --resume of a session whose exploration has closed: play its formal layouts '
                             'with live copies of the frozen agent (agent formal), beside any bundle formal it ran')
    parser.add_argument('--resume', action='store_true',
                        help="Relaunch a stopped/exited/failed session: same workspace and state, "
                             "continuing the agent's own conversation")
    # agent_cli: claude (operator configuration) runs Claude Code instead; see harness/claude_cli.
    parser.add_argument('--claude-key-file', type=Path,
                        help='anthropic: an API key file; claude-login: override the experiment setup-token '
                             'file (default: harness/claude_cli/experiment_login.py). The personal '
                             '~/.claude login is always refused.')
    parser.add_argument('--claude-model', help='Override the provider default model')
    parser.add_argument('--claude-base-url', help='Anthropic-compatible endpoint for the custom provider')
    parser.add_argument('--agent-binary', type=Path, help='Native third-party agent CLI executable')
    parser.add_argument('--agent-model', help='Model override for the native agent backend')
    parser.add_argument('--agent-key-file', type=Path, help='Private model key file for the native agent backend')
    parser.add_argument('--agent-base-url', help='Private model API base URL')
    parser.add_argument('--agent-auth-home', type=Path, help='Dedicated native agent account login directory')
    args = parser.parse_args()
    if args.prepare_only and (args.resume or args.formal_only or args.agent_formal):
        parser.error('--prepare-only checks a fresh deployment; it cannot be combined with --resume, '
                     '--formal-only or --agent-formal')
    if not args.profile:
        chosen = args.provider_home/'experiment-profile'
        args.profile = chosen.read_text().strip() if chosen.is_file() else 'openrouter'
    args.key_file = openrouter_key_file(args.key_file)
    config = Configuration.from_dict(json.loads(args.config.read_text()))
    from services.controller.config import with_task_timeouts
    config = with_task_timeouts(config)
    def stop(*_):
        raise KeyboardInterrupt
    # A closed terminal (SIGHUP) stops the session like Stop does: the frontend runs in its own
    # session and would otherwise outlive the launcher with its simulators. A launcher started
    # under nohup keeps ignoring it.
    signal.signal(signal.SIGTERM, stop)
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, stop)
    config.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (config.root/'launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Another launcher is running this session ({config.root}); stop it with '
                             'scripts/stop_auto_research_agent.sh and wait for its cleanup before launching again')
        lifecycle = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'preparing'}
        record = config.root/'agent-lifecycle.json'
        from services.agent_formal.checkpoint import exploration_closed
        closed = args.resume and exploration_closed(config)  # Only the frozen formal remains to run.
        if args.formal_only:
            if args.resume or not (config.agent_formal and config.adaptation_source):
                raise SystemExit('--formal-only needs a fresh configuration with agent formal and an adaptation source')
            closed = True
        if args.agent_formal and not (args.resume and closed):
            raise SystemExit('--agent-formal needs --resume of a session whose exploration has closed')
        if args.resume:
            try:
                attempts = resumable_lifecycle(record)
            except ValueError as exc:
                raise SystemExit(str(exc))
            if not (config.root/'configuration.json').is_file():
                failed = attempts[-1].get('error_type') if attempts else None
                raise SystemExit('Nothing to resume: the first launch failed before the agent workspace was '
                                 f'prepared{f" ({failed})" if failed else ""}; read its error in the launch output '
                                 f'({config.root/"agent.log"} for dashboard launches), fix the cause and prepare a '
                                 'fresh configuration')
            lifecycle.update(resumed=True, attempts=attempts)
        elif record.exists():
            # A repeated launch must not overwrite the original run's lifecycle.
            raise FileExistsError('This session was already launched; use --resume or a fresh configuration')
        persist(record, lifecycle)
        deployed = None
        try:
            if config.agent_cli == 'claude':
                from harness.claude_cli import auto_research as claude
            elif config.agent_cli not in ('codex', 'claude'):
                from harness.backends import get
                external_cli = get(config.agent_cli)
                if args.agent_formal or args.formal_only:
                    raise ValueError(f'{config.agent_cli} supports code formal; live agent formal is not implemented')
            if args.resume:
                if config.agent_cli == 'claude' and config.agent_provider == 'claude-login' and not args.claude_key_file:
                    claude.preflight_login()  # The token may have been revoked since the first launch.
                deployed = (external_cli.redeploy(config, binary=args.agent_binary) if config.agent_cli not in ('codex', 'claude') else
                            claude.redeploy(config) if config.agent_cli == 'claude' else redeploy(config))
            elif config.agent_cli not in ('codex', 'claude'):
                deployed = external_cli.deploy(config, binary=args.agent_binary, model=args.agent_model,
                    key_file=args.agent_key_file or args.key_file, base_url=args.agent_base_url,
                    auth_home=args.agent_auth_home)
            elif config.agent_cli == 'claude':
                provider = config.agent_provider
                key_file = {'openrouter': args.key_file, 'anthropic': args.claude_key_file,
                            'claude-login': args.claude_key_file, 'custom': args.claude_key_file}[provider]
                if provider == 'claude-login' and not args.claude_key_file:
                    # Fail before provisioning if the experiment token is missing or rejected.
                    claude.preflight_login()
                deployed = claude.deploy(config, provider=provider, model=args.claude_model, key_file=key_file,
                                        base_url=args.claude_base_url)
            else:
                deployed = deploy(config, provider_home=args.provider_home, profile=args.profile, key_file=args.key_file)
            if args.prepare_only:
                print(json.dumps({k: str(v) for k, v in deployed.items()}, indent=2))
                lifecycle['status'] = 'prepared-only'
                return
            prompt = args.exec_prompt or RESUME_PROMPT if args.resume else startup_prompt(deployed['workspace'], args.exec_prompt)
            if (not args.resume and config.adaptation_source and config.adaptation_context == 'compact'
                    and deployed.get('resume_session')):
                # Compact the carried conversation once, before its first turn here.
                from harness.session_context import compact
                persist(config.root/'adaptation-compaction.json', compact(
                    config, deployed, deployed['resume_session'], log=config.root/'compaction.jsonl'))
            if args.formal_only:
                # No exploring agent: the imported one is frozen as it arrived.
                persist(config.root/'exploration-complete.json', {'finished_at': time.time(), 'formal_only': True})
            if closed:
                returncode = 0  # Exploration already ended; only the frozen formal remains.
            else:
                lifecycle['status'] = 'running'
                persist(record, lifecycle)
                interactive = not (args.non_interactive or args.exec_prompt)
                base_wait = 0.0
                def usage_wait(until, waited):
                    nonlocal base_wait
                    if until is None:  # One wait finished; record the total time waited.
                        lifecycle.update(status='running', usage_wait_seconds=base_wait+waited)
                        lifecycle.pop('usage_limited_until', None)
                    else:
                        lifecycle.update(status='usage_wait', usage_limited_until=until)
                    persist(record, lifecycle)
                from services.exploration.pool import recover_interrupted, report_refunds
                recovered = recover_interrupted(config.root/'exploration')  # Left live by a previous launcher.
                if recovered:
                    lifecycle.setdefault('interrupted_episodes', []).extend(recovered)
                    persist(record, lifecycle)
                from harness.backends import get
                cli = get(config.agent_cli)
                home = deployed.get(cli.HOME_KEY)
                def agent_runs(run):
                    """Every agent run of this launcher, against one frontend: open episodes outlive a run."""
                    def start(prompt):
                        nonlocal base_wait
                        base_wait = lifecycle.get('usage_wait_seconds', 0.0)
                        note = report_refunds(config.root/'exploration')  # Told once, with the next prompt.
                        return run(f'{prompt}\n\n{note}' if note else prompt)
                    return run_agent(
                        start, deployed=deployed, prompt=prompt, interactive=interactive,
                        open_exploration=lambda: not exploration_closed(config),
                        latest=lambda: cli.latest_session(home),
                        last_error=lambda: (cli.last_error(home) if config.agent_cli != 'codex'
                                            else deployed.get('agent_error')),
                        lifecycle=lifecycle, save=lambda: persist(record, lifecycle))
                if config.agent_cli not in ('codex', 'claude'):
                    returncode = external_cli.launch(config, deployed, gemini_key_file=args.key_file, prompt=prompt,
                                           interactive=interactive, agent_runs=agent_runs).returncode
                elif config.agent_cli == 'claude':
                    returncode = claude.launch(config, deployed, gemini_key_file=args.key_file, prompt=prompt,
                                               interactive=interactive, on_wait=usage_wait,
                                               agent_runs=agent_runs).returncode
                else:
                    returncode = launch(config, deployed, key_file=args.key_file, prompt=prompt,
                                        interactive=interactive, agent_runs=agent_runs).returncode
            lifecycle.update(status='exited', returncode=returncode)
            # The exploring agent has ended: freeze it, then run formal in isolation.
            lifecycle['status'] = 'formal'
            persist(record, lifecycle)
            from services.agent_formal.checkpoint import after_exploration
            summary = after_exploration(config, deployed, gemini_key_file=args.key_file, live_copies=args.agent_formal)
            lifecycle.update(status='exited', formal=summary['status'],
                             **({'agent_formal': summary['agent_formal'].get('status')}
                                if isinstance(summary.get('agent_formal'), dict) else {}))
            raise SystemExit(returncode)
        except KeyboardInterrupt:
            lifecycle['status'] = 'stopped'
            raise SystemExit(130)
        except Exception as exc:
            lifecycle.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            lifecycle['finished_at'] = time.time()
            persist(record, lifecycle)
            if deployed and lifecycle['status'] != 'prepared-only':
                record_subagents(config, deployed)


def record_subagents(config, deployed):
    """Subagent spawn/concurrency record for the multi-agent baseline (best effort)."""
    from services import subagent_metrics
    from services.mcp_contract import Contract
    if config.agent_cli not in ('codex', 'claude'):
        from harness.backends import get
        return get(config.agent_cli).record_subagents(config, deployed)
    claude = config.agent_cli == 'claude'
    home = deployed.get('claude_home' if claude else 'codex_home')
    if home:
        subagent_metrics.write(home, config.root/'subagents.json', cli='claude' if claude else 'codex',
                               cap=Contract.from_config(config).subagent_cap)


if __name__ == '__main__':
    main()
