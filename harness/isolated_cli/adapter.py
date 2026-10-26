"""Run the real vendor CLI inside the common RAX filesystem/process boundary."""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess

from harness import agent_host
from harness.cli_runtime import (LOGIN_FILE, PID_FILE, expose_published,
    native_session, prepare_workspace, reference_sources, refresh_reviews, start_frontend, stop_services)
from harness.isolated_cli.credentials import Credential
from harness.isolated_cli.relay import serve
from services.controller.storage import persist

HERE = Path(__file__).resolve().parent


class Adapter:
    def __init__(self, spec):
        self.spec = spec
        self.HOME_KEY = spec.NAME+'_home'
        self.HOME_NAME = spec.NAME+'-home'
        self.HOME_FILES = ()

    def binary(self, path=None):
        binary = Path(path or self.spec.default_binary()).resolve()
        if not binary.is_file():
            raise RuntimeError(f'Install the pinned {self.spec.NAME} CLI or pass --agent-binary')
        return binary

    def credential(self, deployed):
        value = json.loads(Path(deployed['login']).read_text())
        return Credential(self.spec, value['provider'], key_file=value.get('key_file'),
            auth_home=value.get('auth_home'), base_url=value.get('base_url'), binary=value['binary'])

    def deploy(self, config, *, binary=None, model=None, key_file=None, auth_home=None, base_url=None):
        binary = self.binary(binary)
        credential = Credential(self.spec, config.agent_provider, key_file=key_file,
                                auth_home=auth_home, base_url=base_url, binary=binary)
        prepared = prepare_workspace(config, self.HOME_NAME)
        model = model or self.spec.model(config.agent_provider)
        persist(config.root/LOGIN_FILE, {'agent_cli': self.spec.NAME, 'model': model,
            **credential.record(), 'binary': str(binary), 'node': shutil.which('node')})
        deployed = {'workspace': prepared['workspace'], self.HOME_KEY: prepared['home'],
            'binary': binary, 'configuration': prepared['configuration'],
            'login': config.root/LOGIN_FILE, 'resume_session': None, 'subagents': 0}
        self.write_home(config, deployed)
        return deployed

    def redeploy(self, config, *, binary=None):
        native_session(config.root, self.HOME_NAME)
        value = json.loads((config.root/LOGIN_FILE).read_text())
        deployed = {'workspace': config.root/'agent-workspace', self.HOME_KEY: config.root/self.HOME_NAME,
            'binary': self.binary(binary or value['binary']), 'configuration': config.root/'configuration.json',
            'login': config.root/LOGIN_FILE, 'subagents': 0,
            'resume_session': self.latest_session(config.root/self.HOME_NAME)}
        expose_published(config.root)
        refresh_reviews(config, deployed['workspace'])
        self.credential(deployed)
        return deployed

    def write_home(self, config, deployed, socket=None):
        socket = socket or agent_host.socket_path(config.root)
        value = json.loads(Path(deployed['login']).read_text())
        self.spec.write_home(config, Path(deployed[self.HOME_KEY]), value['model'], socket)

    def sandbox(self, config, deployed, command):
        workspace = Path(deployed['workspace']).resolve()
        home = Path(deployed[self.HOME_KEY]).resolve()
        socket_dir = agent_host.socket_path(config.root).parent.resolve()
        value = json.loads(Path(deployed['login']).read_text())
        install = self.spec.installation(Path(value['binary']))
        # Mask the operator's entire home, not only known credential filenames.
        hidden = [Path.home().resolve(), *agent_host.hidden_paths(config.root)]
        hidden = list(dict.fromkeys(hidden))
        hidden = [p for p in hidden if not any(p != q and p.is_relative_to(q) for q in hidden)]
        args = [shutil.which('bwrap') or '/usr/bin/bwrap', '--ro-bind', '/', '/', '--dev', '/dev',
                '--unshare-net', '--unshare-pid', '--proc', '/proc', '--tmpfs', '/tmp', '--die-with-parent']
        masks = []
        for path in hidden:
            if path.is_dir():
                args += ['--tmpfs', str(path)]
                masks.append(path)
            elif path.exists():
                args += ['--ro-bind', '/dev/null', str(path)]
        readable = [*agent_host.reexposed_paths(), Path(value['node']).resolve().parent,
                    install, HERE, *getattr(self.spec, 'READABLE_PATHS', ()), config.root/'published',
                    *(source for source, _ in reference_sources(config))]
        for path in dict.fromkeys(Path(p).resolve() for p in readable):
            args += ['--ro-bind', str(path), str(path)]
        # The stdio bridge is the only executable exposed from the controller.
        args += ['--ro-bind', str(agent_host.BRIDGE), '/tmp/rax-bridge/mcp_bridge.py',
                 '--bind', str(workspace), str(workspace), '--bind', str(home), str(home),
                 '--bind', str(socket_dir), str(socket_dir)]
        if hasattr(self.spec, 'runtime_mounts'):
            for source, destination in self.spec.runtime_mounts(home):
                args += ['--bind', str(source), str(destination)]
        for path in masks:
            args += ['--remount-ro', str(path)]
        return [*args, '--chdir', str(workspace), '--', *command]

    def agent_command(self, config, deployed, *, prompt, interactive=False, json_output=False):
        if interactive:
            raise ValueError(f'{self.spec.NAME} currently uses its headless CLI')
        value = json.loads(Path(deployed['login']).read_text())
        command = [value['node'], str(deployed['binary']),
                   *self.spec.arguments(prompt, deployed.get('resume_session'))]
        return self.sandbox(config, deployed, command)

    def agent_environment(self, config, deployed, socket):
        value = json.loads(Path(deployed['login']).read_text())
        env = agent_host.environment(home=deployed[self.HOME_KEY], workspace=deployed['workspace'],
            socket=socket, cpus=config.training.cpus, binary=value['node'])
        env.update(self.spec.environment(Path(deployed[self.HOME_KEY]), install=self.spec.installation(Path(deployed['binary']))))
        env.update(NODE_OPTIONS=f'--import={HERE/"transport.mjs"}',
                   RAX_MODEL_SOCKET=str(Path(socket).parent/'model.sock'), TMPDIR='/tmp')
        return env

    def latest_session(self, home):
        path = Path(home)/'rax-session.json'
        return json.loads(path.read_text())['session_id'] if path.exists() else None

    def last_error(self, home):
        path = Path(home)/'rax-error.json'
        return json.loads(path.read_text())['error'] if path.exists() else None

    def capture_event(self, home, line):
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        ident, error, usage = self.spec.event(event)
        if ident:
            persist(Path(home)/'rax-session.json', {'session_id': ident})
        if error:
            persist(Path(home)/'rax-error.json', {'error': error})
        if usage:
            with (Path(home)/'usage.jsonl').open('a') as stream:
                stream.write(json.dumps(usage)+'\n')

    def session_files(self, home, session_id):
        return self.spec.session_files(Path(home), session_id)

    def record_subagents(self, config, deployed):
        persist(config.root/'subagents.json', {'cli': self.spec.NAME, 'cap': 0, 'spawned': 0,
                                             'enforcement': 'vendor subagent tools disabled'})

    @contextmanager
    def relay(self, config, deployed):
        value = json.loads(Path(deployed['login']).read_text())
        home = Path(deployed[self.HOME_KEY])
        state = self.spec.runtime_state(home) if hasattr(self.spec, 'runtime_state') else nullcontext()
        with state, serve(agent_host.socket_path(config.root).parent/'model.sock',
                credential=self.credential(deployed), model=value['model'], effort=config.agent_effort,
                audit=home/'request-audit.jsonl') as server:
            yield server

    def launch(self, config, deployed, *, gemini_key_file=None, prompt, interactive=False,
               capture=False, agent_runs=None, **unused):
        if interactive or capture:
            raise ValueError('Use non-interactive streaming mode')
        socket = agent_host.socket_path(config.root)
        self.write_home(config, deployed, socket)
        server = agent = None
        home = Path(deployed[self.HOME_KEY])
        with (config.root/'frontend.log').open('ab') as log, self.relay(config, deployed):
            try:
                server = start_frontend(config, deployed, socket, log, key_file=gemini_key_file)
                def run(text):
                    nonlocal agent
                    (home/'rax-error.json').unlink(missing_ok=True)
                    agent = agent_host.start(self.agent_command(config, deployed, prompt=text),
                        cwd=deployed['workspace'], env=self.agent_environment(config, deployed, socket),
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                    agent_host.record(agent, config.root/PID_FILE)
                    try:
                        for line in agent.stdout:
                            self.capture_event(home, line)
                            print(line, end='', flush=True)
                        return agent.wait()
                    finally:
                        agent_host.stop(agent)
                        (config.root/PID_FILE).unlink(missing_ok=True)
                code = agent_runs(run) if agent_runs else run(prompt)
                return subprocess.CompletedProcess(agent.args, code)
            finally:
                stop_services(agent, server)
