"""Qoder uses the shared native lifecycle and a restricted account connection."""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path

from harness import agent_host
from harness.claude_cli.login import private_text
from harness.isolated_cli.adapter import Adapter
from . import auth_proxy, spec


def token_file(auth_home):
    if not auth_home:
        raise ValueError('Qoder requires --agent-auth-home with browser login or a private qoder-pat file')
    home = Path(auth_home).resolve()
    if not home.is_dir() or home.stat().st_uid != os.getuid() or home.stat().st_mode & 0o077:
        raise ValueError('Qoder experiment auth home must belong to this user and have mode 0700')
    path = home/'qoder-pat'
    if not path.exists():
        for name in ('user', 'machine_id'):
            if not private_text(home/'.auth'/name).strip():
                raise ValueError('Qoder browser login is incomplete; run qodercn login first')
        return None
    if not private_text(path).strip():
        raise ValueError('Qoder personal access token is empty')
    return path


class QoderAdapter(Adapter):
    def deploy(self, config, *, binary=None, model=None, key_file=None, auth_home=None, base_url=None):
        token_file(auth_home)
        return super().deploy(config, binary=binary, model=model, key_file=key_file,
                              auth_home=auth_home, base_url=base_url)

    def credential(self, deployed):
        credential = super().credential(deployed)
        token_file(credential.auth_home)
        return credential

    def sandbox(self, config, deployed, command):
        args = super().sandbox(config, deployed, command)
        credential = self.credential(deployed)
        if token_file(credential.auth_home) is None:
            # Only this experiment's browser login store is shared with the CLI.
            destination = Path(deployed[self.HOME_KEY])/'.auth'
            destination.mkdir(mode=0o700, exist_ok=True)
            position = args.index('--chdir')
            args[position:position] = ['--bind', str(Path(credential.auth_home)/'.auth'), str(destination)]
        return args

    def agent_command(self, config, deployed, *, prompt, interactive=False, json_output=False):
        command = super().agent_command(config, deployed, prompt=prompt,
                                       interactive=interactive, json_output=json_output)
        # The CLI starts in the workspace; its MCP file belongs to its private home.
        command[command.index('--mcp-config')+1] = str(Path(deployed[self.HOME_KEY])/'mcp.json')
        login = json.loads(Path(deployed['login']).read_text())
        position = command.index('--print')
        model = login['model'] if login['provider'] == spec.ACCOUNT else 'rax/'+login['model']
        command[position:position] = ['--model', model,
                                     '--reasoning-effort', config.agent_effort]
        return command

    def agent_environment(self, config, deployed, socket):
        env = super().agent_environment(config, deployed, socket)
        if token_file(self.credential(deployed).auth_home) is None:
            del env['QODERCN_PERSONAL_ACCESS_TOKEN']
        env.update(NODE_OPTIONS=f'--import={Path(__file__).with_name("transport.mjs")}',
                   RAX_QODER_ENTRY=str(deployed['binary']),
                   RAX_QODER_AUTH_SOCKET=str(Path(socket).parent/'qoder-auth.sock'))
        if config.agent_provider == spec.ACCOUNT:
            env.update(RAX_QODER_AUDIT=str(Path(deployed[self.HOME_KEY])/'request-audit.jsonl'),
                       RAX_QODER_MODEL=json.loads(Path(deployed['login']).read_text())['model'],
                       RAX_QODER_EFFORT=config.agent_effort)
        return env

    @contextmanager
    def relay(self, config, deployed):
        credential = self.credential(deployed)
        builtin = credential.provider == spec.ACCOUNT
        with (nullcontext() if builtin else super().relay(config, deployed)), auth_proxy.serve(
                agent_host.socket_path(config.root).parent/'qoder-auth.sock',
                token_file(credential.auth_home), inference=builtin) as server:
            yield server


_adapter = QoderAdapter(spec)
HOME_KEY, HOME_NAME, HOME_FILES = _adapter.HOME_KEY, _adapter.HOME_NAME, _adapter.HOME_FILES
deploy, redeploy = _adapter.deploy, _adapter.redeploy
write_home, agent_command = _adapter.write_home, _adapter.agent_command
agent_environment, launch = _adapter.agent_environment, _adapter.launch
latest_session, session_files = _adapter.latest_session, _adapter.session_files
last_error, capture_event = _adapter.last_error, _adapter.capture_event
record_subagents, relay = _adapter.record_subagents, _adapter.relay
binary, sandbox = _adapter.binary, _adapter.sandbox
