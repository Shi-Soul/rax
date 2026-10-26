#!/usr/bin/env python3
"""Validate a real native agent: MCP, images, files, shell, resume and isolation.

This uses a tiny MCP fixture and an existing robot image. It consumes model
tokens, but no robot exploration episodes or GPU allocations.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import socketserver
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from harness import agent_host
from harness.backends import get
from harness.cli_runtime import CONTAINER
from harness.model_serving.lifecycle import write_json
from services.controller.config import Configuration, Limits


class Fixture(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            request = json.loads(line)
            if 'id' not in request:
                continue
            method, params = request['method'], request.get('params', {})
            result = {}
            if method == 'initialize':
                result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'rax-native-cli-validation', 'version': '1'}}
            elif method == 'tools/list':
                result = {'tools': [
                    {'name': 'report_number', 'description': 'Accept integer 7 and return a validation marker.',
                     'inputSchema': {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']}},
                    {'name': 'get_robot_image', 'description': 'Return a robot scene image for this protocol test.',
                     'inputSchema': {'type': 'object', 'properties': {}}}]}
            elif method == 'tools/call':
                name = params['name']
                if name == 'report_number':
                    if params['arguments']['value'] != 7:
                        raise ValueError('Wrong fixture argument')
                    result = {'content': [{'type': 'text', 'text': 'Validation marker: '+self.server.marker}]}
                elif name == 'get_robot_image':
                    result = {'content': [{'type': 'image', 'mimeType': 'image/jpeg', 'data': self.server.image}]}
                else:
                    raise ValueError('Unknown validation tool')
                self.server.calls.append(name)
            self.wfile.write((json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result})+'\n').encode())
            self.wfile.flush()


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def fixture_workspace(cli, config, args):
    """Tiny protocol fixture; no simulator, demonstrations or trial storage image."""
    native = cli.native if CONTAINER else cli
    workspace, home = config.root/'agent-workspace', config.root/cli.HOME_NAME
    for path in (workspace/'output', workspace/'runtime', home, config.root/'published'):
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    artifacts = workspace/'runtime/autonomous_controller'
    if args.backend == 'dsh':
        artifacts.symlink_to(config.root/'published', target_is_directory=True)
    else:
        artifacts.mkdir()
    (workspace/'AGENTS.md').write_text('Validate the MCP, native file and shell tools. No robot episodes are available.\n')
    if args.backend == 'dsh':
        from harness.deepseek_cli.login import Credential, MODEL
        binary = native.dsh_binary(args.binary)
        credential = Credential(args.provider, key_file=args.key_file,
                                auth_home=args.auth_home, base_url=args.base_url)
        model = args.model or (MODEL if args.provider == 'openrouter' else 'deepseek-flash')
        binaries = {'dsh': binary}
    else:
        import importlib
        from harness.isolated_cli.credentials import Credential
        spec = importlib.import_module('harness.'+args.backend+'_cli.spec')
        binary = native.binary(args.binary)
        credential = Credential(spec, args.provider, key_file=args.key_file,
            auth_home=args.auth_home, base_url=args.base_url, binary=binary)
        model, binaries = args.model or spec.model(args.provider), {'binary': binary}
    import shutil
    login = config.root/'agent-login.json'
    write_json(login, {'agent_cli': args.backend, 'model': model, **credential.record(),
                      'binary': str(binary), 'node': shutil.which('node')})
    login.chmod(0o600)
    deployed = {'workspace': workspace, cli.HOME_KEY: home, **binaries,
        'login': login, 'configuration': config.root/'configuration.json', 'resume_session': None,
        'subagents': 0}
    cli.write_home(config, deployed)
    return deployed


def boundary(cli, config, deployed):
    outside = config.root/'private-canary.txt'
    outside.write_text('private validation canary')
    workspace = deployed['workspace']
    link = workspace/'output/canary-link'
    link.symlink_to(outside)
    code = f'''
import os,socket
from pathlib import Path
assert not Path('/dev/nvidia0').exists()
assert not any(('KEY' in k or 'TOKEN' in k) and k != 'GPG_KEY' and v != 'rax-session-transport' for k, v in os.environ.items())
for path in {list(map(str, [outside, link, PROJECT/'README.md', Path.home()/'.config/rax']))!r}:
    try: Path(path).read_text()
    except OSError: pass
    else: raise AssertionError('Private file visible: '+path)
for path in {list(map(str, [config.root/'published/forbidden.txt', workspace/'runtime/autonomous_controller/forbidden.txt', Path('/workspace/runtime/autonomous_controller/forbidden.txt')]))!r}:
    try: Path(path).write_text('forbidden')
    except OSError: pass
    else: raise AssertionError('Published recordings writable: '+path)
client=socket.socket();client.settimeout(2)
assert client.connect_ex(('1.1.1.1',443)) != 0
client.close()
client=socket.socket(socket.AF_UNIX);client.settimeout(5)
client.connect({str(agent_host.socket_path(config.root))!r})
client.sendall(b'{{"jsonrpc":"2.0","id":11,"method":"tools/list"}}\\n')
assert b'result' in client.recv(65536)
Path({str(workspace/'output/boundary.txt')!r}).write_text('OK')
'''
    try:
        result = subprocess.run(cli.sandbox(config, deployed,
            ['/usr/local/bin/python' if CONTAINER else str(agent_host.agent_python()), '-I', '-c', code]),
            cwd=workspace, env=(dict(os.environ) if CONTAINER else
                cli.agent_environment(config, deployed, agent_host.socket_path(config.root))),
            text=True, capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError('Process boundary validation failed: '+result.stderr[-2000:])
    finally:
        link.unlink(missing_ok=True)
    return {'offline_tools': True, 'hidden_paths': True, 'symlink_containment': True,
            'readonly_recordings': True, 'no_gpu': True, 'scrubbed_credentials': True, 'robot_socket': True}


def run(cli, config, deployed, prompt, label, timeout):
    events = []
    with (config.root/f'{label}.stderr.log').open('w') as err, (config.root/f'{label}.stdout.jsonl').open('w') as out:
        child = agent_host.start(cli.agent_command(config, deployed, prompt=prompt),
            cwd=deployed['workspace'], env=(dict(os.environ) if CONTAINER else
                cli.agent_environment(config, deployed, agent_host.socket_path(config.root))),
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, text=True)
        try:
            code = child.wait(timeout=timeout)
        finally:
            agent_host.stop(child)
    for line in (config.root/f'{label}.stdout.jsonl').read_text().splitlines():
        cli.capture_event(deployed[cli.HOME_KEY], line)
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    if code:
        raise RuntimeError(f'{label}: CLI exited {code}; inspect its private validation logs')
    if cli.last_error(deployed[cli.HOME_KEY]):
        raise RuntimeError(f'{label}: native CLI emitted an error')
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--provider', default='openrouter')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--auth-home', type=Path)
    parser.add_argument('--base-url')
    parser.add_argument('--model')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--fixture-workspace', action='store_true', help='Use only the small MCP fixture workspace')
    parser.add_argument('--container-image', help='Digest-pinned CPU image for container validation')
    parser.add_argument('--benchmark')
    parser.add_argument('--task')
    args = parser.parse_args()
    cli = get(args.backend)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    limits = Limits(600, 8192, 4, 256, 1024, 10*1024**3)
    from services.benchmarks import DEFAULT, BENCHMARKS
    benchmark = args.benchmark or DEFAULT
    if CONTAINER and not args.container_image:
        parser.error('Container validation requires --container-image')
    config = Configuration(root=root, task=args.task or BENCHMARKS[benchmark].task_names()[0],
        exploration_seeds=tuple(range(16)), formal_seed=0, eval_seed=0, sim_gpu='GPU-aaaa',
        development=limits, formal=limits, training=limits, agent_cli=args.backend,
        agent_provider=args.provider, agent_effort='max', benchmark=benchmark,
        formal_episodes=1, formal_eval_seed=1, external_perception=False,
        demonstration_context='none' if args.fixture_workspace else 'terminal_state',
        **({'image': args.container_image} if CONTAINER else {}))
    deployed = (fixture_workspace(cli, config, args) if args.fixture_workspace else
        cli.deploy(config, binary=args.binary, model=args.model, key_file=args.key_file,
                   auth_home=args.auth_home, base_url=args.base_url))
    path = agent_host.socket_path(root)
    path.unlink(missing_ok=True)
    server = Server(str(path), Fixture)
    server.marker = 'RAX-'+secrets.token_hex(4)
    server.image = base64.b64encode(args.image.read_bytes()).decode()
    server.calls = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    report = {'status': 'running', 'backend': args.backend, 'provider': args.provider,
        'model': json.loads(deployed['login'].read_text())['model'], 'effort': 'max',
        'started_at': time.time(), 'synthetic_mcp_fixture': True, 'exploration_episodes_used': 0}
    write_json(root/'validation.json', report)
    try:
        with cli.relay(config, deployed):
            report['boundary'] = boundary(cli, config, deployed)
            prompt = ('Infrastructure validation. Read AGENTS.md, but do not start robot episodes. '
                'Call report_number with value 7, then get_robot_image and inspect its image. '
                'Use your native file tool to create output/native-file.txt containing NATIVE_OK. '
                'Use your native Bash tool to run Python and save the exact marker returned by report_number '
                'in output/native-shell.txt. Finish with READY, the exact marker, and a description of the image. '
                'Only these two fixture MCP tools are available; no robot episodes can be started.')
            run(cli, config, deployed, prompt, 'first', args.timeout)
            assert (deployed['workspace']/'output/native-file.txt').read_text().strip() == 'NATIVE_OK'
            assert (deployed['workspace']/'output/native-shell.txt').read_text().strip() == server.marker
            assert {'report_number', 'get_robot_image'} <= set(server.calls)
            session_id = cli.latest_session(deployed[cli.HOME_KEY])
            assert session_id, 'No native session identity was emitted'
            deployed['resume_session'] = session_id
            calls = len(server.calls)
            events = run(cli, config, deployed,
                'Use no tools. Repeat the exact validation marker from our previous conversation and say RESUMED.',
                'resume', args.timeout)
            # Vendor CLIs may split the remembered marker across streaming chunks.
            text = json.dumps(events)+''.join(
                e.get('payload', {}).get('delta', '') for e in events
                if e.get('type') == 'model.streaming' and
                e.get('payload', {}).get('kind') == 'text_delta')
            assert server.marker in text and 'RESUMED' in text, 'Persisted conversation did not resume'
            assert len(server.calls) == calls, 'Resume validation unexpectedly called a tool'
            assert cli.latest_session(deployed[cli.HOME_KEY]) == session_id
        audit = [json.loads(line) for line in (deployed[cli.HOME_KEY]/'request-audit.jsonl').read_text().splitlines()]
        assert audit and all(r['effort'] == 'max' for r in audit)
        assert any(r['image_blocks'] > 0 for r in audit), 'No image reached the provider request'
        report.update(status='passed', tool_call=True, tool_result=True, native_file=True, native_shell=True,
                      session_id=session_id, resume=True, vision_wire=True, wire_max_effort=True,
                      requests=len(audit), tool_calls=len(server.calls))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        server.shutdown()
        server.server_close()
        report['finished_at'] = time.time()
        write_json(root/'validation.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
