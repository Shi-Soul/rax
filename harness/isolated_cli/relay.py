"""A session-local Chat Completions transport, with host-owned credentials.

This forwards requests made by the vendor's own agent loop. It does not choose
tools, generate prompts, or implement an agent. Only the selected model and
endpoint are reachable; audit records omit prompts, images and credentials.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import socketserver
import threading
import time
import uuid
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_request(url, method, headers, body, timeout):
    """Use the host proxy settings without following credential-bearing redirects."""
    request = Request(url, data=body, headers=headers, method=method)
    try:
        return build_opener(NoRedirect).open(request, timeout=timeout)
    except HTTPError as response:
        return response


def image_count(value):
    if isinstance(value, list):
        return sum(map(image_count, value))
    if isinstance(value, dict):
        return int(value.get('type') in ('image', 'image_url', 'input_image')) + sum(
            image_count(v) for v in value.values())
    return 0


def prepare_request(body, model, effort, *, openrouter=False):
    if body.get('model') != model:
        raise ValueError('Only the configured experiment model is allowed')
    if not isinstance(body.get('messages'), list):
        raise ValueError('Expected Chat Completions messages')
    # Model effort is an operator setting, never a model-selected option.
    body = dict(body)
    body.pop('reasoning_effort', None)
    if openrouter:
        body['reasoning'] = {'effort': effort}
        body['provider'] = {**body.get('provider', {}), 'require_parameters': True}
        if body.get('stream'):
            body['stream_options'] = {**body.get('stream_options', {}), 'include_usage': True}
    else:
        body['reasoning_effort'] = effort
        body['thinking'] = {'type': 'enabled'}
    return body


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *_):
        pass

    def do_POST(self):
        relay = self.server
        if self.path != '/v1/chat/completions':
            self.send_error(403, 'Only Chat Completions is enabled')
            return
        upstream, response_started, usage = None, False, None
        request_id, status = uuid.uuid4().hex, None
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 128*1024**2:
                raise ValueError('Invalid request size')
            incoming = json.loads(self.rfile.read(size))
            body = prepare_request(incoming, relay.model, relay.effort,
                                   openrouter=relay.provider == 'openrouter')
            event = {'at': time.time(), 'request_id': request_id, 'model': body['model'], 'effort': relay.effort,
                     'cli_effort': incoming.get('reasoning_effort', incoming.get('reasoning')),
                     'image_blocks': image_count(body['messages']),
                     'tools': [t.get('function', {}).get('name') for t in body.get('tools', [])]}
            with relay.audit_lock, relay.audit.open('a') as log:
                log.write(json.dumps(event)+'\n')
            headers = {'Content-Type': 'application/json', **relay.credential.headers()}
            upstream = open_request(relay.credential.endpoint.rstrip('/')+'/chat/completions',
                                    'POST', headers, json.dumps(body).encode(), 900)
            response = upstream
            status = response.status
            self.send_response(response.status)
            self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
            self.end_headers()
            response_started, pending = True, b''
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if line.startswith(b'data:'):
                        try:
                            value = json.loads(line[5:])
                            if isinstance(value.get('usage'), dict):
                                usage = value['usage']
                        except (ValueError, AttributeError):
                            pass
                pending = pending[-4*1024**2:]
            if usage is None and pending:
                try:
                    usage = json.loads(pending).get('usage')
                except (ValueError, AttributeError):
                    pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            # Never include upstream exception payloads or request headers.
            if not response_started:
                self.send_error(502, 'Model transport failed: '+type(error).__name__)
        finally:
            if upstream:
                upstream.close()
            with relay.audit_lock, relay.audit.with_name('response-usage.jsonl').open('a') as log:
                log.write(json.dumps({'request_id': request_id, 'at': time.time(),
                    'status': status, 'usage': usage})+'\n')


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


@contextmanager
def serve(path, *, credential, model, effort, audit):
    path = Path(path)
    path.unlink(missing_ok=True)
    server = Server(str(path), Handler)
    path.chmod(0o600)
    server.credential, server.model, server.effort = credential, model, effort
    server.provider, server.audit = credential.provider, Path(audit)
    server.audit_lock = threading.Lock()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        path.unlink(missing_ok=True)
