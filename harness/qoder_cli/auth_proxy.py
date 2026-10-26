"""Permit Qoder login and catalog requests without exposing the operator's PAT."""
from contextlib import contextmanager
import base64
from http.server import BaseHTTPRequestHandler
import json
import re
import threading
from urllib.parse import urlsplit

from harness.claude_cli.login import private_text
from harness.isolated_cli.relay import Server, open_request

HOSTS = {'openapi.qoder.com.cn', 'gateway.qoder.com.cn'}
GET_PATHS = {'/api/v1/userinfo', '/api/v3/user/status', '/api/v2/user/plan',
             '/api/v2/model/list', '/algo/api/v2/byok/config', '/api/v2/config/getDataPolicy',
             '/algo/api/v2/model/list', '/algo/api/v2/config/getDataPolicy',
             '/api/v3/service/region/endpoints', '/api/v5/service/region/endpoints',
             '/algo/api/v3/service/region/endpoints', '/algo/api/v5/service/region/endpoints'}
POST_PATHS = {'/api/v1/jobToken/exchange', '/api/v1/jobToken/refresh',
              '/api/v1/deviceToken/refresh', '/api/v1/qcs/config/resolve'}
MODEL_PATHS = {'/model/v1/chat/completions', '/algo/api/v2/service/pro/sse/agent_chat_generation'}


def target(url, method, *, inference=False):
    parsed = urlsplit(url)
    allowed = (method == 'GET' and (parsed.path in GET_PATHS or
               re.fullmatch(r'/api/v1/organizations/[A-Za-z0-9_-]+/tags', parsed.path) or
               re.fullmatch(r'/sash/api/v1/organizations/[A-Za-z0-9_-]+/'
                            r'(plugins/effective|mcp-governance/policy|policies)', parsed.path))) or (
               method == 'POST' and parsed.path in POST_PATHS and parsed.hostname == 'openapi.qoder.com.cn') or (
               inference and method == 'POST' and parsed.path in MODEL_PATHS and parsed.hostname == 'gateway.qoder.com.cn')
    if (parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.fragment or not allowed):
        raise ValueError('Only Qoder authentication and account metadata endpoints are allowed')
    return parsed


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        upstream, response_started = None, False
        try:
            if self.path != '/qoder':
                raise ValueError('Unexpected proxy path')
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 128*1024**2:
                raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(size))
            url = target(request['url'], request['method'], inference=self.server.inference)
            model_request = url.path in MODEL_PATHS
            if not model_request and size > 1024**2:
                raise ValueError('Account request exceeds the limit')
            body = (base64.b64decode(request['body'], validate=True) if request.get('bodyEncoding') == 'base64'
                    else request['body'].encode())
            if url.path == '/api/v1/jobToken/exchange':
                if self.server.token_file is None:
                    raise ValueError('This session uses browser authentication')
                body = json.loads(body)
                if body['personal_token'] != 'rax-session-transport':
                    raise ValueError('Expected the session credential placeholder')
                body['personal_token'] = private_text(self.server.token_file).strip()
                body = json.dumps(body).encode()
            headers = {k: v for k, v in request['headers'].items()
                       if k.lower() not in ('host', 'content-length', 'connection', 'accept-encoding')}
            upstream = open_request(request['url'], request['method'], headers,
                                    body if body else None, 900 if model_request else 60)
            response = upstream
            if model_request:
                self.send_response(response.status)
                for name, value in response.headers.items():
                    if name.lower() not in ('connection', 'transfer-encoding', 'content-length'):
                        self.send_header(name, value)
                self.end_headers()
                response_started = True
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return
            data = response.read(8*1024**2+1)
            if len(data) > 8*1024**2:
                raise ValueError('Qoder response exceeds the limit')
            self.send_response(response.status)
            self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            # Signed headers, credentials and upstream response bodies stay out of logs.
            if not response_started:
                self.send_error(502, 'Qoder authentication transport failed: '+type(error).__name__)
        finally:
            if upstream:
                upstream.close()


@contextmanager
def serve(path, token_file, *, inference=False):
    path.unlink(missing_ok=True)
    server = Server(str(path), Handler)
    path.chmod(0o600)
    server.token_file = token_file
    server.inference = inference
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        path.unlink(missing_ok=True)
