"""Permit Qoder login and catalog requests without exposing the operator's PAT."""
from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler
import json
import re
import threading
from urllib.parse import urlsplit

from harness.claude_cli.login import private_text
from harness.isolated_cli.relay import Server

HOSTS = {'openapi.qoder.sh', 'center.qoder.sh', 'api1.qoder.sh',
         'api2.qoder.sh', 'api3.qoder.sh', 'api2-v2.qoder.sh'}
GET_PATHS = {'/api/v1/userinfo', '/api/v3/user/status', '/api/v2/user/plan',
             '/api/v2/model/list', '/algo/api/v2/byok/config', '/api/v2/config/getDataPolicy'}
POST_PATHS = {'/api/v1/jobToken/exchange', '/api/v1/jobToken/refresh'}


def target(url, method):
    parsed = urlsplit(url)
    allowed = (method == 'GET' and (parsed.path in GET_PATHS or
               re.fullmatch(r'/api/v1/organizations/[A-Za-z0-9_-]+/tags', parsed.path))) or (
               method == 'POST' and parsed.path in POST_PATHS and parsed.hostname == 'openapi.qoder.sh')
    if (parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.fragment or not allowed):
        raise ValueError('Only Qoder authentication and catalog endpoints are allowed')
    return parsed


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        upstream = None
        try:
            if self.path != '/qoder':
                raise ValueError('Unexpected proxy path')
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 1024**2:
                raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(size))
            url = target(request['url'], request['method'])
            body = request['body']
            if url.path == '/api/v1/jobToken/exchange':
                body = json.loads(body)
                if body['personal_token'] != 'rax-session-transport':
                    raise ValueError('Expected the session credential placeholder')
                body['personal_token'] = private_text(self.server.token_file).strip()
                body = json.dumps(body)
            headers = {k: v for k, v in request['headers'].items()
                       if k.lower() not in ('host', 'content-length', 'connection', 'accept-encoding')}
            upstream = http.client.HTTPSConnection(url.hostname, timeout=60)
            upstream.request(request['method'], url.path+('?' + url.query if url.query else ''),
                             body=body.encode() if body else None, headers=headers)
            response = upstream.getresponse()
            data = response.read(8*1024**2+1)
            if len(data) > 8*1024**2:
                raise ValueError('Qoder response exceeds the limit')
            self.send_response(response.status)
            self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            # Signed headers, credentials and upstream response bodies stay out of logs.
            self.send_error(502, 'Qoder authentication transport failed: '+type(error).__name__)
        finally:
            if upstream:
                upstream.close()


@contextmanager
def serve(path, token_file):
    path.unlink(missing_ok=True)
    server = Server(str(path), Handler)
    path.chmod(0o600)
    server.token_file = token_file
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        path.unlink(missing_ok=True)
