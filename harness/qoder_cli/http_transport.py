"""Stream Qoder HTTP responses, including errors sent before an upload finishes."""
import http.client
import subprocess
from tempfile import TemporaryFile
from types import SimpleNamespace


class Response(http.client.HTTPResponse):
    def __init__(self, process):
        self.process = process
        super().__init__(SimpleNamespace(makefile=lambda mode: process.stdout))

    def close(self):
        super().close()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def open_request(url, method, headers, body, timeout):
    """curl reads early replies during upload; urllib waits until sendall finishes.

    Credentials and payload use anonymous files inherited only by this process.
    Disable curl's user configuration and redirects. Host proxy settings still apply.
    Raw HTTP/1.1 output lets HTTPResponse handle interim replies and chunked streams.
    """
    if any('\r' in text or '\n' in text for pair in headers.items() for text in pair):
        raise ValueError('HTTP headers must not contain newlines')
    with TemporaryFile() as header_file, TemporaryFile() as payload:
        header_file.write(''.join(f'{key}: {value}\n' for key, value in headers.items()).encode())
        header_file.seek(0)
        command = ['curl', '-q', '--silent', '--http1.1', '--include', '--raw', '--no-buffer',
                   '--suppress-connect-headers', '--max-time', str(timeout), '--request', method,
                   '--header', f'@/proc/self/fd/{header_file.fileno()}', '--url', url]
        if body is not None:
            payload.write(body)
            payload.seek(0)
            command += ['--data-binary', f'@/proc/self/fd/{payload.fileno()}']
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL,
                                   pass_fds=(header_file.fileno(), payload.fileno()))
    response = Response(process)
    try:
        response.begin()
        return response
    except Exception:
        response.close()
        raise
