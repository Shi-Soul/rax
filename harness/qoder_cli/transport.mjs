// Model calls use the existing relay; Qoder login uses a separate restricted relay.
import '../isolated_cli/transport.mjs';
import http from 'node:http';
import { Readable } from 'node:stream';

const modelFetch = globalThis.fetch;
globalThis.fetch = async function(input, init = {}) {
  const request = new Request(input, init);
  if (new URL(request.url).origin === 'http://rax-model.invalid') return modelFetch(request);
  const body = Buffer.from(JSON.stringify({url: request.url, method: request.method,
    headers: Object.fromEntries(request.headers), body: await request.text()}));
  return new Promise((resolve, reject) => {
    const outgoing = http.request({socketPath: process.env.RAX_QODER_AUTH_SOCKET,
      path: '/qoder', method: 'POST', headers: {'content-type': 'application/json',
        'content-length': body.length}}, response => {
      const empty = [204, 205, 304].includes(response.statusCode);
      if (empty) response.resume();
      resolve(new Response(empty ? null : Readable.toWeb(response), {
        status: response.statusCode, headers: {'content-type': response.headers['content-type']}}));
    });
    const abort = () => outgoing.destroy(new Error('Qoder request aborted'));
    request.signal.addEventListener('abort', abort, {once: true});
    outgoing.on('close', () => request.signal.removeEventListener('abort', abort));
    outgoing.on('error', reject);
    if (request.signal.aborted) abort();
    else outgoing.end(body);
  });
};
