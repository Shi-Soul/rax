// Model calls use the existing relay; Qoder login uses a separate restricted relay.
import http from 'node:http';
import { Readable } from 'node:stream';
import { createHash } from 'node:crypto';
import { registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import { appendFileSync } from 'node:fs';

const localFetch = globalThis.fetch;
await import('../isolated_cli/transport.mjs');
const modelFetch = globalThis.fetch;
globalThis.fetch = async function(input, init = {}) {
  const request = new Request(input, init);
  // The vendor loads embedded WebAssembly from data URLs; these use no network.
  if (new URL(request.url).protocol === 'data:') return localFetch(request);
  if (new URL(request.url).origin === 'http://rax-model.invalid') return modelFetch(request);
  const body = Buffer.from(JSON.stringify({url: request.url, method: request.method,
    headers: Object.fromEntries(request.headers), bodyEncoding: 'base64',
    body: Buffer.from(await request.arrayBuffer()).toString('base64')}));
  return new Promise((resolve, reject) => {
    const outgoing = http.request({socketPath: process.env.RAX_QODER_AUTH_SOCKET,
      path: '/qoder', method: 'POST', headers: {'content-type': 'application/json',
        'content-length': body.length}}, response => {
      const empty = [204, 205, 304].includes(response.statusCode);
      if (empty) response.resume();
      resolve(new Response(empty ? null : Readable.toWeb(response), {
        status: response.statusCode, headers: Object.fromEntries(Object.entries(response.headers)
          .filter(([key]) => !['connection', 'transfer-encoding', 'content-length'].includes(key)))}));
    });
    const abort = () => outgoing.destroy(new Error('Qoder request aborted'));
    request.signal.addEventListener('abort', abort, {once: true});
    outgoing.on('close', () => request.signal.removeEventListener('abort', abort));
    outgoing.on('error', reject);
    if (request.signal.aborted) abort();
    else outgoing.end(body);
  });
};

// Qoder CLI CN 1.1.78 uses a bundled Undici client, not global fetch, for timed requests.
// Replace only its transport entry while loading; the installed package stays intact.
// A different bundle must be reviewed and tested before it can run in this sandbox.
const relayFetch = Symbol.for('rax.qoder.fetch');
globalThis[relayFetch] = globalThis.fetch;
const entry = pathToFileURL(process.env.RAX_QODER_ENTRY).href;
const bundleHash = 'bc22c27e592a065bce126d39831446a24170755b0b7f3b2288facea8dd39b275';
const original = 'function bT(A,e,t={}){return Iji(Hgl(A,e,t))}';
const inferenceEntry = 'function iqi(A,e,t,i,r){';
globalThis[Symbol.for('rax.qoder.audit')] = text => {
  if (!process.env.RAX_QODER_AUDIT) return;
  const body = JSON.parse(text);
  if (body.model_config?.display_name !== process.env.RAX_QODER_MODEL ||
      body.parameters?.reasoning_effort !== process.env.RAX_QODER_EFFORT)
    throw new Error('RAX: Qoder changed the configured model or reasoning effort');
  const count = value => {
    if (Array.isArray(value)) return value.reduce((n, v) => n + count(v), 0);
    if (!value || typeof value !== 'object') return 0;
    let images = ['image', 'image_url', 'input_image'].includes(value.type) ? 1 : 0;
    for (const [key, item] of Object.entries(value)) {
      if (key === 'inlineData' && item?.mimeType?.startsWith('image/')) images++;
      images += count(item);
    }
    return images;
  };
  const images = count(body);
  appendFileSync(process.env.RAX_QODER_AUDIT, JSON.stringify({at: Date.now()/1000,
    model: body.model_config?.display_name, wire_model: body.model_config?.key,
    effort: body.parameters.reasoning_effort, thinking: body.parameters.enable_thinking,
    image_blocks: images}) + '\n');
};
registerHooks({load(url, context, nextLoad) {
  const loaded = nextLoad(url, context);
  if (url !== entry) return loaded;
  const source = loaded.source.toString();
  if (createHash('sha256').update(source).digest('hex') !== bundleHash ||
      source.split(original).length !== 2 || source.split(inferenceEntry).length !== 2)
    throw new Error('RAX: unsupported Qoder bundle; install the pinned CLI CN 1.1.78 release');
  return {...loaded, source: source.replace(original,
    'function bT(A,e,t={}){return Iji(globalThis[Symbol.for("rax.qoder.fetch")](A,e))}')
    .replace(inferenceEntry, inferenceEntry + 'globalThis[Symbol.for("rax.qoder.audit")](e);')};
}});
