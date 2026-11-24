import assert from 'node:assert/strict';
import { test } from 'node:test';
import { exportMessage, readMessage, EXPORT_CHUNK_BYTES, MAX_EXPORT_BYTES } from '../../nacelle-view/src/host-protocol.ts';
import { ExportBuffer } from '../src/transfer.ts';
import { configurationJson, escapeAttribute, resourcePath, rewriteAssets } from '../src/html.ts';
import { checkDebugResult, commandList, debugEnvironment, debugInteger } from '../src/debug-policy.ts';

test('bounded ordered export preserves exact UTF-8 bytes across chunk boundaries', () => {
  const bytes = Buffer.from(JSON.stringify({ source: 'source \u{1f4a1} \u03bb'.repeat(12000) }));
  const transfer = new ExportBuffer(bytes.length);
  for (let offset = 0; offset < bytes.length; offset += EXPORT_CHUNK_BYTES) {
    const chunk = bytes.subarray(offset, offset + EXPORT_CHUNK_BYTES);
    assert.equal(transfer.append(offset, chunk.toString('base64')), offset + chunk.length);
  }
  assert.deepEqual(Buffer.from(transfer.finish()), bytes);
  assert.throws(() => transfer.finish(), /already finished/);
  assert.throws(() => transfer.append(bytes.length, 'YQ=='), /stale/);
});

test('invalid IPC cannot supply a destination or bypass declared bounds', () => {
  const start = { version: 1, kind: 'export-start', id: 1, size: 1, name: 'nacelle-trace.json' };
  assert.equal(exportMessage(start).kind, 'export-start');
  for (const changed of [{ ...start, path: '/outside' }, { ...start, size: MAX_EXPORT_BYTES + 1 },
    { ...start, name: '../output' }, { ...start, id: NaN }, { ...start, version: 2 }]) {
    assert.throws(() => exportMessage(changed));
  }
  const transfer = new ExportBuffer(2);
  assert.throws(() => transfer.append(1, 'YQ=='), /order/);
  assert.throws(() => transfer.append(0, 'YQ==?'), /invalid/);
  assert.throws(() => transfer.append(0, 'YWJj'), /declared size/);
  transfer.append(0, 'YQ==');
  assert.throws(() => transfer.finish(), /incomplete/);
});

test('capture reads cannot choose another file or send malformed offsets', () => {
  assert.deepEqual(readMessage({ version: 1, kind: 'read-chunk', id: 2, offset: 65536 }),
    { version: 1, kind: 'read-chunk', id: 2, offset: 65536 });
  for (const message of [
    { version: 1, kind: 'read-start', id: 1, uri: 'file:///private' },
    { version: 1, kind: 'read-chunk', id: 1, offset: -1 },
    { version: 1, kind: 'read-chunk', id: 1, offset: Infinity },
    { version: 2, kind: 'read-start', id: 1 },
  ]) assert.throws(() => readMessage(message));
});

test('webview configuration and packaged URLs cannot escape markup or resource roots', () => {
  const name = '</script><script>alert("capture")</script>';
  const encoded = configurationJson({ name });
  assert.ok(!encoded.includes('</script>'));
  assert.equal(JSON.parse(encoded).name, name);
  assert.equal(escapeAttribute('x" onload="unsafe'), 'x&quot; onload=&quot;unsafe');
  for (const path of ['../private', '/private', 'https://outside', 'assets\\outside', 'file.js?query']) {
    assert.throws(() => resourcePath(path));
  }
  const html = rewriteAssets('<script type="module" src="./assets/main.js"></script>', path => `local:${path}`, 'nonce');
  assert.ok(html.includes('nonce="nonce"'));
  assert.ok(html.includes('src="local:assets/main.js"'));
});

test('debugger replies and launch command arrays fail explicitly on unsupported shapes', () => {
  assert.deepEqual(debugEnvironment({ VALUE: 'literal=with spaces' }), ['VALUE=literal=with spaces']);
  assert.throws(() => debugEnvironment({ 'BAD=NAME': 'value' }));
  assert.throws(() => debugEnvironment({ VALUE: 42 }));
  assert.throws(() => checkDebugResult({ result: "error: use of undeclared identifier 'snapshot'\n" }), /LLDB command failed/);
  assert.throws(() => checkDebugResult({ result: "(lldb) expression\nerror: 'command' has unknown type" }), /LLDB command failed/);
  assert.equal(debugInteger({ result: '0x000f' }, 16), 15);
  for (const value of [{ result: '<optimized out>' }, { result: '-1' }, { result: '16 lanes' }, { result: '17' }]) {
    assert.throws(() => debugInteger(value, 16));
  }
  assert.throws(() => commandList('not an array', 'initCommands'));
  assert.throws(() => commandList([42], 'preRunCommands'));
});
