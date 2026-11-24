import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const tauriDir = join(root, 'src-tauri');

test('tauri.conf.json enforces strict offline CSP, no global Tauri, no assetProtocol, and offline WebView2 mode', async () => {
  const conf = JSON.parse(await readFile(join(tauriDir, 'tauri.conf.json'), 'utf8'));

  assert.equal(conf.app?.withGlobalTauri, false, 'withGlobalTauri must remain false');
  assert.deepEqual(conf.app?.windows, [], 'main window must be built in Rust with navigation/new-window guards');
  assert.equal(conf.app?.security?.assetProtocol?.enable, false, 'assetProtocol must remain disabled');
  assert.deepEqual(conf.app?.security?.assetProtocol?.scope, [], 'assetProtocol scope must be empty');
  assert.deepEqual(conf.app?.security?.capabilities, ['main-capability']);
  assert.equal(conf.bundle?.windows?.webviewInstallMode?.type, 'skip', 'WebView2 must not silently download a bootstrapper');

  const csp = conf.app?.security?.csp;
  assert.equal(typeof csp, 'string');
  assert.match(csp, /default-src 'none'/);
  assert.match(csp, /script-src 'self' 'wasm-unsafe-eval'/);
  assert.match(csp, /connect-src ipc: http:\/\/ipc\.localhost\b/);
  assert.match(csp, /object-src 'none'/);
  assert.match(csp, /base-uri 'none'/);
  assert.match(csp, /form-action 'none'/);

  // Forbid dangerous CSP tokens and remote origins.
  assert.doesNotMatch(csp, /'unsafe-eval'/);
  assert.doesNotMatch(csp, /\bstyle-src\s[^;]*'unsafe-inline'/);
  assert.doesNotMatch(csp, /https?:\/\/(?!ipc\.localhost)/i);
  assert.doesNotMatch(csp, /wss?:/i);
  assert.doesNotMatch(csp, /\*/);
});

test('capabilities/main.json grants only minimal local commands and zero remote/fs/shell/dialog plugin access', async () => {
  const cap = JSON.parse(await readFile(join(tauriDir, 'capabilities/main.json'), 'utf8'));

  assert.equal(cap.identifier, 'main-capability');
  assert.equal(cap.local, true);
  assert.equal(cap.remote, undefined, 'remote capabilities must remain disabled by default');
  assert.deepEqual(cap.windows, ['main']);

  const permissions = cap.permissions;
  assert.ok(Array.isArray(permissions) && permissions.length > 0);
  const allowed = new Set([
    'core:event:default',
    'allow-desktop-get-initial-capture',
    'allow-desktop-open-dialog',
    'allow-desktop-reload-capture',
    'allow-desktop-read-chunk',
    'allow-desktop-cancel-read',
    'allow-desktop-export-start',
    'allow-desktop-export-chunk',
    'allow-desktop-export-finish',
    'allow-desktop-export-cancel',
  ]);
  for (const perm of permissions) {
    assert.equal(typeof perm, 'string');
    assert.ok(allowed.has(perm), `unexpected desktop capability permission: ${perm}`);
    assert.doesNotMatch(perm, /^(fs|shell|http|dialog|opener|process|os|clipboard):/);
    assert.doesNotMatch(perm, /smoke/i);
  }
});

test('Cargo.toml keeps smoke-harness out of default features and adds no network crates', async () => {
  const cargo = await readFile(join(tauriDir, 'Cargo.toml'), 'utf8');
  assert.match(cargo, /^default\s*=\s*\[\s*\]/m);
  assert.match(cargo, /^smoke-harness\s*=\s*\[/m);
  assert.doesNotMatch(cargo, /tauri-plugin-(fs|shell|http|opener|updater|websocket)/);
  assert.doesNotMatch(cargo, /\b(reqwest|hyper|tokio-tungstenite)\b/);
});

test('staged dist-viewer strips browser meta CSP, loads desktop-bridge before index, and omits smoke flag by default', async () => {
  const indexPath = join(root, 'dist-viewer/index.html');
  const bridgePath = join(root, 'dist-viewer/assets/desktop-bridge.js');
  const speedscopePath = join(root, 'dist-viewer/speedscope/index.html');

  const indexHtml = await readFile(indexPath, 'utf8');
  assert.doesNotMatch(indexHtml, /http-equiv="Content-Security-Policy"/i);
  const bridgePos = indexHtml.indexOf('./assets/desktop-bridge.js');
  const mainMatch = indexHtml.match(/\.\/assets\/index-[A-Za-z0-9_-]+\.js/);
  assert.ok(bridgePos >= 0, 'desktop-bridge.js must be referenced by staged index.html');
  assert.ok(mainMatch && mainMatch.index !== undefined && bridgePos < mainMatch.index,
    'desktop-bridge.js must load before main index module');

  const speedscopeHtml = await readFile(speedscopePath, 'utf8');
  assert.match(speedscopeHtml, /<style data-aphrodite nonce="nacelle-editor"><\/style>/);

  const bridgeJs = await readFile(bridgePath, 'utf8');
  if (process.env.NACELLE_DESKTOP_SMOKE_HARNESS !== '1') {
    assert.doesNotMatch(bridgeJs, /desktop_smoke_report/,
      'production desktop bridge must not include smoke-harness command references');
  }
});
