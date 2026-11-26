import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const nativeProfile = process.env.NACELLE_NATIVE_PROFILE;
const nativeTrace = process.env.NACELLE_NATIVE_TRACE;
const nativeReader = process.env.NACELLE_NATIVE_READER;

if (!nativeProfile) {
  throw new Error('Set NACELLE_NATIVE_PROFILE to a real .nacelle-profile capture for desktop smoke verification.');
}

const work = await mkdtemp(join(tmpdir(), 'nacelle-desktop-smoke-'));
let expectedTracePath = nativeTrace;

try {
  if (!expectedTracePath) {
    if (!nativeReader) {
      throw new Error('Set either NACELLE_NATIVE_TRACE or NACELLE_NATIVE_READER to compare exported trace bytes.');
    }
    expectedTracePath = join(work, 'expected-trace.json');
    execFileSync(nativeReader, [nativeProfile, '--trace', expectedTracePath], { stdio: 'inherit' });
  }

  // 1. Stage frontend with smoke-harness bridge enabled.
  const smokeEnv = { ...process.env, NACELLE_DESKTOP_SMOKE_HARNESS: '1' };
  execFileSync(process.execPath, [join(root, 'scripts/build-frontend.mjs')], {
    cwd: root,
    stdio: 'inherit',
    env: smokeEnv,
  });

  // 2. Build the Tauri binary with --features smoke-harness (never used in production packaging).
  execFileSync(
    'cargo',
    ['build', '--manifest-path', join(root, 'src-tauri/Cargo.toml'), '--features', 'smoke-harness'],
    { cwd: root, stdio: 'inherit', env: smokeEnv },
  );

  const exe = process.platform === 'win32' ? 'nacelle-desktop.exe' : 'nacelle-desktop';
  const binary = join(root, 'src-tauri/target/debug', exe);

  // 3. Launch the real native WebView binary with the real capture as CLI argument.
  const childEnv = {
    ...smokeEnv,
    NACELLE_DESKTOP_SMOKE_DIR: work,
  };
  if (process.platform === 'linux' && !childEnv.WEBKIT_DISABLE_DMABUF_RENDERER) {
    childEnv.WEBKIT_DISABLE_DMABUF_RENDERER = '1';
  }
  const child = spawn(binary, [resolve(nativeProfile)], {
    stdio: 'inherit',
    env: childEnv,
  });

  const exitCode = await new Promise((resolveExit, rejectExit) => {
    const timer = setTimeout(() => {
      child.kill();
      rejectExit(new Error('Desktop WebView smoke timed out after 60 seconds.'));
    }, 60000);
    child.on('error', err => {
      clearTimeout(timer);
      rejectExit(err);
    });
    child.on('exit', code => {
      clearTimeout(timer);
      resolveExit(code);
    });
  });

  assert.equal(exitCode, 0, `Desktop viewer exited with unexpected code ${exitCode}`);

  // 4. Verify real WebView rendering, Speedscope canvas, blocked external navigation, and exports.
  const summary = JSON.parse(await readFile(join(work, 'smoke-result.json'), 'utf8'));
  assert.ok(['complete', 'incomplete'].includes(summary.loaded?.state),
    `Expected complete/incomplete capture state, got: ${JSON.stringify(summary.loaded)}`);
  assert.ok(summary.loaded.spans > 0, `Expected rendered timing spans, got ${summary.loaded.spans}`);
  assert.equal(summary.loaded.speedscopeReady, true, 'Embedded Speedscope canvas did not become ready');
  assert.ok(summary.blockedNavigationCount >= 1, 'Expected on_navigation to block external https:// navigation');
  assert.match(summary.currentUrl, /^(tauri:\/\/localhost|https?:\/\/tauri\.localhost)(\/|$)/,
    `Unexpected WebView URL after blocked navigation: ${summary.currentUrl}`);
  assert.equal(summary.traceExported, true, 'Chrome trace export did not commit');
  assert.equal(summary.speedscopeExported, true, 'Filtered Speedscope export did not commit');

  // 5. Verify exact byte equality of exported Chrome trace against nacelle-dbg.
  const [actualTrace, expectedTrace] = await Promise.all([
    readFile(join(work, 'nacelle-trace.json')),
    readFile(expectedTracePath),
  ]);
  assert.ok(actualTrace.equals(expectedTrace), 'Exported Chrome trace bytes diverged from native CLI export');

  // 6. Verify exported Speedscope profile JSON and zero leftover staging files.
  const speedscopeJson = JSON.parse(await readFile(join(work, 'nacelle-speedscope.json'), 'utf8'));
  assert.match(String(speedscopeJson.exporter ?? ''), /nacelle/i);
  assert.ok(Array.isArray(speedscopeJson.profiles) && speedscopeJson.profiles.length > 0);

  const files = await readdir(work);
  const stagingLeftovers = files.filter(name => name.startsWith('.nacelle-export-'));
  assert.deepEqual(stagingLeftovers, [], 'Export staging temporary files were left behind');

  console.log(`Real native WebView smoke passed (${process.platform}): open, decode, Speedscope canvas, blocked navigation (${summary.blockedNavigationCount}), exact trace export, and Speedscope export verified.`);
} finally {
  // Restore clean non-smoke frontend staging so subsequent policy/package runs stay clean.
  try {
    execFileSync(process.execPath, [join(root, 'scripts/build-frontend.mjs')], {
      cwd: root,
      stdio: 'ignore',
      env: { ...process.env, NACELLE_DESKTOP_SMOKE_HARNESS: '0' },
    });
  } catch {
    // Best-effort cleanup.
  }
  await rm(work, { recursive: true, force: true });
}
