import { _electron } from '@playwright/test';
import { downloadAndUnzipVSCode } from '@vscode/test-electron';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const repository = resolve(root, '../..');
const untrusted = process.env.NACELLE_EXTENSION_SMOKE_UNTRUSTED === '1';
const sourceFile = process.env.NACELLE_EXTENSION_SOURCE_FILE ?? join(repository, 'tests/shaders/source_debug.hlsl');
const program = process.env.NACELLE_EXTENSION_DEBUG_PROGRAM ??
  join(repository, 'build/runtime-check/tests/profile_runtime_static_test');
const cli = process.env.NACELLE_EXTENSION_PROFILE_READER ??
  join(repository, 'build/runtime-check/tools/nacelle-dbg/nacelle-dbg');
const executablePath = await downloadAndUnzipVSCode({
  version: '1.106.3', cachePath: join(root, '.vscode-test'),
});
const work = await mkdtemp(join(tmpdir(), 'nacelle-vscode-smoke-'));
const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith('NACELLE_')));
delete env.ELECTRON_RUN_AS_NODE;
env.NACELLE_DISABLE_TIER2 = '1';
env.NACELLE_CACHE_DIR = join(work, 'cache');
env.NACELLE_EXTENSION_SMOKE_DIR = work;
env.NACELLE_EXTENSION_DEBUG_PROGRAM = program;
env.NACELLE_EXTENSION_SMOKE_UNTRUSTED = untrusted ? '1' : '0';
env.NACELLE_EXTENSION_SOURCE_FILE = sourceFile;
let app;
let child;
let closed = false;
try {
  execFileSync(program, [join(work, 'capture.nacelle-profile'), 'tier1'], { env, stdio: 'inherit' });
  execFileSync(cli, [join(work, 'capture.nacelle-profile'), '--trace', join(work, 'expected-trace.json')], { env, stdio: 'inherit' });
  await mkdir(join(work, 'extensions'));
  await mkdir(join(work, 'user/User'), { recursive: true });
  await writeFile(join(work, 'user/User/settings.json'), JSON.stringify({
    'security.workspace.trust.startupPrompt': 'never', 'window.dialogStyle': 'custom',
    'telemetry.telemetryLevel': 'off', 'extensions.autoCheckUpdates': false,
    'extensions.autoUpdate': false, 'update.mode': 'none',
  }));
  app = await _electron.launch({
    executablePath, env, timeout: 120000,
    args: ['--no-sandbox', '--disable-gpu', '--skip-welcome', '--skip-release-notes',
      ...(untrusted ? [] : ['--disable-workspace-trust']), '--user-data-dir', join(work, 'user'),
      '--extensions-dir', join(work, 'extensions'), `--extensionDevelopmentPath=${root}`,
      `--extensionTestsPath=${join(root, 'dist/smoke-suite.cjs')}`, work],
  });
  child = app.process();
  const page = await app.firstWindow();
  app.on('close', () => { closed = true; });
  page.on('console', message => {
    if (message.type() === 'error') console.error('VS Code renderer:', message.text());
  });
  page.on('response', response => {
    if (response.status() >= 400) console.error(`Webview resource ${response.status()}: ${response.url()}`);
  });
  const deadline = Date.now() + 240000;
  let saved = false;
  while (Date.now() < deadline) {
    const phase = await readFile(join(work, 'stage.json'), 'utf8').then(JSON.parse).catch(error => {
      if (error.code === 'ENOENT') return undefined;
      throw error;
    });
    if (phase?.stage === 'complete') {
      console.log(untrusted ?
        'Real restricted-mode render/Speedscope/denied export and HLSL controls/teardown: PASS' :
        'Real Extension Host render/Speedscope/export/teardown and native/pinned/optimized shader DAP steps, source and lanes: PASS');
      break;
    }
    if (closed) throw new Error('VS Code exited before completing the extension smoke.');
    if (phase?.stage === 'render') {
      for (const frame of page.frames()) {
        if (!frame.url().startsWith('about:srcdoc')) continue;
        const text = await frame.locator('body').innerText();
        if (/Time Order/.test(text) && /Left Heavy/.test(text) &&
            /capture\.nacelle-profile/.test(text) && await frame.locator('canvas').count() > 0) {
          await writeFile(join(work, 'render-verified'), 'Real Speedscope profile and canvas loaded.');
        }
      }
    }
    if (phase?.stage === 'export' && !saved) {
      const input = page.locator('.quick-input-widget input').first();
      await input.waitFor({ state: 'visible', timeout: 30000 });
      await input.fill(join(work, 'nacelle-trace.json'));
      await input.press('Enter');
      saved = true;
    }
    if (phase?.stage === 'lane') {
      const input = page.locator('.quick-input-widget input').first();
      if (await input.isVisible()) await input.press('Enter');
    }
    if (phase?.stage === 'source') {
      const input = page.locator('.quick-input-widget input').first();
      if (await input.isVisible()) {
        await input.fill(sourceFile);
        await input.press('Enter');
      }
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  const final = JSON.parse(await readFile(join(work, 'stage.json'), 'utf8'));
  if (final.stage !== 'complete') throw new Error('Extension smoke did not complete its real acceptance flow.');
} catch (error) {
  try { console.error('Real DAP acceptance transcript:', (await readFile(join(work, 'dap.jsonl'), 'utf8')).slice(-32768)); }
  catch (readError) { if (readError.code !== 'ENOENT') console.error('Unable to read smoke transcript:', readError); }
  throw error;
} finally {
  if (app && !closed && child?.exitCode === null) await app.close();
  await rm(work, { recursive: true, force: true });
}
