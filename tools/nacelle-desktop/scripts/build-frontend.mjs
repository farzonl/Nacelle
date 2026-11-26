import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const viewer = resolve(root, '../nacelle-view');
const outDir = join(root, 'dist-viewer');
const smokeHarness = process.env.NACELLE_DESKTOP_SMOKE_HARNESS === '1';

const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const npmExec = process.env.npm_execpath;
if (npmExec) {
  execFileSync(process.execPath, [npmExec, 'run', 'build'], {
    cwd: viewer,
    stdio: 'inherit',
    env: { ...process.env, VITE_ENABLE_PRELOADED_PROFILES: '0' },
  });
} else {
  execFileSync(npmCmd, ['run', 'build'], {
    cwd: viewer,
    stdio: 'inherit',
    shell: process.platform === 'win32',
    env: { ...process.env, VITE_ENABLE_PRELOADED_PROFILES: '0' },
  });
}

await rm(outDir, { recursive: true, force: true });
await cp(join(viewer, 'dist'), outDir, { recursive: true });

await build({
  entryPoints: [join(root, 'src/bridge.ts')],
  outfile: join(outDir, 'assets/desktop-bridge.js'),
  bundle: true,
  minifySyntax: true,
  platform: 'browser',
  format: 'esm',
  target: 'es2022',
  define: {
    __NACELLE_SMOKE_HARNESS__: smokeHarness ? 'true' : 'false',
  },
});

const indexPath = join(outDir, 'index.html');
let indexHtml = await readFile(indexPath, 'utf8');
// Remove browser-only <meta> CSP so Tauri's authoritative HTTP CSP header controls IPC and assets.
indexHtml = indexHtml.replace(/<meta\s+http-equiv="Content-Security-Policy"[^>]*>\s*/i, '');
const moduleScriptPattern = /<script\s+type="module"\s+[^>]*src="\.\/assets\/index-[^"]+\.js"[^>]*><\/script>/;
if (!moduleScriptPattern.test(indexHtml)) {
  throw new Error('Could not locate main viewer module script in nacelle-view/dist/index.html.');
}
indexHtml = indexHtml.replace(
  moduleScriptPattern,
  match => `<script type="module" src="./assets/desktop-bridge.js"></script>\n    ${match}`,
);
await writeFile(indexPath, indexHtml, 'utf8');

// Inject nonce-bearing Aphrodite placeholder into bundled Speedscope so strict style-src works without unsafe-inline.
const speedscopeIndexPath = join(outDir, 'speedscope/index.html');
let speedscopeHtml = await readFile(speedscopeIndexPath, 'utf8');
if (!speedscopeHtml.includes('data-aphrodite')) {
  speedscopeHtml = speedscopeHtml.replace(
    '</head>',
    '  <style data-aphrodite nonce="nacelle-editor"></style>\n  </head>',
  );
  await writeFile(speedscopeIndexPath, speedscopeHtml, 'utf8');
}

// Collect third-party JS notices from nacelle-view and nacelle-desktop.
async function collectNotices(modulesDir, prefix, notices) {
  let entries;
  try {
    entries = await readdir(modulesDir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const folders = entry.name.startsWith('@')
      ? (await readdir(join(modulesDir, entry.name))).map(name => join(modulesDir, entry.name, name))
      : [join(modulesDir, entry.name)];
    for (const folder of folders) {
      let names;
      try {
        names = await readdir(folder);
      } catch {
        continue;
      }
      for (const name of names) {
        if (!/^licen[sc]e(?:[.-].*)?$/i.test(name)) continue;
        const pkg = folder.slice(modulesDir.length + 1).replace(/\\/g, '/');
        notices.push(`${prefix}${pkg}/${name}\n${await readFile(join(folder, name), 'utf8')}`);
      }
    }
  }
}

const notices = [];
await collectNotices(join(viewer, 'node_modules'), 'nacelle-view/', notices);
await collectNotices(join(root, 'node_modules'), 'nacelle-desktop/', notices);
await mkdir(outDir, { recursive: true });
await writeFile(join(outDir, 'THIRD_PARTY_NOTICES.txt'), notices.join('\n\n'), 'utf8');

console.log(`Staged Nacelle desktop frontend at ${outDir}${smokeHarness ? ' (smoke harness)' : ''}`);
