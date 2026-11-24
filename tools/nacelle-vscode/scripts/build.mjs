import { build } from 'esbuild';
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';
import { resourcePath } from '../src/html.ts';

const root = fileURLToPath(new URL('..', import.meta.url));
const viewer = resolve(root, '../nacelle-view');
if (!process.env.npm_execpath) throw new Error('Run the build through npm run build.');
execFileSync(process.execPath, [process.env.npm_execpath, 'run', 'build'], {
  cwd: viewer, stdio: 'inherit', env: { ...process.env, VITE_ENABLE_PRELOADED_PROFILES: '0' },
});
await mkdir(join(root, 'dist'), { recursive: true });
await build({ entryPoints: [join(root, 'src/extension.ts')], outfile: join(root, 'dist/extension.js'),
  bundle: true, platform: 'node', format: 'cjs', target: 'node22', external: ['vscode'] });
await build({ entryPoints: [join(root, 'test/policy.test.ts')], outfile: join(root, 'dist/policy-test.cjs'),
  bundle: true, platform: 'node', format: 'cjs', target: 'node22', external: ['vscode'] });
await build({ entryPoints: [join(root, 'test/suite.ts')], outfile: join(root, 'dist/smoke-suite.cjs'),
  bundle: true, platform: 'node', format: 'cjs', target: 'node22', external: ['vscode'] });
await rm(join(root, 'dist/viewer'), { recursive: true, force: true });
await cp(join(viewer, 'dist'), join(root, 'dist/viewer'), { recursive: true });
const speedscope = join(root, 'dist/viewer/speedscope');
let frame = await readFile(join(speedscope, 'index.html'), 'utf8');
for (const match of frame.matchAll(/<link rel="stylesheet" href="([^"]+)">/g)) {
  let css = await readFile(join(speedscope, resourcePath(match[1])), 'utf8');
  for (const font of css.matchAll(/url\(["']?([^"')]+)["']?\)/g)) {
    const bytes = await readFile(join(speedscope, resourcePath(font[1])));
    css = css.replace(font[0], `url(data:font/woff2;base64,${bytes.toString('base64')})`);
  }
  frame = frame.replace(match[0], () => `<style nonce="__NACELLE_WEBVIEW_NONCE__">${css.replace(/<\/style/gi, '<\\/style')}</style>`);
}
for (const match of frame.matchAll(/<script src="([^"]+)"><\/script>/g)) {
  const script = await readFile(join(speedscope, resourcePath(match[1])), 'utf8');
  frame = frame.replace(match[0], () => `<script nonce="__NACELLE_WEBVIEW_NONCE__">${script.replace(/<\/script/gi, '<\\/script')}</script>`);
}
frame = frame.replace(/<link\b[^>]*>/g, '');
frame = frame.replace('</head>', '<style data-aphrodite nonce="__NACELLE_WEBVIEW_NONCE__"></style></head>');
await writeFile(join(speedscope, 'editor.html'), frame);
await mkdir(join(root, 'dist/debug'), { recursive: true });
for (const name of ['source-debug.lldb', 'optimized-source-debug.lldb']) {
  await cp(resolve(root, '../nacelle-dbg', name), join(root, 'dist/debug', name));
}
const notices = [];
const modules = join(viewer, 'node_modules');
for (const entry of await readdir(modules, { withFileTypes: true })) {
  if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
  const folders = entry.name.startsWith('@') ?
    (await readdir(join(modules, entry.name))).map(name => join(modules, entry.name, name)) :
    [join(modules, entry.name)];
  for (const folder of folders) {
    for (const name of await readdir(folder)) {
      if (!/^licen[sc]e(?:[.-].*)?$/i.test(name)) continue;
      notices.push(`${folder.slice(modules.length + 1)}/${name}\n${await readFile(join(folder, name), 'utf8')}`);
    }
  }
}
await writeFile(join(root, 'dist/THIRD_PARTY_NOTICES.txt'), notices.join('\n\n'));
