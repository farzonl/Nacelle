import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const bundleTarget = process.platform === 'win32'
  ? 'nsis'
  : process.platform === 'darwin'
    ? 'app'
    : 'deb';

if (process.env.NACELLE_DESKTOP_SMOKE_HARNESS === '1') {
  throw new Error('Refusing to package a production desktop bundle with NACELLE_DESKTOP_SMOKE_HARNESS=1.');
}

const tauriBin = process.platform === 'win32'
  ? 'node_modules\\.bin\\tauri.cmd'
  : 'node_modules/.bin/tauri';

execFileSync(tauriBin, ['build', '--bundles', bundleTarget], {
  cwd: root,
  stdio: 'inherit',
  shell: process.platform === 'win32',
});
