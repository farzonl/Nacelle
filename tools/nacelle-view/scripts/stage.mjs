import { cp, mkdir, readdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';

const output = process.argv[2];
if (!output) throw new Error('Usage: npm run stage -- <Pages staging directory>');
const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const target = join(resolve(output), 'viewer');
// Publish only the build allowlist, never capture files or source directories.
const files = await readdir(dist);
if (!files.includes('index.html') || files.some(file => !['index.html', 'assets'].includes(file))) {
  throw new Error('Unexpected viewer build contents; refusing to publish.');
}
for (const name of await readdir(join(dist, 'assets'))) {
  if (!/^index-[A-Za-z0-9_-]+\.(js|css)$/.test(name)) {
    throw new Error(`Unexpected viewer asset: ${name}`);
  }
}
await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });
for (const file of files) await cp(join(dist, file), join(target, file), { recursive: true });
console.log(`Staged bundled offline profile viewer at ${target}`);
