import { cp, mkdir, readFile, readdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';

const args = process.argv.slice(2);
let output = '';
let profilesDir = '';

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--profiles-dir') {
    profilesDir = args[++i] ?? '';
    if (!profilesDir) throw new Error('--profiles-dir requires a directory path.');
  } else if (!output) {
    output = args[i];
  } else {
    throw new Error(`Unexpected argument: ${args[i]}`);
  }
}

if (!output) {
  throw new Error('Usage: npm run stage -- <Pages staging directory> [--profiles-dir <dir>]');
}

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const target = join(resolve(output), 'viewer');
// Publish only the build allowlist, never arbitrary capture files or source directories.
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

if (profilesDir) {
  const srcProfiles = resolve(profilesDir);
  const manifestPath = join(srcProfiles, 'manifest.json');
  const raw = JSON.parse(await readFile(manifestPath, 'utf-8'));
  if (!raw || typeof raw !== 'object' || raw.schemaVersion !== 1 || !Array.isArray(raw.profiles)) {
    throw new Error(`Invalid preloaded profiles manifest at ${manifestPath}`);
  }
  const destProfiles = join(target, 'profiles');
  await mkdir(destProfiles, { recursive: true });
  await cp(manifestPath, join(destProfiles, 'manifest.json'));
  let stagedCount = 0;
  for (const entry of raw.profiles) {
    if (!entry || typeof entry.file !== 'string' || !/^[A-Za-z0-9_.-]+\.nacelle-profile$/.test(entry.file)) {
      throw new Error(`Unsafe preloaded profile filename: ${String(entry?.file)}`);
    }
    await cp(join(srcProfiles, entry.file), join(destProfiles, entry.file));
    stagedCount++;
  }
  console.log(`Staged bundled profile viewer with ${stagedCount} preloaded benchmark profiles at ${target}`);
} else {
  console.log(`Staged bundled offline profile viewer at ${target}`);
}
