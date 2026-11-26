import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, it } from 'vitest';

it('stages the hashed editor host chunk but refuses captures and unrecognized assets before publishing', async () => {
  const work = await mkdtemp(join(tmpdir(), 'nacelle-view-staging-'));
  const script = join(work, 'scripts', 'stage.mjs');
  const assets = join(work, 'dist', 'assets');
  const output = join(work, 'published');
  const run = () => promisify(execFile)(process.execPath, [script, output]);
  try {
    await mkdir(join(work, 'scripts'));
    await mkdir(assets, { recursive: true });
    await cp(fileURLToPath(new URL('../scripts/stage.mjs', import.meta.url)), script);
    await writeFile(join(work, 'dist', 'index.html'), '<html></html>');
    await writeFile(join(assets, 'index-abc.js'), 'viewer');
    await writeFile(join(assets, 'profile-worker-def.js'), 'worker');
    await writeFile(join(assets, 'vscode-host-ghi.js'), 'editor host');
    await writeFile(join(assets, 'desktop-host-jkl.js'), 'desktop host');
    await writeFile(join(assets, 'host-protocol-mno.js'), 'shared host protocol');
    await run();
    const staged = join(output, 'viewer', 'assets', 'vscode-host-ghi.js');
    expect(await readFile(staged, 'utf8')).toBe('editor host');
    expect(await readFile(join(output, 'viewer', 'assets', 'desktop-host-jkl.js'), 'utf8')).toBe('desktop host');
    expect(await readFile(join(output, 'viewer', 'assets', 'host-protocol-mno.js'), 'utf8')).toBe('shared host protocol');

    await writeFile(join(assets, 'vscode-host-ghi.js'), 'replacement');
    for (const name of ['private.nacelle-profile', 'vscode-host.js', 'desktop-host.js', 'desktop-bridge.js']) {
      const unexpected = join(assets, name);
      await writeFile(unexpected, 'not publishable');
      await expect(run()).rejects.toThrow(`Unexpected viewer asset: ${name}`);
      expect(await readFile(staged, 'utf8')).toBe('editor host');
      await rm(unexpected);
    }
  } finally {
    await rm(work, { recursive: true, force: true });
  }
});
