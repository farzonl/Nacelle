import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { fuzzProfileBytes, profileCorpus } from './profile-corpus';

it.each(profileCorpus())('validates shared malformed corpus: $name', async ({ bytes, accepted }) => {
  expect((await fuzzProfileBytes(bytes)).accepted).toBe(accepted);
});

it.runIf(Boolean(process.env.NACELLE_NATIVE_READER))('matches native reader decisions on the same corpus bytes', () => {
  const directory = mkdtempSync(join(tmpdir(), 'nacelle-parser-parity-'));
  try {
    for (const { name, bytes, accepted } of profileCorpus()) {
      const path = join(directory, 'input.nacelle-profile');
      writeFileSync(path, new Uint8Array(bytes));
      const child = spawnSync(process.env.NACELLE_NATIVE_READER!, [path], { encoding: 'utf8', timeout: 10_000 });
      expect(child.error, name).toBeUndefined();
      expect(child.signal, name).toBeNull();
      expect(child.status, `${name}: ${child.stderr}`).toBe(accepted ? (name === 'incomplete' ? 2 : 0) : 1);
      if (!accepted) expect(child.stderr, name).not.toBe('');
    }
  } finally {
    rmSync(directory, { recursive: true });
  }
});

it.runIf(Boolean(process.env.NACELLE_FUZZ_CORPUS))('exports sanitized seeds for the native fuzz entry', () => {
  for (const [index, entry] of profileCorpus().entries())
    writeFileSync(join(process.env.NACELLE_FUZZ_CORPUS!, `profile-${index}.nacelle-profile`), new Uint8Array(entry.bytes));
});

it('fuzzes bounded mutations of valid and malformed profile seeds', async () => {
  const rounds = Number(process.env.NACELLE_FUZZ_ROUNDS ?? 256);
  expect(Number.isInteger(rounds) && rounds > 0 && rounds <= 100_000).toBe(true);
  const seeds = profileCorpus();
  let state = 0x31415926;
  const next = () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0);
  for (let round = 0; round < rounds; round++) {
    const bytes = seeds[next() % seeds.length].bytes.slice(0);
    const data = new Uint8Array(bytes);
    for (let edit = 0; edit < 1 + next() % 4; edit++) data[next() % data.length] ^= 1 << (next() % 8);
    const result = await fuzzProfileBytes(round % 3 ? bytes : bytes.slice(0, next() % bytes.byteLength));
    if (!result.accepted) expect(result.error).not.toBe('');
  }
});
