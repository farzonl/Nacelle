import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { FileProfileDataSource } from '../src/file-source';
import { loadProfile } from '../src/profile';
import { artifactFixture } from './artifact-fixtures';
import type { ArtifactMetadata } from '../src/artifacts';

let corpusIndex = 0;
function nativeDecision(bytes: ArrayBuffer, accepted: boolean): void {
  if (process.env.NACELLE_FUZZ_CORPUS)
    writeFileSync(join(process.env.NACELLE_FUZZ_CORPUS, `artifacts-${corpusIndex++}.nacelle-profile`), new Uint8Array(bytes));
  if (!process.env.NACELLE_NATIVE_READER) return;
  const directory = mkdtempSync(join(tmpdir(), 'nacelle-artifacts-'));
  try {
    const path = join(directory, 'input.nacelle-profile');
    writeFileSync(path, new Uint8Array(bytes));
    const child = spawnSync(process.env.NACELLE_NATIVE_READER, [path], { encoding: 'utf8', timeout: 10_000 });
    expect(child.error).toBeUndefined();
    expect(child.signal).toBeNull();
    expect(child.status, child.stderr).toBe(accepted ? 0 : 1);
    if (!accepted) expect(child.stderr).not.toBe('');
  } finally {
    rmSync(directory, { recursive: true });
  }
}

const load = async (change?: (metadata: ArtifactMetadata) => void, options: Record<string, unknown> = {}) => {
  const bytes = await artifactFixture(change, options);
  nativeDecision(bytes, !change && !Object.keys(options).length);
  return loadProfile(FileProfileDataSource.fromBuffer(bytes));
};

describe('typed artifact payload validation', () => {
  it('resolves exact old/new versions, partial source mappings and object-relative instructions', async () => {
    const profile = await load();
    expect(profile.artifacts?.selections[0].nativeAvailability).toBe('tier1-unavailable');
    expect(profile.artifacts?.selections[1].nativeAvailability).toBe('available');
    expect(profile.artifacts?.payloads[1].mappings[1].sourceLine).toBe(2);
    expect(profile.trace.traceEvents.map(event => event.args.codeVersionId)).toEqual([1, 2]);
  });
  it.each([
    (metadata: ArtifactMetadata) => { Reflect.set(metadata, 'schemaVersion', 2); },
    (metadata: ArtifactMetadata) => { Reflect.set(metadata.payloads[0], 'privatePath', '/private/source.hlsl'); },
    (metadata: ArtifactMetadata) => { metadata.payloads[0].label = '/private/source.hlsl'; },
    (metadata: ArtifactMetadata) => { metadata.payloads[0].text += 'tampered'; },
    (metadata: ArtifactMetadata) => { metadata.payloads[1].mappings[0].sourceLine = 999; },
    (metadata: ArtifactMetadata) => { metadata.payloads[1].mappings[0].source = 'missing'; },
    (metadata: ArtifactMetadata) => { metadata.selections[0].payloadIds.push(metadata.payloads[2].id); },
    (metadata: ArtifactMetadata) => { metadata.selections[1].codeVersionId = 1; },
    (metadata: ArtifactMetadata) => { metadata.payloads[2].mappings[0].offset = 256 * 1024 * 1024; },
    (metadata: ArtifactMetadata) => { metadata.selections[1].nativeAvailability = 'not-retained'; },
  ])('rejects malformed artifacts without returning a partial profile', async change => {
    await expect(load(change)).rejects.toThrow();
  });
  it.each([{ embeddedSources: false }, { sourceBytes: 1 }, { artifactBytes: 1 }, { sourceBytes: 0 }])(
    'rejects source consent and declared budget violations in both readers: %j', async options => {
      await expect(load(undefined, options)).rejects.toThrow();
    },
  );
});
