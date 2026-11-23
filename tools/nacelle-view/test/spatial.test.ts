import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { FileProfileDataSource } from '../src/file-source';
import { loadProfile } from '../src/profile';
import type { SpatialMetadata } from '../src/spatial';
import { spatialFixture } from './spatial-fixtures';

let seed = 0;
async function check(change?: (metadata: SpatialMetadata) => void, accepted = !change,
  nativeStatus = 0, stage: 'compute' | 'pixel' = 'compute') {
  const bytes = await spatialFixture(change, stage);
  if (process.env.NACELLE_FUZZ_CORPUS)
    writeFileSync(join(process.env.NACELLE_FUZZ_CORPUS, `spatial-${seed++}.nacelle-profile`), new Uint8Array(bytes));
  if (process.env.NACELLE_NATIVE_READER) {
    const directory = mkdtempSync(join(tmpdir(), 'nacelle-spatial-'));
    try {
      const path = join(directory, 'input.nacelle-profile');
      writeFileSync(path, new Uint8Array(bytes));
      const child = spawnSync(process.env.NACELLE_NATIVE_READER, [path], { encoding: 'utf8', timeout: 10_000 });
      expect(child.error).toBeUndefined();
      expect(child.signal).toBeNull();
      expect(child.status, child.stderr).toBe(accepted ? nativeStatus : 1);
    } finally {
      rmSync(directory, { recursive: true });
    }
  }
  const result = await loadProfile(FileProfileDataSource.fromBuffer(bytes)).then(
    profile => ({ profile, error: undefined }),
    error => ({ profile: undefined, error }),
  );
  expect(!result.error).toBe(accepted);
  if (!accepted) expect(result.error).toBeInstanceOf(Error);
  return result.profile;
}

it('loads sparse sampled coordinates with observed zero, exact version identity and no inferred lane precision', async () => {
  const profile = await check();
  if (!profile) throw new Error('Valid spatial fixture was rejected');
  expect(profile.spatial?.operations[0].cells.map(cell => cell.xyz)).toEqual([[0, 0, 0], [2, 0, 0]]);
  expect(profile.spatial?.operations[0].cells[0].wallNs).toBe(0);
  expect(profile.spatial?.operations[0].cells[0].coveredLanes).toEqual([]);
  expect(profile.spatial?.skipped).toBe(4);
});

it('loads incomplete observations without conflating spatial drops with event drops', async () => {
  const profile = await check(metadata => {
    metadata.complete = false;
    metadata.dropped = 1;
    const operation = metadata.operations[0];
    operation.complete = false;
    operation.dropped = 1;
    operation.missingStatus = 'dropped-or-unsampled';
    for (const cell of operation.cells) cell.status = 'partial';
  }, true, 2);
  if (!profile) throw new Error('Incomplete spatial fixture was rejected');
  expect(profile.manifest.complete).toBe(false);
  expect(profile.manifest.droppedEvents).toBe(0);
  expect(profile.spatial?.dropped).toBe(1);
});

it('preserves explicit empty and unsupported operations rather than inventing zero-valued cells', async () => {
  for (const unsupported of [false, true]) {
    const profile = await check(metadata => {
      metadata.recorded = metadata.skipped = 0;
      if (unsupported) {
        const operation = metadata.operations[0];
        operation.recorded = operation.skipped = 0;
        operation.cells = [];
        operation.availability = 'dispatch-wide-entry-unavailable';
        operation.missingStatus = 'unavailable';
      } else metadata.operations = [];
    }, true);
    expect(profile?.spatial?.recorded).toBe(0);
    expect(profile?.spatial?.operations[0]?.missingStatus).toBe(unsupported ? 'unavailable' : undefined);
  }
});

it('validates every pixel slot against SIMD batches, not merely matching aggregate lane totals', async () => {
  expect((await check(undefined, true, 0, 'pixel'))?.spatial?.operations[0].laneLayout).toBe('quad-major-xy');
  expect(await check(metadata => {
    const cell = metadata.operations[0].cells[0];
    cell.coveredLanes[0] = 2;
    cell.coveredLanes[1] = 0;
  }, false, 0, 'pixel')).toBeUndefined();
});

it.each([
  (m: SpatialMetadata) => { Reflect.set(m, 'schemaVersion', 2); },
  (m: SpatialMetadata) => { Reflect.set(m.operations[0], 'privatePath', '/secret'); },
  (m: SpatialMetadata) => { m.recordBytes = 424; },
  (m: SpatialMetadata) => { m.sampleStride = 0; },
  (m: SpatialMetadata) => { m.operations[0].codeVersionId = 2; },
  (m: SpatialMetadata) => { m.operations[0].eventId = 2; },
  (m: SpatialMetadata) => { m.operations[0].pso = 'd'.repeat(32); },
  (m: SpatialMetadata) => { m.operations[0].dimensions = [0xffffffff, 0xffffffff, 0xffffffff]; },
  (m: SpatialMetadata) => { m.operations[0].cells[0].xyz = [3, 0, 0]; },
  (m: SpatialMetadata) => { m.operations[0].cells[0].xyz = [1, 0, 0]; },
  (m: SpatialMetadata) => { m.operations[0].cells[1].xyz = [0, 0, 0]; },
  (m: SpatialMetadata) => { m.operations[0].cells[0].coveredLanes = [1]; },
  (m: SpatialMetadata) => { m.operations[0].cells[0].invocations = 7; },
  (m: SpatialMetadata) => { m.operations[0].cells[0].wallNs = Number.MAX_SAFE_INTEGER + 1; },
  (m: SpatialMetadata) => { m.recorded = 1; },
  (m: SpatialMetadata) => { m.complete = false; },
  (m: SpatialMetadata) => { m.operations[0].availability = 'dispatch-wide-entry-unavailable'; },
])('rejects the same malformed spatial payload in browser and native readers', async change => {
  expect(await check(change)).toBeUndefined();
});

it.runIf(Boolean(process.env.NACELLE_NATIVE_SPATIAL_PROFILE))('loads actual captured threadgroups and raster tile/lane counts', async () => {
  const bytes = readFileSync(process.env.NACELLE_NATIVE_SPATIAL_PROFILE!);
  const profile = await loadProfile(FileProfileDataSource.fromBuffer(bytes.buffer.slice(
    bytes.byteOffset, bytes.byteOffset + bytes.byteLength,
  )));
  expect(profile.spatial?.operations.some(operation => operation.stage === 'compute' && operation.cells.length > 0)).toBe(true);
  expect(profile.spatial?.operations.some(operation => operation.stage === 'pixel' && operation.cells.length > 0)).toBe(true);
  expect(profile.spatial?.complete).toBe(true);
});
