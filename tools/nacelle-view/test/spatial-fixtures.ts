import { strToU8, unzipSync, zipSync } from 'fflate';
import { artifactDigest } from '../src/artifacts';
import type { SpatialMetadata, SpatialOperation } from '../src/spatial';
import { fixture, span, trace } from './fixtures';
import { nativeManifest } from './profile-corpus';
import { artifactFixture } from './artifact-fixtures';

export async function spatialFixture(change?: (metadata: SpatialMetadata) => void,
  stage: 'compute' | 'pixel' = 'compute', simdWidth = 8): Promise<ArrayBuffer> {
  const pixel = stage === 'pixel';
  const code = { id: 1, pipelineId: 1, pso: 'c'.repeat(32), stage,
    simdWidth, version: 1, tier: 1, cache: 'miss', ...(pixel ? { entryRole: 'pixel' } : {}) };
  const operation: SpatialOperation = {
    id: 1, eventId: 1, pipelineId: 1, codeVersionId: 1, pso: code.pso, stage,
    granularity: pixel ? 'raster-tile' : 'threadgroup', orientation: pixel ? 'top-left-y-down' : 'group-xyz', availability: 'available',
    timeUnit: 'ns', laneSemantics: 'entry-submission', laneLayout: pixel ? 'quad-major-xy' : 'unavailable',
    dimensions: [3, 2, 1], extent: pixel ? [96, 64, 1] : [3, 2, 1],
    threadsPerGroup: pixel ? [0, 0, 0] : [4, 2, 1], cellExtent: pixel ? [32, 32, 1] : [1, 1, 1], simdWidth,
    recorded: 2, skipped: 4, dropped: 0, complete: true, missingStatus: 'unsampled',
    cells: [0, 2].map(x => ({ xyz: [x, 0, 0], status: 'observed', calls: 1, wallNs: x * 10,
      invocations: pixel ? simdWidth : 8, coveredLanes: pixel ? Array(simdWidth).fill(1) : [],
      companionLanes: pixel ? Array(simdWidth).fill(0) : [], paddingLanes: pixel ? Array(simdWidth).fill(0) : [] })),
  };
  const metadata: SpatialMetadata = {
    schemaVersion: 1, flags: pixel ? 2 : 1, recordBytes: 848, sampleStride: 2, minXYZ: [0, 0, 0],
    maxXYZ: [0xffffffff, 0xffffffff, 0xffffffff], complete: true, recorded: 2, skipped: 4, dropped: 0,
    operations: [operation],
  };
  change?.(metadata);
  const bytes = strToU8(JSON.stringify(metadata)), digest = await artifactDigest(bytes);
  return fixture({ ...nativeManifest, schemaVersion: 3, complete: metadata.complete,
    captureOptions: { ...nativeManifest.captureOptions, spatialBytes: metadata.recordBytes },
  }, trace([span({ args: { id: 1, parent: 0, pso: code.pso, stage: code.stage, simdWidth,
    tier: 1, cache: 'miss', pipelineId: 1, codeVersionId: 1 } })]), (entries, manifest) => {
    entries['shaders.json'] = strToU8(JSON.stringify({ schemaVersion: 2, codeVersions: [code] }));
    manifest.entries[1].uncompressedSize = entries['shaders.json'].length;
    entries['spatial.json'] = bytes;
    manifest.entries.push({ path: 'spatial.json', encoding: 'json', uncompressedSize: bytes.length, digest });
  });
}

export async function spatialArtifactFixture(): Promise<ArrayBuffer> {
  const spatialEntries = unzipSync(new Uint8Array(await spatialFixture()));
  const spatial: SpatialMetadata = JSON.parse(new TextDecoder().decode(spatialEntries['spatial.json']));
  spatial.operations[0].codeVersionId = 2;
  spatial.operations[0].eventId = 2;
  spatial.operations[0].dimensions = spatial.operations[0].extent = [3, 2, 3];
  spatial.operations[0].cells[1].xyz = [2, 0, 2];
  const entries = unzipSync(new Uint8Array(await artifactFixture(undefined, { spatialBytes: spatial.recordBytes })));
  const manifest = JSON.parse(new TextDecoder().decode(entries['manifest.json']));
  manifest.schemaVersion = 3;
  const bytes = strToU8(JSON.stringify(spatial));
  entries['spatial.json'] = bytes;
  manifest.entries.push({ path: 'spatial.json', encoding: 'json', uncompressedSize: bytes.length,
    digest: await artifactDigest(bytes) });
  entries['manifest.json'] = strToU8(JSON.stringify(manifest));
  const archive = zipSync(entries, { level: 0 });
  return archive.buffer.slice(archive.byteOffset, archive.byteOffset + archive.byteLength);
}
