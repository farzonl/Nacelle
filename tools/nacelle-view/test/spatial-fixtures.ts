import { strToU8 } from 'fflate';
import { artifactDigest } from '../src/artifacts';
import type { SpatialMetadata, SpatialOperation } from '../src/spatial';
import { fixture, span, trace } from './fixtures';
import { nativeManifest } from './profile-corpus';

export async function spatialFixture(change?: (metadata: SpatialMetadata) => void,
  stage: 'compute' | 'pixel' = 'compute'): Promise<ArrayBuffer> {
  const pixel = stage === 'pixel';
  const code = { id: 1, pipelineId: 1, pso: 'c'.repeat(32), stage,
    simdWidth: 8, version: 1, tier: 1, cache: 'miss', ...(pixel ? { entryRole: 'pixel' } : {}) };
  const operation: SpatialOperation = {
    id: 1, eventId: 1, pipelineId: 1, codeVersionId: 1, pso: code.pso, stage,
    granularity: pixel ? 'raster-tile' : 'threadgroup', orientation: pixel ? 'top-left-y-down' : 'group-xyz', availability: 'available',
    timeUnit: 'ns', laneSemantics: 'entry-submission', laneLayout: pixel ? 'quad-major-xy' : 'unavailable',
    dimensions: [3, 2, 1], extent: pixel ? [96, 64, 1] : [3, 2, 1],
    threadsPerGroup: pixel ? [0, 0, 0] : [4, 2, 1], cellExtent: pixel ? [32, 32, 1] : [1, 1, 1], simdWidth: 8,
    recorded: 2, skipped: 4, dropped: 0, complete: true, missingStatus: 'unsampled',
    cells: [0, 2].map(x => ({ xyz: [x, 0, 0], status: 'observed', calls: 1, wallNs: x * 10,
      invocations: 8, coveredLanes: pixel ? Array(8).fill(1) : [],
      companionLanes: pixel ? Array(8).fill(0) : [], paddingLanes: pixel ? Array(8).fill(0) : [] })),
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
  }, trace([span({ args: { id: 1, parent: 0, pso: code.pso, stage: code.stage, simdWidth: 8,
    tier: 1, cache: 'miss', pipelineId: 1, codeVersionId: 1 } })]), (entries, manifest) => {
    entries['shaders.json'] = strToU8(JSON.stringify({ schemaVersion: 2, codeVersions: [code] }));
    manifest.entries[1].uncompressedSize = entries['shaders.json'].length;
    entries['spatial.json'] = bytes;
    manifest.entries.push({ path: 'spatial.json', encoding: 'json', uncompressedSize: bytes.length, digest });
  });
}
