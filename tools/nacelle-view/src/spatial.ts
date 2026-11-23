import { artifactDigest } from './artifacts';
import { object, parseJson, requireValid, type ProfileManifest, type ShaderMetadata } from './profile';

export type XYZ = [number, number, number];
export interface SpatialCell {
  xyz: XYZ;
  status: 'observed' | 'partial';
  calls: number;
  wallNs: number;
  invocations: number;
  coveredLanes: number[];
  companionLanes: number[];
  paddingLanes: number[];
}
export interface SpatialOperation {
  id: number;
  eventId: number;
  pipelineId: number;
  codeVersionId: number;
  stage: 'pixel' | 'compute';
  pso: string;
  granularity: 'raster-tile' | 'threadgroup';
  orientation: 'top-left-y-down' | 'group-xyz';
  availability: 'available' | 'dispatch-wide-entry-unavailable';
  timeUnit: 'ns';
  laneSemantics: 'entry-submission';
  laneLayout: 'unavailable' | 'quad-major-xy';
  dimensions: XYZ;
  extent: XYZ;
  threadsPerGroup: XYZ;
  cellExtent: XYZ;
  simdWidth: number;
  recorded: number;
  skipped: number;
  dropped: number;
  complete: boolean;
  missingStatus: 'unsampled' | 'dropped-or-unsampled' | 'unavailable';
  cells: SpatialCell[];
}
export interface SpatialMetadata {
  schemaVersion: 1;
  flags: number;
  recordBytes: number;
  sampleStride: number;
  minXYZ: XYZ;
  maxXYZ: XYZ;
  complete: boolean;
  recorded: number;
  skipped: number;
  dropped: number;
  operations: SpatialOperation[];
}
// Fixed native record representation; the pool and decoded cells share this bound.
export const SPATIAL_RECORD_BYTES = 424;

function keys(value: unknown, names: string[]): asserts value is Record<string, unknown> {
  requireValid(object(value) && Object.keys(value).length === names.length &&
    Object.keys(value).every(key => names.includes(key)), 'Invalid spatial fields.');
}
function integer(value: unknown, max = Number.MAX_SAFE_INTEGER): number {
  requireValid(typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 && value <= max,
    'Invalid spatial integer.');
  return value;
}
function xyz(value: unknown): XYZ {
  requireValid(Array.isArray(value) && value.length === 3, 'Invalid spatial XYZ.');
  return [integer(value[0], 0xffffffff), integer(value[1], 0xffffffff), integer(value[2], 0xffffffff)];
}
function product(values: XYZ): number {
  let result = 1;
  for (const value of values) {
    requireValid(value > 0 && value <= Number.MAX_SAFE_INTEGER / result, 'Invalid spatial dimension product.');
    result *= value;
  }
  return result;
}
function add(value: number, increment: number): number {
  requireValid(increment <= Number.MAX_SAFE_INTEGER - value, 'Spatial sum exceeds exact integer range.');
  return value + increment;
}
function lanes(value: unknown, count: number): number[] {
  requireValid(Array.isArray(value) && value.length === count, 'Invalid spatial lane slots.');
  return value.map(item => integer(item));
}

export async function validateSpatial(
  bytes: ArrayBuffer, manifest: ProfileManifest, shaders: ShaderMetadata,
): Promise<SpatialMetadata> {
  const entry = manifest.entries.find(item => item.path === 'spatial.json');
  requireValid(entry && bytes.byteLength === entry.uncompressedSize &&
    await artifactDigest(new Uint8Array(bytes)) === entry.digest, 'Spatial size or digest mismatch.');
  const value = parseJson(bytes);
  keys(value, ['schemaVersion', 'flags', 'recordBytes', 'sampleStride', 'minXYZ', 'maxXYZ',
    'complete', 'recorded', 'skipped', 'dropped', 'operations']);
  requireValid(value.schemaVersion === 1 && typeof value.complete === 'boolean' &&
    Array.isArray(value.operations) && value.operations.length <= 4096, 'Invalid spatial schema or operations.');
  const recordBytes = integer(value.recordBytes, 16 * 1024 * 1024);
  const flags = integer(value.flags, 3);
  const sampleStride = integer(value.sampleStride, 0xffffffff);
  const minXYZ = xyz(value.minXYZ), maxXYZ = xyz(value.maxXYZ);
  requireValid(recordBytes >= SPATIAL_RECORD_BYTES && recordBytes === manifest.captureOptions.spatialBytes &&
    flags > 0 && sampleStride > 0 && minXYZ.every((minimum, axis) => minimum < maxXYZ[axis]), 'Invalid spatial budget or sampling.');
  const operations: SpatialOperation[] = [];
  let recorded = 0, skipped = 0, dropped = 0, cellCount = 0;
  for (const [index, item] of value.operations.entries()) {
    keys(item, ['id', 'eventId', 'pipelineId', 'codeVersionId', 'stage', 'pso', 'granularity', 'orientation',
      'availability', 'timeUnit', 'laneSemantics', 'laneLayout', 'dimensions', 'extent', 'threadsPerGroup', 'cellExtent', 'simdWidth',
      'recorded', 'skipped', 'dropped', 'complete', 'missingStatus', 'cells']);
    const id = integer(item.id), eventId = integer(item.eventId);
    const codeVersionId = integer(item.codeVersionId), pipelineId = integer(item.pipelineId);
    const code = shaders.codeVersions[codeVersionId - 1];
    requireValid(id === index + 1 && eventId > 0 && code &&
      (item.stage === 'pixel' || item.stage === 'compute') && item.stage === code.stage &&
      item.pso === code.pso && pipelineId === code.pipelineId && item.simdWidth === code.simdWidth &&
      item.timeUnit === 'ns' && item.laneSemantics === 'entry-submission',
    'Invalid spatial version attribution or metric semantics.');
    const pixel = item.stage === 'pixel';
    requireValid(item.granularity === (pixel ? 'raster-tile' : 'threadgroup') &&
      (flags & (pixel ? 2 : 1)) !== 0 && item.laneLayout === (pixel ? 'quad-major-xy' : 'unavailable') &&
      item.orientation === (pixel ? 'top-left-y-down' : 'group-xyz') &&
      (item.availability === 'available' || item.availability === 'dispatch-wide-entry-unavailable') &&
      (!pixel || [4, 8, 16].includes(code.simdWidth)), 'Invalid spatial granularity or capability.');
    const dimensions = xyz(item.dimensions), extent = xyz(item.extent), threadsPerGroup = xyz(item.threadsPerGroup);
    const cellExtent = xyz(item.cellExtent);
    requireValid(cellExtent.every((value, axis) => value === (pixel && axis < 2 ? 32 : 1) &&
      dimensions[axis] === Math.ceil(extent[axis] / value)), 'Spatial cell extents disagree.');
    product(dimensions);
    product(extent);
    const threads = pixel ? 0 : product(threadsPerGroup);
    const counts = { recorded: integer(item.recorded), skipped: integer(item.skipped), dropped: integer(item.dropped) };
    requireValid(typeof item.complete === 'boolean' && item.complete === (counts.dropped === 0) &&
      item.missingStatus === (item.availability !== 'available' ? 'unavailable' :
        counts.dropped ? 'dropped-or-unsampled' : 'unsampled') &&
      (item.availability === 'available' || !pixel && !counts.recorded && !counts.skipped && !counts.dropped) &&
      Array.isArray(item.cells) && item.cells.length <= 131072, 'Invalid spatial completeness or cell count.');
    const cells: SpatialCell[] = [];
    const coordinates = new Set<string>();
    let calls = 0;
    for (const cell of item.cells) {
      keys(cell, ['xyz', 'status', 'calls', 'wallNs', 'invocations', 'coveredLanes', 'companionLanes', 'paddingLanes']);
      requireValid(cell.status === (item.complete ? 'observed' : 'partial'), 'Invalid spatial cell status.');
      const coordinate = xyz(cell.xyz), key = coordinate.join(',');
      requireValid(++cellCount <= 131072 && !coordinates.has(key) &&
        coordinate.every((coordinate, axis) => coordinate < dimensions[axis] && coordinate >= minXYZ[axis] &&
          coordinate < maxXYZ[axis] && (coordinate - minXYZ[axis]) % sampleStride === 0),
      'Invalid spatial coordinate or sampling selection.');
      coordinates.add(key);
      const count = integer(cell.calls), invocations = integer(cell.invocations);
      requireValid(count > 0, 'Observed spatial cell must have calls.');
      calls = add(calls, count);
      const coveredLanes = lanes(cell.coveredLanes, pixel ? code.simdWidth : 0);
      const companionLanes = lanes(cell.companionLanes, coveredLanes.length);
      const paddingLanes = lanes(cell.paddingLanes, coveredLanes.length);
      if (pixel) {
        requireValid(coveredLanes.every((covered, lane) =>
          add(add(covered, companionLanes[lane]), paddingLanes[lane]) === invocations / code.simdWidth) &&
          invocations / code.simdWidth >= count && invocations % code.simdWidth === 0,
        'Spatial lane submissions disagree with SIMD batches.');
        requireValid([...coveredLanes, ...companionLanes, ...paddingLanes].reduce(add, 0) === invocations,
          'Spatial lane totals disagree.');
      } else requireValid(count <= Number.MAX_SAFE_INTEGER / threads && invocations === count * threads,
        'Spatial compute invocation count disagrees.');
      cells.push({ xyz: coordinate, status: item.complete ? 'observed' : 'partial',
        calls: count, wallNs: integer(cell.wallNs), invocations,
        coveredLanes, companionLanes, paddingLanes });
    }
    requireValid(calls === counts.recorded, 'Spatial calls disagree with recorded count.');
    recorded = add(recorded, counts.recorded);
    skipped = add(skipped, counts.skipped);
    dropped = add(dropped, counts.dropped);
    operations.push({ id, eventId, pipelineId, codeVersionId, stage: item.stage, pso: code.pso,
      granularity: pixel ? 'raster-tile' : 'threadgroup', orientation: pixel ? 'top-left-y-down' : 'group-xyz',
      availability: item.availability, timeUnit: 'ns', laneSemantics: 'entry-submission',
      laneLayout: pixel ? 'quad-major-xy' : 'unavailable', dimensions, extent,
      threadsPerGroup, cellExtent, simdWidth: code.simdWidth, ...counts, complete: item.complete,
      missingStatus: item.availability !== 'available' ? 'unavailable' : counts.dropped ? 'dropped-or-unsampled' : 'unsampled', cells });
  }
  requireValid(recorded === integer(value.recorded) && skipped === integer(value.skipped) &&
    dropped === integer(value.dropped) && recorded <= Math.floor(recordBytes / SPATIAL_RECORD_BYTES) &&
    value.complete === (dropped === 0), 'Spatial pool/totals/completeness disagree.');
  return { schemaVersion: 1, flags, recordBytes, sampleStride, minXYZ, maxXYZ, complete: value.complete,
    recorded, skipped, dropped, operations };
}
