import type { SpatialCell, SpatialMetadata, SpatialOperation, XYZ } from './spatial';

export type SpatialMetric = 'wallNs' | 'calls' | 'invocations' | 'coveredLanes' | 'companionLanes' | 'paddingLanes';
export type SpatialPlane = 'xy' | 'xz' | 'yz';
export const planeAxes: Record<SpatialPlane, [number, number, number]> = {
  xy: [0, 1, 2], xz: [0, 2, 1], yz: [1, 2, 0],
};
export const metricUnits: Record<SpatialMetric, string> = {
  wallNs: 'host entry-call ns', calls: 'entry calls', invocations: 'submitted invocations',
  coveredLanes: 'covered submissions', companionLanes: 'quad-companion submissions', paddingLanes: 'padding submissions',
};

export function spatialValue(cell: SpatialCell, metric: SpatialMetric, lane: number): number {
  const value = cell[metric];
  return typeof value === 'number' ? value : lane < 0 ? value.reduce((sum, count) => sum + count, 0) : value[lane];
}

export function coordinateStatus(metadata: SpatialMetadata, operation: SpatialOperation, xyz: XYZ,
  cell?: SpatialCell): string {
  if (operation.availability !== 'available') return 'Unavailable: dispatch-wide entry has no per-group observations.';
  if (cell) return cell.status === 'partial' ? 'Partial: dropped coordinates are unknown; this cell may be incomplete.' : 'Observed';
  if (xyz.some((value, axis) => value < metadata.minXYZ[axis] || value >= metadata.maxXYZ[axis] ||
    (value - metadata.minXYZ[axis]) % metadata.sampleStride !== 0))
    return 'Unsampled: outside ROI or deterministic sampling selection.';
  return operation.dropped ? 'Dropped-or-missing: lost coordinates cannot be identified; no numeric value.' :
    'Missing: no recorded submission at this selected coordinate; not numeric zero.';
}

export function heatColor(value: number, minimum: number, maximum: number, logarithmic: boolean): string {
  const transform = logarithmic ? Math.log1p : (number: number) => number;
  const fraction = Math.max(0, Math.min(1,
    (transform(value) - transform(minimum)) / (transform(maximum) - transform(minimum))));
  return `rgb(${Math.round(30 + 225 * fraction)}, ${Math.round(80 + 100 * fraction)}, ${Math.round(220 - 190 * fraction)})`;
}

export function cellRectangle(operation: SpatialOperation, xyz: XYZ, plane: SpatialPlane,
  width: number, height: number): [number, number, number, number] {
  const [horizontal, vertical] = planeAxes[plane];
  const startX = xyz[horizontal] * operation.cellExtent[horizontal];
  const startY = xyz[vertical] * operation.cellExtent[vertical];
  return [startX / operation.extent[horizontal] * width, startY / operation.extent[vertical] * height,
    Math.min(operation.cellExtent[horizontal], operation.extent[horizontal] - startX) / operation.extent[horizontal] * width,
    Math.min(operation.cellExtent[vertical], operation.extent[vertical] - startY) / operation.extent[vertical] * height];
}
