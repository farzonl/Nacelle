import { expect, it } from 'vitest';
import { cellRectangle, coordinateStatus, heatColor, spatialValue } from '../src/heatmap';
import { FileProfileDataSource } from '../src/file-source';
import { loadProfile } from '../src/profile';
import { spatialFixture } from './spatial-fixtures';

it('distinguishes observed zero, selection omissions, missing submissions, partial loss and unsupported precision', async () => {
  const profile = await loadProfile(FileProfileDataSource.fromBuffer(await spatialFixture()));
  const metadata = profile.spatial!;
  const operation = metadata.operations[0];
  const zero = operation.cells[0];
  expect(spatialValue(zero, 'wallNs', -1)).toBe(0);
  expect(coordinateStatus(metadata, operation, zero.xyz, zero)).toBe('Observed');
  expect(coordinateStatus(metadata, operation, [1, 0, 0])).toContain('Unsampled');
  expect(coordinateStatus(metadata, operation, [2, 0, 0])).toContain('Missing');
  operation.dropped = 1;
  zero.status = 'partial';
  expect(coordinateStatus(metadata, operation, zero.xyz, zero)).toContain('Partial');
  expect(coordinateStatus(metadata, operation, [2, 0, 0])).toContain('Dropped-or-missing');
  operation.availability = 'dispatch-wide-entry-unavailable';
  expect(coordinateStatus(metadata, operation, [2, 0, 0])).toContain('Unavailable');
});

it('projects odd framebuffer edges at true extents and 3D group slices without padding the image', async () => {
  const profile = await loadProfile(FileProfileDataSource.fromBuffer(await spatialFixture(metadata => {
    metadata.operations[0].extent = [65, 33, 1];
  }, 'pixel')));
  const operation = profile.spatial!.operations[0];
  expect(cellRectangle(operation, [2, 1, 0], 'xy', 650, 330)).toEqual([640, 320, 10, 10]);
  const compute = (await loadProfile(FileProfileDataSource.fromBuffer(await spatialFixture()))).spatial!.operations[0];
  expect(cellRectangle(compute, [2, 1, 0], 'xz', 300, 100)).toEqual([200, 0, 100, 100]);
  expect(cellRectangle(compute, [2, 1, 0], 'yz', 200, 100)).toEqual([100, 0, 100, 100]);
});

it('colors exact zero and clipped ranges, supports log scale, and selects true quad slots', async () => {
  expect(heatColor(0, 0, 100, false)).toBe('rgb(30, 80, 220)');
  expect(heatColor(200, 0, 100, false)).toBe('rgb(255, 180, 30)');
  expect(heatColor(10, 0, 100, true)).not.toBe(heatColor(10, 0, 100, false));
  const profile = await loadProfile(FileProfileDataSource.fromBuffer(await spatialFixture(undefined, 'pixel')));
  const cell = profile.spatial!.operations[0].cells[0];
  cell.coveredLanes = [0, 1, 2, 3, 4, 5, 6, 7];
  expect(spatialValue(cell, 'coveredLanes', -1)).toBe(28);
  expect(spatialValue(cell, 'coveredLanes', 6)).toBe(6);
});
