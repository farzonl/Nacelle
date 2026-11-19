import { zstdCompressSync } from 'node:zlib';
import { strToU8, zipSync } from 'fflate';
import type { ProfileManifest, Trace, TraceEvent } from '../src/profile';

export function span(overrides: Partial<TraceEvent> = {}): TraceEvent {
  return { name: 'dispatch', cat: 'compute', ph: 'X', ts: 1250.5, dur: 42.25, pid: 1, tid: 2, args: {}, ...overrides };
}

export function trace(events: TraceEvent[] = [span()]): Trace {
  return { traceEvents: events, displayTimeUnit: 'ms' };
}

export function fixture(
  overrides: Partial<ProfileManifest> = {},
  document: unknown = trace(),
  modify?: (entries: Record<string, Uint8Array>, manifest: ProfileManifest) => void,
): ArrayBuffer {
  const json = strToU8(JSON.stringify(document));
  const entries: Record<string, Uint8Array> = {
    'trace.json.zst': new Uint8Array(zstdCompressSync(json)),
    'shaders.json': strToU8('[]'),
  };
  const manifest: ProfileManifest = {
    schemaVersion: 1, kind: 'profile', replayable: false, complete: true,
    recordedEvents: 1, droppedEvents: 0, clock: 'monotonic', timestampUnit: 'ns',
    build: { version: 'test' }, captureOptions: {},
    entries: [
      { path: 'trace.json.zst', encoding: 'zstd', uncompressedSize: json.length },
      { path: 'shaders.json', encoding: 'json', uncompressedSize: 2 },
    ],
    ...overrides,
  };
  modify?.(entries, manifest);
  entries['manifest.json'] = strToU8(JSON.stringify(manifest));
  return zipSync(entries, { level: 0 }).slice().buffer as ArrayBuffer;
}

export function centralOffsets(buffer: ArrayBuffer): number[] {
  const view = new DataView(buffer);
  const end = buffer.byteLength - 22;
  let cursor = view.getUint32(end + 16, true);
  const offsets = [];
  while (cursor < end) {
    offsets.push(cursor);
    cursor += 46 + view.getUint16(cursor + 28, true)
      + view.getUint16(cursor + 30, true) + view.getUint16(cursor + 32, true);
  }
  return offsets;
}
