export interface ProfileManifest {
  schemaVersion: 1;
  kind: 'profile';
  replayable: false;
  complete: boolean;
  recordedEvents: number;
  droppedEvents: number;
  clock: 'monotonic';
  timestampUnit: 'ns';
  build: Record<string, unknown>;
  captureOptions: Record<string, unknown>;
  entries: { path: string; encoding: 'json' | 'zstd'; uncompressedSize: number }[];
}

/** Entry buffers contain decoded bytes, regardless of storage encoding. */
export interface ProfileDataSource {
  readManifest(): Promise<ProfileManifest>;
  readEntry(path: string): Promise<ArrayBuffer>;
  close(): void;
}

export const LIMITS = {
  archive: 256 * 1024 * 1024,
  entry: 256 * 1024 * 1024,
  total: 256 * 1024 * 1024,
  manifest: 1024 * 1024,
  entries: 16,
  events: 1_000_000,
} as const;

export function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function requireValid(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

export function safePath(path: string): boolean {
  return path.length <= 256 && /^[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*$/.test(path)
    && path.split('/').every(part => !['.', '..', '__proto__', 'constructor', 'prototype'].includes(part));
}

export function parseJson(data: ArrayBuffer | Uint8Array): unknown {
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(data));
}

export function validateManifest(value: unknown): ProfileManifest {
  requireValid(object(value), 'Manifest must be an object.');
  requireValid(value.schemaVersion === 1 && value.kind === 'profile'
    && value.replayable === false && typeof value.complete === 'boolean'
    && value.clock === 'monotonic' && value.timestampUnit === 'ns',
  'Unsupported or invalid profile manifest schema.');
  for (const key of ['recordedEvents', 'droppedEvents']) {
    requireValid(Number.isSafeInteger(value[key]) && (value[key] as number) >= 0,
      `Invalid manifest ${key}.`);
  }
  requireValid(object(value.build) && object(value.captureOptions), 'Invalid capture metadata.');
  requireValid(Array.isArray(value.entries) && value.entries.length > 0
    && value.entries.length <= LIMITS.entries, 'Invalid manifest entry count.');
  const paths = new Set<string>();
  let total = 0;
  for (const entry of value.entries) {
    requireValid(object(entry) && typeof entry.path === 'string' && safePath(entry.path),
      'Invalid or unsafe manifest entry path.');
    requireValid(!paths.has(entry.path), 'Duplicate manifest entry.');
    paths.add(entry.path);
    requireValid(entry.encoding === 'json' || entry.encoding === 'zstd', 'Invalid entry encoding.');
    requireValid(Number.isSafeInteger(entry.uncompressedSize)
      && (entry.uncompressedSize as number) >= 0
      && (entry.uncompressedSize as number) <= LIMITS.entry, 'Entry exceeds decoded size limit.');
    total += entry.uncompressedSize as number;
  }
  requireValid(total <= LIMITS.total, 'Profile exceeds total decoded size limit.');
  requireValid(value.entries.some(e => e.path === 'trace.json.zst' && e.encoding === 'zstd')
    && value.entries.some(e => e.path === 'shaders.json' && e.encoding === 'json'),
  'Manifest must declare trace.json.zst and shaders.json with the correct encodings.');
  return value as unknown as ProfileManifest;
}

export interface TraceEvent {
  name: string;
  cat: string;
  ph: 'X' | 'M';
  ts: number;
  dur: number;
  pid: number;
  tid: number;
  args: Record<string, unknown>;
}

export interface Trace {
  traceEvents: TraceEvent[];
  displayTimeUnit: 'ms';
}

export interface Profile {
  manifest: ProfileManifest;
  trace: Trace;
  traceBytes: ArrayBuffer;
}

export function validateTrace(value: unknown): Trace {
  requireValid(object(value) && value.displayTimeUnit === 'ms' && Array.isArray(value.traceEvents),
    'Invalid Chrome trace document.');
  requireValid(value.traceEvents.length <= LIMITS.events, 'Trace exceeds event count limit.');
  for (const event of value.traceEvents) {
    requireValid(object(event) && typeof event.name === 'string' && typeof event.cat === 'string'
      && (event.ph === 'X' || event.ph === 'M') && object(event.args)
      && Number.isSafeInteger(event.pid) && Number.isSafeInteger(event.tid)
      && typeof event.ts === 'number' && Number.isFinite(event.ts)
      && typeof event.dur === 'number' && Number.isFinite(event.dur) && event.dur >= 0
      && Number.isFinite(event.ts + event.dur), 'Invalid Chrome trace event.');
  }
  return value as unknown as Trace;
}

/** Loading logic depends only on the data-source contract, not on ZIP or File. */
export async function loadProfile(source: ProfileDataSource): Promise<Profile> {
  try {
    const manifest = validateManifest(await source.readManifest());
    const traceBytes = await source.readEntry('trace.json.zst');
    const trace = validateTrace(parseJson(traceBytes));
    const shaders = parseJson(await source.readEntry('shaders.json'));
    requireValid(object(shaders) || Array.isArray(shaders), 'Invalid shader metadata JSON.');
    return { manifest, trace, traceBytes };
  } finally {
    source.close();
  }
}
