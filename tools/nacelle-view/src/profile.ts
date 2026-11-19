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
  codeVersions: 65_536,
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
  shaders: ShaderMetadata;
}

export interface CodeVersion {
  id: number;
  pipelineId: number;
  pso: string;
  stage: 'compute';
  simdWidth: number;
  version: 1 | 2;
  tier: 1 | 2;
  cache: 'hit' | 'miss';
}

export interface ShaderMetadata {
  schemaVersion: 1;
  codeVersions: CodeVersion[];
}

export function validateShaderMetadata(value: unknown, trace: Trace): ShaderMetadata {
  if (Array.isArray(value) && value.length === 0) value = { schemaVersion: 1, codeVersions: [] };
  requireValid(object(value) && value.schemaVersion === 1 && Array.isArray(value.codeVersions)
    && value.codeVersions.length <= LIMITS.codeVersions, 'Unsupported shader metadata schema or code-version count.');
  const identities = new Set<string>();
  const pipelines = new Map<number, string>();
  for (const [index, code] of value.codeVersions.entries()) {
    requireValid(object(code) && code.id === index + 1
      && Number.isSafeInteger(code.pipelineId) && (code.pipelineId as number) > 0
      && (code.pipelineId as number) <= LIMITS.codeVersions
      && typeof code.pso === 'string' && /^[0-9a-f]{32}$/.test(code.pso)
      && code.stage === 'compute' && Number.isSafeInteger(code.simdWidth)
      && (code.simdWidth as number) > 0 && (code.simdWidth as number) <= 0xffffffff
      && (code.version === 1 || code.version === 2) && (code.tier === 1 || code.tier === 2)
      && (code.cache === 'hit' || code.cache === 'miss')
      && (code.cache !== 'hit' || (code.tier === 2 && code.version === 1))
      && (code.version !== 2 || code.tier === 2), 'Invalid compute code-version metadata.');
    const identity = `${code.pipelineId}:${code.version}`;
    const signature = `${code.pso}:${code.simdWidth}`;
    const pipeline = code.pipelineId as number;
    requireValid(!identities.has(identity)
      && (!pipelines.has(pipeline) || pipelines.get(pipeline) === signature),
    'Inconsistent compute pipeline or duplicate code-version metadata.');
    identities.add(identity);
    pipelines.set(pipeline, signature);
  }
  const shaders = value as unknown as ShaderMetadata;
  for (const { args } of trace.traceEvents) {
    if (args.pipelineId === undefined && args.codeVersionId === undefined) continue;
    requireValid(Number.isSafeInteger(args.codeVersionId) && (args.codeVersionId as number) > 0
      && (args.codeVersionId as number) <= shaders.codeVersions.length, 'Invalid trace code-version reference.');
    const code = shaders.codeVersions[(args.codeVersionId as number) - 1];
    requireValid(args.pipelineId === code.pipelineId && args.pso === code.pso
      && args.stage === code.stage && args.simdWidth === code.simdWidth
      && args.tier === code.tier && args.cache === code.cache,
    'Trace does not match referenced compute code version.');
  }
  return shaders;
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
    const shaders = validateShaderMetadata(parseJson(await source.readEntry('shaders.json')), trace);
    return { manifest, trace, traceBytes, shaders };
  } finally {
    source.close();
  }
}
