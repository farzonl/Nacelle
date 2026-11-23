import { validateArtifacts, type ArtifactMetadata } from './artifacts';
import { validateSpatial, type SpatialMetadata } from './spatial';

export interface ProfileManifest {
  schemaVersion: 1 | 2 | 3;
  kind: 'profile';
  replayable: false;
  complete: boolean;
  recordedEvents: number;
  droppedEvents: number;
  clock: 'monotonic';
  timestampUnit: 'ns';
  build: Record<string, unknown>;
  captureOptions: Record<string, unknown>;
  entries: { path: string; encoding: 'json' | 'zstd'; uncompressedSize: number; digest?: string }[];
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
  requireValid((value.schemaVersion === 1 || value.schemaVersion === 2 || value.schemaVersion === 3) && value.kind === 'profile'
    && value.replayable === false && typeof value.complete === 'boolean'
    && value.clock === 'monotonic' && value.timestampUnit === 'ns',
  'Unsupported or invalid profile manifest schema.');
  for (const key of ['recordedEvents', 'droppedEvents']) {
    requireValid(Number.isSafeInteger(value[key]) && (value[key] as number) >= 0,
      `Invalid manifest ${key}.`);
  }
  requireValid((value.recordedEvents as number) <= LIMITS.events &&
    (value.schemaVersion === 3 || value.complete === (value.droppedEvents === 0)), 'Inconsistent capture completeness or event count.');
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
  if (value.schemaVersion === 2 || value.schemaVersion === 3 && paths.has('artifacts.json')) {
    const artifacts = value.entries.find(entry => entry.path === 'artifacts.json');
    requireValid(artifacts?.encoding === 'json' && typeof artifacts.digest === 'string'
      && /^[0-9a-f]{16}$/.test(artifacts.digest) &&
      (value.schemaVersion === 3 || value.entries.length === 3),
    'Artifact manifest requires one declared checksummed artifacts.json.');
    const options = value.captureOptions;
    requireValid(typeof options.embeddedSources === 'boolean' &&
      Number.isSafeInteger(options.artifactBytes) && (options.artifactBytes as number) > 0 &&
      (options.artifactBytes as number) <= 64 * 1024 * 1024 &&
      Number.isSafeInteger(options.sourceBytes) && (options.sourceBytes as number) > 0 &&
      (options.sourceBytes as number) <= 16 * 1024 * 1024,
    'Invalid artifact/source budgets or consent.');
  }
  if (value.schemaVersion === 3) {
    const spatial = value.entries.find(entry => entry.path === 'spatial.json');
    requireValid(spatial?.encoding === 'json' && typeof spatial.digest === 'string' &&
      /^[0-9a-f]{16}$/.test(spatial.digest) && value.entries.length === (paths.has('artifacts.json') ? 4 : 3) &&
      value.entries.every(entry => ['trace.json.zst', 'shaders.json', 'artifacts.json', 'spatial.json'].includes(entry.path)) &&
      Number.isSafeInteger(value.captureOptions.spatialBytes) &&
      (value.captureOptions.spatialBytes as number) >= 424 &&
      (value.captureOptions.spatialBytes as number) <= 16 * 1024 * 1024,
    'Invalid spatial manifest declaration or budget.');
  }
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
  artifacts?: ArtifactMetadata;
  spatial?: SpatialMetadata;
}

export interface CodeVersion {
  id: number;
  pipelineId: number;
  pso: string;
  stage: 'compute' | 'vertex' | 'pixel' | 'geometry' | 'hull' | 'domain' | 'mesh' | 'amplification';
  simdWidth: number;
  version: number;
  tier: 0 | 1 | 2;
  cache: 'hit' | 'miss' | 'unknown';
  entryRole?: 'vertex' | 'pixel' | 'geometry' | 'hull-control-point' | 'hull-patch-constant'
    | 'domain' | 'mesh' | 'amplification';
  shaderId?: string;
  artifactId?: string;
}

export interface CodeSelection {
  id: number;
  pipelineId: number;
  codeVersionIds: number[];
}

export interface ShaderMetadata {
  schemaVersion: 1 | 2;
  codeVersions: CodeVersion[];
  codeSelections?: CodeSelection[];
}

export function validateShaderMetadata(value: unknown, trace: Trace): ShaderMetadata {
  if (Array.isArray(value) && value.length === 0) value = { schemaVersion: 1, codeVersions: [] };
  requireValid(object(value) && (value.schemaVersion === 1 || value.schemaVersion === 2) && Array.isArray(value.codeVersions)
    && value.codeVersions.length <= LIMITS.codeVersions, 'Unsupported shader metadata schema or code-version count.');
  requireValid(value.codeSelections === undefined || (value.schemaVersion === 2 && Array.isArray(value.codeSelections)
    && value.codeSelections.length + value.codeVersions.length <= LIMITS.codeVersions),
  'Invalid code-selection metadata or count.');
  const identities = new Set<string>();
  const pipelines = new Map<number, string>();
  for (const [index, code] of value.codeVersions.entries()) {
    requireValid(object(code), 'Invalid code-version metadata.');
    const compute = code.stage === 'compute';
    const graphics = value.schemaVersion === 2
      && ((typeof code.stage === 'string' &&
        ['vertex', 'pixel', 'geometry', 'domain', 'mesh', 'amplification'].includes(code.stage)
        && code.entryRole === code.stage)
        || (code.stage === 'hull' && (code.entryRole === 'hull-control-point' || code.entryRole === 'hull-patch-constant')));
    requireValid(object(code) && code.id === index + 1
      && Number.isSafeInteger(code.pipelineId) && (code.pipelineId as number) > 0
      && (code.pipelineId as number) <= LIMITS.codeVersions
      && typeof code.pso === 'string' && /^[0-9a-f]{32}$/.test(code.pso)
      && (compute || graphics) && Number.isSafeInteger(code.simdWidth)
      && (code.simdWidth as number) > 0 && (code.simdWidth as number) <= 0xffffffff
      && Number.isSafeInteger(code.version) && (code.version as number) > 0
      && (code.version as number) <= 0xffffffff && (code.tier === 0 || code.tier === 1 || code.tier === 2)
      && (code.cache === 'hit' || code.cache === 'miss' || code.cache === 'unknown')
      && (code.shaderId === undefined || (value.schemaVersion === 2
        && typeof code.shaderId === 'string' && /^[0-9a-f]{64}$/.test(code.shaderId)))
      && (code.artifactId === undefined || (value.schemaVersion === 2
        && typeof code.artifactId === 'string' && /^[0-9a-f]{64}$/.test(code.artifactId)))
      && (!compute || (code.entryRole === undefined && (code.version === 1 || code.version === 2)
        && code.tier !== 0 && code.cache !== 'unknown'
        && (code.cache !== 'hit' || (code.tier === 2 && code.version === 1))
        && (code.version !== 2 || code.tier === 2))), 'Invalid code-version metadata.');
    const identity = `${code.pipelineId}:${code.entryRole ?? code.stage}:${code.version}`;
    const signature = `${code.pso}:${code.simdWidth}`;
    const pipeline = code.pipelineId as number;
    requireValid(!identities.has(identity)
      && (!pipelines.has(pipeline) || pipelines.get(pipeline) === signature),
    'Inconsistent pipeline or duplicate code-version metadata.');
    identities.add(identity);
    pipelines.set(pipeline, signature);
  }
  const shaders = value as unknown as ShaderMetadata;
  for (const [index, selection] of (shaders.codeSelections ?? []).entries()) {
    requireValid(object(selection) && selection.id === index + 1
      && Number.isSafeInteger(selection.pipelineId) && selection.pipelineId > 0
      && selection.pipelineId <= LIMITS.codeVersions && Array.isArray(selection.codeVersionIds)
      && selection.codeVersionIds.length <= 8, 'Invalid code-selection metadata.');
    const references = new Set<number>();
    for (const id of selection.codeVersionIds) {
      requireValid(Number.isSafeInteger(id) && id > 0 && id <= shaders.codeVersions.length
        && !references.has(id) && shaders.codeVersions[id - 1].pipelineId === selection.pipelineId
        && shaders.codeVersions[id - 1].stage !== 'compute', 'Invalid code-selection entry reference.');
      references.add(id);
    }
  }
  for (const { args } of trace.traceEvents) {
    let selection: CodeSelection | undefined;
    if (args.codeSelectionId !== undefined) {
      requireValid(Number.isSafeInteger(args.codeSelectionId) && (args.codeSelectionId as number) > 0
        && (args.codeSelectionId as number) <= (shaders.codeSelections?.length ?? 0),
      'Invalid trace code-selection reference.');
      selection = shaders.codeSelections?.[(args.codeSelectionId as number) - 1];
      requireValid(selection && args.pipelineId === selection.pipelineId,
        'Trace does not match referenced code selection.');
      for (const id of selection.codeVersionIds) {
        const code = shaders.codeVersions[id - 1];
        requireValid(args.pso === code.pso && args.simdWidth === code.simdWidth,
          'Trace does not match selected pipeline.');
      }
    }
    if (args.codeVersionId === undefined && (args.pipelineId === undefined || selection)) continue;
    requireValid(Number.isSafeInteger(args.codeVersionId) && (args.codeVersionId as number) > 0
      && (args.codeVersionId as number) <= shaders.codeVersions.length, 'Invalid trace code-version reference.');
    const code = shaders.codeVersions[(args.codeVersionId as number) - 1];
    requireValid((!selection || selection.codeVersionIds.includes(code.id))
      && args.pipelineId === code.pipelineId && args.pso === code.pso
      && args.stage === code.stage && args.simdWidth === code.simdWidth
      && args.tier === code.tier && args.cache === code.cache,
    'Trace does not match referenced code version.');
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
      && (event.pid as number) >= 0 && (event.pid as number) <= 0xffffffff
      && (event.tid as number) >= 0 && (event.tid as number) <= 0xffffffff
      && typeof event.ts === 'number' && Number.isFinite(event.ts)
      && event.ts >= 0 && (event.ph !== 'X' || (event.tid as number) > 0)
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
    requireValid(trace.traceEvents.reduce((count, event) => count + Number(event.ph === 'X'), 0) === manifest.recordedEvents,
      'Trace event count disagrees with manifest.');
    const shaders = validateShaderMetadata(parseJson(await source.readEntry('shaders.json')), trace);
    const artifacts = manifest.schemaVersion >= 2 && manifest.entries.some(entry => entry.path === 'artifacts.json')
      ? await validateArtifacts(await source.readEntry('artifacts.json'), manifest, shaders) : undefined;
    const spatial = manifest.schemaVersion === 3
      ? await validateSpatial(await source.readEntry('spatial.json'), manifest, shaders) : undefined;
    if (spatial) {
      const references = new Map(spatial.operations.map(operation => [operation.eventId, operation]));
      requireValid(references.size === spatial.operations.length, 'Duplicate spatial operation event.');
      for (const event of trace.traceEvents) {
        const operation = references.get(Number(event.args.id));
        if (!operation) continue;
        requireValid(operation.codeVersionId === event.args.codeVersionId && operation.pipelineId === event.args.pipelineId,
          'Spatial operation disagrees with its recorded span.');
        references.delete(operation.eventId);
      }
      requireValid(manifest.droppedEvents > 0 || !references.size, 'Spatial operation references a missing span.');
    }
    requireValid(!spatial || manifest.complete === (manifest.droppedEvents === 0 && spatial.complete),
      'Spatial/event completeness disagrees with manifest.');
    return { manifest, trace, traceBytes, shaders, artifacts, spatial };
  } finally {
    source.close();
  }
}
