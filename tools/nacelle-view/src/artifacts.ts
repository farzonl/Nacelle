import xxhash from 'xxhash-wasm';
import { object, parseJson, requireValid, type ProfileManifest, type ShaderMetadata } from './profile';

export type ArtifactKind = 'hlsl' | 'dxil' | 'mlir' | 'llvm' | 'assembly';
export interface ArtifactMapping {
  line: number;
  sourceLine: number;
  source: string;
  offset?: number;
  size?: number;
}
export interface TextArtifact {
  id: string;
  owner: string;
  kind: ArtifactKind;
  label: string;
  text: string;
  mappings: ArtifactMapping[];
}
export interface ArtifactSelection {
  codeVersionId: number;
  payloadIds: string[];
  shaderAvailability: 'available' | 'not-retained';
  nativeAvailability: 'available' | 'not-retained' | 'tier1-unavailable';
  mappingAvailability: 'partial';
  laneAvailability: 'unavailable';
}
export interface ArtifactMetadata {
  schemaVersion: 1;
  payloads: TextArtifact[];
  selections: ArtifactSelection[];
}

let hash: ReturnType<typeof xxhash> | undefined;
export async function artifactDigest(bytes: Uint8Array): Promise<string> {
  return (await (hash ??= xxhash())).h64Raw(bytes).toString(16).padStart(16, '0');
}

function isArtifactKind(value: unknown): value is ArtifactKind {
  return value === 'hlsl' || value === 'dxil' || value === 'mlir' || value === 'llvm' || value === 'assembly';
}

function hasOnlyKeys(value: Record<string, unknown>, keys: string[]): boolean {
  return Object.keys(value).every(key => keys.includes(key));
}

export async function validateArtifacts(
  bytes: ArrayBuffer, manifest: ProfileManifest, shaders: ShaderMetadata,
): Promise<ArtifactMetadata> {
  const entry = manifest.entries.find(item => item.path === 'artifacts.json');
  requireValid(entry && bytes.byteLength === entry.uncompressedSize &&
    await artifactDigest(new Uint8Array(bytes)) === entry.digest, 'Artifact size or digest mismatch.');
  const value = parseJson(bytes);
  requireValid(object(value) && value.schemaVersion === 1 &&
    hasOnlyKeys(value, ['schemaVersion', 'payloads', 'selections']) &&
    Array.isArray(value.payloads) && value.payloads.length <= 4096 &&
    Array.isArray(value.selections) && value.selections.length === shaders.codeVersions.length,
  'Unsupported artifact schema or count.');
  const payloads = new Map<string, TextArtifact>();
  const lineCounts = new Map<string, number>();
  let sourceBytes = 0, artifactBytes = 0;
  for (const payload of value.payloads) {
    requireValid(object(payload) && typeof payload.owner === 'string' && /^[0-9a-f]{64}$/.test(payload.owner)
      && hasOnlyKeys(payload, ['id', 'owner', 'kind', 'label', 'text', 'mappings'])
      && isArtifactKind(payload.kind) &&
      (payload.kind !== 'hlsl' || manifest.captureOptions.embeddedSources === true)
      && typeof payload.label === 'string' && /^[A-Za-z0-9_.-]{1,128}$/.test(payload.label)
      && typeof payload.text === 'string' && Array.isArray(payload.mappings) && payload.mappings.length <= 65536,
    'Invalid artifact kind, label or source consent.');
    const text = new TextEncoder().encode(payload.text);
    const digest = await artifactDigest(text);
    requireValid(payload.id === `${payload.owner}-${payload.kind}-${digest}` && !payloads.has(String(payload.id)),
      'Invalid or duplicate artifact identity.');
    payloads.set(payload.id, { id: payload.id, owner: payload.owner, kind: payload.kind,
      label: payload.label, text: payload.text, mappings: [] });
    if (payload.kind === 'hlsl') sourceBytes += text.byteLength;
    else artifactBytes += text.byteLength;
    let lines = 1;
    for (let index = 0; index < payload.text.length; index++) if (payload.text[index] === '\n') lines++;
    lineCounts.set(payload.id, lines);
  }
  requireValid(typeof manifest.captureOptions.sourceBytes === 'number' && sourceBytes <= manifest.captureOptions.sourceBytes &&
    typeof manifest.captureOptions.artifactBytes === 'number' && artifactBytes <= manifest.captureOptions.artifactBytes,
  'Artifact/source text exceeds declared retention budget.');
  for (const payload of value.payloads) {
    const lines = lineCounts.get(payload.id)!;
    for (const mapping of payload.mappings) {
      requireValid(object(mapping) && hasOnlyKeys(mapping, ['line', 'sourceLine', 'source', 'offset', 'size']) &&
        typeof mapping.line === 'number' && Number.isInteger(mapping.line) &&
        mapping.line > 0 && mapping.line <= lines && typeof mapping.source === 'string' &&
        typeof mapping.sourceLine === 'number' && Number.isInteger(mapping.sourceLine),
      'Invalid artifact mapping line.');
      const source = payloads.get(mapping.source);
      requireValid(mapping.source ? source && source.kind === 'hlsl' && source.owner === payload.owner
        && mapping.sourceLine > 0 && mapping.sourceLine <= lineCounts.get(source.id)! : mapping.sourceLine === 0,
      'Invalid artifact source reference.');
      requireValid((mapping.offset === undefined || typeof mapping.offset === 'number') &&
        (mapping.size === undefined || typeof mapping.size === 'number') &&
        (mapping.offset === undefined) === (mapping.size === undefined) &&
        (mapping.offset === undefined || typeof mapping.offset === 'number' && Number.isSafeInteger(mapping.offset) &&
          mapping.offset >= 0 && typeof mapping.size === 'number' &&
          Number.isInteger(mapping.size) && mapping.size > 0 && mapping.offset + mapping.size <= 256 * 1024 * 1024),
      'Invalid object-relative instruction range.');
      payloads.get(payload.id)!.mappings.push({ line: mapping.line, source: mapping.source,
        sourceLine: mapping.sourceLine, offset: mapping.offset, size: mapping.size });
    }
  }
  const used = new Set<string>();
  const selections: ArtifactSelection[] = [];
  for (const [index, selection] of value.selections.entries()) {
    requireValid(object(selection) && selection.codeVersionId === index + 1 &&
      hasOnlyKeys(selection, ['codeVersionId', 'payloadIds', 'shaderAvailability', 'nativeAvailability',
        'mappingAvailability', 'laneAvailability']) &&
      Array.isArray(selection.payloadIds) && selection.payloadIds.length <= 64 &&
      (selection.shaderAvailability === 'available' || selection.shaderAvailability === 'not-retained') &&
      (selection.nativeAvailability === 'available' || selection.nativeAvailability === 'not-retained' ||
        selection.nativeAvailability === 'tier1-unavailable') &&
      selection.mappingAvailability === 'partial' && selection.laneAvailability === 'unavailable',
    'Invalid artifact code-version selection.');
    const code = shaders.codeVersions[index];
    requireValid(selection.nativeAvailability !== 'tier1-unavailable' || code.tier === 1 && !code.artifactId,
      'Tier-1 unavailable status disagrees with code version.');
    const references = new Set<string>();
    let shader = false, native = false;
    for (const id of selection.payloadIds) {
      const payload = payloads.get(id);
      const nativeKind = payload?.kind === 'assembly' || payload?.kind === 'llvm';
      requireValid(payload && !references.has(id) && payload.owner === (nativeKind ? code.artifactId : code.shaderId),
      'Artifact does not belong to the selected code version.');
      references.add(id);
      used.add(id);
      shader ||= !nativeKind;
      native ||= nativeKind;
    }
    requireValid(shader === (selection.shaderAvailability === 'available') &&
      native === (selection.nativeAvailability === 'available'), 'Artifact availability mismatch.');
    selections.push({ codeVersionId: selection.codeVersionId, payloadIds: [...references],
      shaderAvailability: selection.shaderAvailability, nativeAvailability: selection.nativeAvailability,
      mappingAvailability: 'partial', laneAvailability: 'unavailable' });
  }
  requireValid(used.size === payloads.size, 'Unreferenced artifact payload.');
  return { schemaVersion: 1, payloads: [...payloads.values()], selections };
}
