import { strToU8 } from 'fflate';
import { artifactDigest, type ArtifactMetadata, type TextArtifact } from '../src/artifacts';
import { fixture, span, trace } from './fixtures';
import { nativeManifest } from './profile-corpus';

export async function artifactFixture(change?: (metadata: ArtifactMetadata) => void,
  options: Record<string, unknown> = {}): Promise<ArrayBuffer> {
  const shader = 'a'.repeat(64), native = 'b'.repeat(64);
  const payload = async (kind: TextArtifact['kind'], owner: string, text: string): Promise<TextArtifact> => ({
    kind, owner, text, label: `shader.${kind}`, mappings: [],
    id: `${owner}-${kind}-${await artifactDigest(strToU8(text))}`,
  });
  const source = await payload('hlsl', shader, 'float a = 1;\nfloat b = a;\n// <img src=x onerror=alert(1)>');
  const dxil = await payload('dxil', shader, '; unmapped\n%a = fadd float 0.0, 1.0\n%b = fadd float %a, 0.0');
  dxil.mappings = [{ line: 2, sourceLine: 1, source: source.id }, { line: 3, sourceLine: 2, source: source.id }];
  const assembly = await payload('assembly', native, '0: ret');
  assembly.mappings = [{ line: 1, sourceLine: 0, source: '', offset: 0, size: 1 }];
  const llvm = await payload('llvm', native, 'define void @shader() { ret void }');
  const metadata: ArtifactMetadata = {
    schemaVersion: 1, payloads: [source, dxil, assembly, llvm],
    selections: [
      { codeVersionId: 1, payloadIds: [source.id, dxil.id], shaderAvailability: 'available',
        nativeAvailability: 'tier1-unavailable', mappingAvailability: 'partial', laneAvailability: 'unavailable' },
      { codeVersionId: 2, payloadIds: [source.id, dxil.id, assembly.id, llvm.id], shaderAvailability: 'available',
        nativeAvailability: 'available', mappingAvailability: 'partial', laneAvailability: 'unavailable' },
    ],
  };
  change?.(metadata);
  const artifactBytes = strToU8(JSON.stringify(metadata));
  const digest = await artifactDigest(artifactBytes);
  const code = { id: 1, pipelineId: 1, pso: 'c'.repeat(32), stage: 'compute',
    simdWidth: 8, version: 1, tier: 1, cache: 'miss', shaderId: shader };
  const versions = [code, { ...code, id: 2, version: 2, tier: 2, artifactId: native }];
  return fixture({ ...nativeManifest, schemaVersion: 2,
    captureOptions: { embeddedSources: true, resourceSnapshots: false, eventBufferBytes: 1024,
      artifactBytes: 1024 * 1024, sourceBytes: 1024 * 1024, ...options },
  }, trace(versions.map(version => span({ ts: version.id, args: {
    id: version.id, parent: 0, pso: version.pso, stage: 'compute', simdWidth: 8,
    tier: version.tier, cache: 'miss', pipelineId: 1, codeVersionId: version.id,
  } }))), (entries, manifest) => {
    entries['shaders.json'] = strToU8(JSON.stringify({ schemaVersion: 2, codeVersions: versions }));
    manifest.entries[1].uncompressedSize = entries['shaders.json'].length;
    entries['artifacts.json'] = artifactBytes;
    manifest.entries.push({ path: 'artifacts.json', encoding: 'json', uncompressedSize: artifactBytes.length, digest });
  });
}
