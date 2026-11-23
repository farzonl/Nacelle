import { constants, zstdCompressSync } from 'node:zlib';
import { strToU8 } from 'fflate';
import { FileProfileDataSource } from '../src/file-source';
import { LIMITS, loadProfile, type ProfileManifest } from '../src/profile';
import { centralOffsets, fixture, span, trace } from './fixtures';

const nativeManifest: Partial<ProfileManifest> = {
  build: { compiler: '', operatingSystem: '', architecture: '', revision: '',
    compilerHash: '', targetTriple: '', cpu: '', cpuFeatures: [] },
  captureOptions: { eventBufferBytes: 1024, embeddedSources: false, resourceSnapshots: false },
};
const event = span({ args: { id: 1, parent: 0, pso: '', stage: 'unknown', simdWidth: 0, tier: 0, cache: 'unknown' } });

export interface CorpusCase {
  name: string;
  bytes: ArrayBuffer;
  accepted: boolean;
}

export function profileCorpus(): CorpusCase[] {
  const valid = () => fixture(nativeManifest, trace([event]));
  const cases: CorpusCase[] = [
    { name: 'complete', bytes: valid(), accepted: true },
    { name: 'empty', bytes: fixture(nativeManifest, trace([])), accepted: true },
    { name: 'incomplete', bytes: fixture({ ...nativeManifest, complete: false, droppedEvents: 7 }, trace([event])), accepted: true },
  ];
  for (const [name, offset] of [['flags', 6], ['method', 8], ['crc', 14], ['name', 30]] as const) {
    const bytes = valid();
    new Uint8Array(bytes)[offset] ^= 1;
    cases.push({ name: `local-${name}-conflict`, bytes, accepted: false });
  }
  for (const mode of ['overlap', 'entry-size', 'directory-bounds', 'payload-crc', 'central-flags']) {
    const bytes = valid();
    const view = new DataView(bytes);
    const offsets = centralOffsets(bytes);
    const local = view.getUint32(offsets[0] + 42, true);
    if (mode === 'overlap') view.setUint32(offsets[1] + 42, local, true);
    if (mode === 'entry-size') view.setUint32(offsets[0] + 24, LIMITS.entry + 1, true);
    if (mode === 'directory-bounds') view.setUint32(bytes.byteLength - 22 + 16, bytes.byteLength, true);
    if (mode === 'central-flags') view.setUint16(offsets[0] + 8, 1, true);
    if (mode === 'payload-crc') new Uint8Array(bytes)[local + 30 + view.getUint16(local + 26, true)] ^= 1;
    cases.push({ name: mode, bytes, accepted: false });
  }
  const duplicate = valid();
  const duplicateView = new DataView(duplicate);
  const [first, second] = centralOffsets(duplicate);
  duplicateView.setUint16(second + 28, duplicateView.getUint16(first + 28, true), true);
  new Uint8Array(duplicate, second + 46, duplicateView.getUint16(first + 28, true))
    .set(new Uint8Array(duplicate, first + 46, duplicateView.getUint16(first + 28, true)));
  cases.push({ name: 'duplicate-directory-name', bytes: duplicate, accepted: false });
  const trailing = new Uint8Array(valid().byteLength + 1);
  trailing.set(new Uint8Array(valid()));
  cases.push({ name: 'trailing-zip-data', bytes: trailing.buffer, accepted: false });
  for (const path of ['../escape', '/absolute', 'C:\\private', '__proto__']) {
    cases.push({ name: `unsafe-${path}`, accepted: false,
      bytes: fixture(nativeManifest, trace([event]), entries => { entries[path] = strToU8('{}'); }) });
  }
  const checksumBytes = new Uint8Array(zstdCompressSync(strToU8(JSON.stringify(trace([event]))), {
    params: { [constants.ZSTD_c_checksumFlag]: 1 },
  }));
  for (const corrupt of [false, true]) {
    const payload = checksumBytes.slice();
    if (corrupt) payload[payload.length - 1] ^= 1;
    cases.push({ name: corrupt ? 'zstd-checksum-corrupt' : 'zstd-checksum-valid', accepted: !corrupt,
      bytes: fixture(nativeManifest, trace([event]), entries => { entries['trace.json.zst'] = payload; }) });
  }
  for (const mode of ['truncated', 'concatenated', 'dictionary', 'window']) {
    cases.push({ name: `zstd-${mode}`, accepted: false,
      bytes: fixture(nativeManifest, trace([event]), entries => {
        const data = entries['trace.json.zst'];
        if (mode === 'truncated') entries['trace.json.zst'] = data.slice(0, -1);
        if (mode === 'concatenated') {
          const both = new Uint8Array(data.length * 2);
          both.set(data);
          both.set(data, data.length);
          entries['trace.json.zst'] = both;
        }
        if (mode === 'dictionary') data[4] |= 1;
        if (mode === 'window') entries['trace.json.zst'] = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, 0, 0xf8]);
      }) });
  }
  for (const overrides of [
    { recordedEvents: 2 }, { droppedEvents: 1 }, { schemaVersion: 2 as never },
  ]) cases.push({ name: `manifest-${Object.keys(overrides)[0]}`, accepted: false,
    bytes: fixture({ ...nativeManifest, ...overrides }, trace([event])) });
  cases.push({ name: 'trace-negative-time', accepted: false,
    bytes: fixture(nativeManifest, trace([{ ...event, ts: -1 }])) });
  cases.push({ name: 'trace-invalid-display-unit', accepted: false,
    bytes: fixture(nativeManifest, { ...trace([event]), displayTimeUnit: 'seconds' }) });
  cases.push({ name: 'decoded-size-mismatch', accepted: false,
    bytes: fixture(nativeManifest, trace([event]), (_, manifest) => { manifest.entries[0].uncompressedSize--; }) });
  cases.push({ name: 'shader-reference-mismatch', accepted: false,
    bytes: fixture(nativeManifest, trace([{ ...event, args: { ...event.args, pipelineId: 1, codeVersionId: 1 } }])) });
  cases.push({ name: 'truncated-zip', bytes: valid().slice(0, -1), accepted: false });
  return cases;
}

// Reusable browser parser fuzz entry: malformed inputs are expected; bounds faults are not.
export async function fuzzProfileBytes(bytes: ArrayBuffer): Promise<{ accepted: boolean; error?: string }> {
  try {
    await loadProfile(FileProfileDataSource.fromBuffer(bytes));
    return { accepted: true };
  } catch (error) {
    if (!(error instanceof Error) || error instanceof RangeError) throw error;
    return { accepted: false, error: error.message };
  }
}
