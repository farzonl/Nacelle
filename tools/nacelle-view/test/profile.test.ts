import { describe, expect, it, vi } from 'vitest';
import { strToU8 } from 'fflate';
import { FileProfileDataSource } from '../src/file-source';
import { LIMITS, loadProfile, validateManifest, type ProfileDataSource } from '../src/profile';
import { captureStatus, groupThreads, summaryText } from '../src/presentation';
import { centralOffsets, fixture, span, trace } from './fixtures';

const open = (buffer: ArrayBuffer) => FileProfileDataSource.fromBuffer(buffer);

describe('file data source', () => {
  it('loads a native-style stored ZIP with real zstd compression and exports exact decoded bytes', async () => {
    const source = open(fixture());
    const manifest = await source.readManifest();
    manifest.complete = false;
    expect((await source.readManifest()).complete).toBe(true);
    const first = await source.readEntry('trace.json.zst');
    const second = await source.readEntry('trace.json.zst');
    expect(first).toBeInstanceOf(ArrayBuffer);
    expect(first).not.toBe(second);
    const profile = await loadProfile(source);
    expect(new TextDecoder().decode(profile.traceBytes)).toBe(JSON.stringify(trace()));
    expect(profile.trace.traceEvents[0].ts).toBe(1250.5);
    expect(profile.trace.traceEvents[0].dur).toBe(42.25);
    await expect(source.readManifest()).rejects.toThrow('closed');
    await expect(source.readEntry('shaders.json')).rejects.toThrow('closed');
    source.close();
  });

  it('checks browser file size before reading into memory', async () => {
    const arrayBuffer = vi.fn();
    await expect(FileProfileDataSource.open({ size: LIMITS.archive + 1, arrayBuffer })).rejects.toThrow('limit');
    expect(arrayBuffer).not.toHaveBeenCalled();
    const bytes = fixture();
    const source = await FileProfileDataSource.open({
      size: bytes.byteLength, arrayBuffer: async () => bytes,
    });
    expect((await source.readManifest()).schemaVersion).toBe(1);
    await expect(source.readEntry('../manifest.json')).rejects.toThrow('not declared');
    source.close();
  });

  it.each([
    { schemaVersion: 2 }, { kind: 'capture' }, { replayable: true }, { complete: 1 },
    { recordedEvents: -1 }, { droppedEvents: 0.1 }, { clock: 'wall' },
    { timestampUnit: 'us' }, { build: [] }, { captureOptions: null }, { entries: [] },
    { recordedEvents: Number.MAX_SAFE_INTEGER + 1 },
  ])('rejects invalid manifest fields %j', overrides => {
    expect(() => open(fixture(overrides as never))).toThrow();
  });

  it.each(['../escape.json', '/absolute.json', 'a/../b', 'C:\\trace', 'a//b', './trace', '__proto__'])(
    'rejects unsafe ZIP paths: %s', path => {
      expect(() => open(fixture({}, trace(), entries => { entries[path] = strToU8('{}'); }))).toThrow(/path/);
    },
  );

  it('rejects unsafe and duplicate manifest paths', () => {
    const buffer = fixture();
    const source = open(buffer);
    return source.readManifest().then(manifest => {
      expect(() => validateManifest({ ...manifest, entries: [...manifest.entries, manifest.entries[0]] })).toThrow('Duplicate');
      expect(() => validateManifest({ ...manifest, entries: [{ path: '../trace', encoding: 'json', uncompressedSize: 0 }] })).toThrow('path');
      source.close();
    });
  });

  it('rejects duplicate ZIP names even when central and local names agree', () => {
    const buffer = fixture({}, trace(), (entries, manifest) => {
      entries['another.json'] = strToU8('[]');
      manifest.entries.push({ path: 'another.json', encoding: 'json', uncompressedSize: 2 });
    });
    const view = new DataView(buffer);
    const offsets = centralOffsets(buffer);
    const replacement = strToU8('shaders.json');
    const extra = offsets.find(offset =>
      new TextDecoder().decode(new Uint8Array(buffer, offset + 46, view.getUint16(offset + 28, true))) === 'another.json')!;
    const local = view.getUint32(extra + 42, true);
    new Uint8Array(buffer, extra + 46, replacement.length).set(replacement);
    new Uint8Array(buffer, local + 30, replacement.length).set(replacement);
    expect(() => open(buffer)).toThrow('duplicate');
  });

  it('rejects truncated archives and corrupted payload CRCs', () => {
    const buffer = fixture();
    expect(() => open(buffer.slice(0, -1))).toThrow();
    expect(() => open(new ArrayBuffer(4))).toThrow();
    const view = new DataView(buffer);
    const offset = centralOffsets(buffer)[0];
    const local = view.getUint32(offset + 42, true);
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    new Uint8Array(buffer)[start] ^= 1;
    expect(() => open(buffer)).toThrow('Corrupt ZIP entry');
  });

  it('rejects excessive ZIP entry count and oversized ZIP entries before extraction', () => {
    let buffer = fixture();
    let view = new DataView(buffer);
    const end = buffer.byteLength - 22;
    view.setUint16(end + 8, LIMITS.entries + 1, true);
    view.setUint16(end + 10, LIMITS.entries + 1, true);
    expect(() => open(buffer)).toThrow('count');
    buffer = fixture();
    view = new DataView(buffer);
    const offset = centralOffsets(buffer)[0];
    view.setUint32(offset + 20, LIMITS.entry + 1, true);
    view.setUint32(offset + 24, LIMITS.entry + 1, true);
    expect(() => open(buffer)).toThrow('size limit');
  });

  it('rejects missing and undeclared payloads', () => {
    expect(() => open(fixture({}, trace(), entries => { delete entries['shaders.json']; }))).toThrow('Missing entry');
    expect(() => open(fixture({}, trace(), entries => { entries['extra.json'] = strToU8('{}'); }))).toThrow('undeclared');
  });

  it('rejects invalid encodings, decoded size limits and size mismatches', () => {
    expect(() => open(fixture({}, trace(), (_, manifest) => {
      manifest.entries[0].encoding = 'json';
    }))).toThrow();
    expect(() => open(fixture({}, trace(), (_, manifest) => {
      manifest.entries[0].uncompressedSize = LIMITS.entry + 1;
    }))).toThrow('limit');
    expect(() => open(fixture({}, trace(), (_, manifest) => {
      manifest.entries.push({ path: 'extra', encoding: 'json', uncompressedSize: LIMITS.entry });
      manifest.entries[0].uncompressedSize = LIMITS.entry;
    }))).toThrow('total');
    expect(() => open(fixture({}, trace(), (_, manifest) => {
      manifest.entries[1].uncompressedSize++;
    }))).toThrow('size disagrees');
  });

  it('rejects corrupted, truncated, concatenated and oversized-window zstd frames', async () => {
    for (const payload of [
      new Uint8Array([0, 0, 0, 0, 0, 0]),
      new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, 0, 0xf8]),
    ]) {
      const source = open(fixture({}, trace(), entries => { entries['trace.json.zst'] = payload; }));
      await expect(loadProfile(source)).rejects.toThrow();
      await expect(source.readManifest()).rejects.toThrow('closed');
    }
    for (const mode of ['truncated', 'concatenated']) {
      const source = open(fixture({}, trace(), entries => {
        const data = entries['trace.json.zst'];
        const both = new Uint8Array(data.length * 2);
        both.set(data);
        both.set(data, data.length);
        entries['trace.json.zst'] = mode === 'truncated' ? data.slice(0, -1) : both;
      }));
      await expect(loadProfile(source)).rejects.toThrow();
    }
    const mismatch = open(fixture({}, trace(), (_, manifest) => { manifest.entries[0].uncompressedSize--; }));
    await expect(loadProfile(mismatch)).rejects.toThrow('size disagrees');
  });

  it('bounds actual zstd output even when a frame omits its content size', async () => {
    const payload = new Uint8Array([0x28, 0xb5, 0x2f, 0xfd, 0, 0, 0x19, 0, 0, 65, 66, 67]);
    const source = open(fixture({}, trace(), (entries, manifest) => {
      entries['trace.json.zst'] = payload;
      manifest.entries[0].uncompressedSize = 1;
    }));
    await expect(loadProfile(source)).rejects.toThrow('exceeds declared');
  });
});

describe('transport-independent loading and presentation', () => {
  const code = {
    id: 1, pipelineId: 1, pso: '00000000000000020000000000000001',
    stage: 'compute', simdWidth: 8, version: 1, tier: 2, cache: 'hit',
  };
  function codeFixture(metadata: unknown = { schemaVersion: 1, codeVersions: [code] },
    args: Record<string, unknown> = { ...code, codeVersionId: 1 }) {
    return fixture({}, trace([span({ args })]), (entries, manifest) => {
      entries['shaders.json'] = strToU8(JSON.stringify(metadata));
      manifest.entries[1].uncompressedSize = entries['shaders.json'].length;
    });
  }

  it('resolves compute entry versions and preserves the exact exported trace', async () => {
    const profile = await loadProfile(open(codeFixture()));
    expect(profile.shaders.codeVersions).toEqual([code]);
    expect(profile.trace.traceEvents[0].args.codeVersionId).toBe(1);
    expect(JSON.parse(new TextDecoder().decode(profile.traceBytes))).toEqual(profile.trace);
    expect((await loadProfile(open(fixture()))).shaders.codeVersions).toEqual([]);
  });

  it.each([
    { schemaVersion: 3, codeVersions: [code] },
    { schemaVersion: 1, codeVersions: [{ ...code, id: 2 }] },
    { schemaVersion: 1, codeVersions: [{ ...code, pipelineId: 0 }] },
    { schemaVersion: 1, codeVersions: [{ ...code, tier: 0 }] },
    { schemaVersion: 1, codeVersions: [{ ...code, tier: 1 }] },
    { schemaVersion: 1, codeVersions: [{ ...code, pso: '/private/shader' }] },
    { schemaVersion: 1, codeVersions: [{ ...code, version: 2 }] },
    { schemaVersion: 1, codeVersions: [code, code] },
    { schemaVersion: 1, codeVersions: [code, { ...code, id: 2 }] },
    { schemaVersion: 1, codeVersions: [code,
      { ...code, id: 2, version: 2, cache: 'miss', simdWidth: 4 }] },
  ])('rejects invalid versioned compute metadata %j', async metadata => {
    await expect(loadProfile(open(codeFixture(metadata)))).rejects.toThrow(/metadata/);
  });

  it('accepts schema-v2 mixed-stage versions while preserving legacy compute readers', async () => {
    const vertex = { ...code, stage: 'vertex', entryRole: 'vertex', shaderId: 'a'.repeat(64) };
    const pixel = { ...vertex, id: 2, stage: 'pixel', entryRole: 'pixel',
      tier: 1, cache: 'hit', shaderId: 'b'.repeat(64) };
    const metadata = { schemaVersion: 2, codeVersions: [vertex, pixel, { ...code, id: 3, pipelineId: 2 }] };
    const loaded = await loadProfile(open(codeFixture(metadata, { ...vertex, codeVersionId: 1 })));
    expect(loaded.shaders.codeVersions).toEqual(metadata.codeVersions);
    expect(loaded.shaders.schemaVersion).toBe(2);
    expect((await loadProfile(open(codeFixture()))).shaders.schemaVersion).toBe(1);
  });

  it('resolves composite selections and distinguishes hull entry roles with the same version', async () => {
    const control = { ...code, stage: 'hull', entryRole: 'hull-control-point' };
    const patch = { ...control, id: 2, entryRole: 'hull-patch-constant' };
    const metadata = { schemaVersion: 2, codeVersions: [control, patch],
      codeSelections: [{ id: 1, pipelineId: 1, codeVersionIds: [1, 2] }] };
    const loaded = await loadProfile(open(codeFixture(metadata,
      { ...control, codeVersionId: 1, codeSelectionId: 1 })));
    expect(loaded.shaders.codeSelections).toEqual(metadata.codeSelections);
    expect(loaded.shaders.codeVersions[1].entryRole).toBe('hull-patch-constant');
  });

  it.each([
    { id: 2, pipelineId: 1, codeVersionIds: [1] },
    { id: 1, pipelineId: 2, codeVersionIds: [1] },
    { id: 1, pipelineId: 1, codeVersionIds: [0] },
    { id: 1, pipelineId: 1, codeVersionIds: [1, 1] },
  ])('rejects malformed composite selections %j', async selection => {
    const vertex = { ...code, stage: 'vertex', entryRole: 'vertex' };
    await expect(loadProfile(open(codeFixture({ schemaVersion: 2,
      codeVersions: [vertex], codeSelections: [selection] }, { ...vertex, codeVersionId: 1 }))))
      .rejects.toThrow(/selection/);
  });

  it.each([
    { stage: 'pixel' },
    { stage: 'pixel', entryRole: 'vertex' },
    { stage: 'pixel', entryRole: 'pixel', shaderId: '/private/shader.hlsl' },
    { stage: 'pixel', entryRole: 'pixel', artifactId: 'pointer-0x1234' },
    { stage: 'pixel', entryRole: 'pixel', version: 0 },
    { stage: 'pixel', entryRole: 'pixel', tier: 3 },
  ])('rejects invalid schema-v2 stage metadata %j', async fields => {
    const invalid = { ...code, ...fields };
    await expect(loadProfile(open(codeFixture({ schemaVersion: 2, codeVersions: [invalid] }))))
      .rejects.toThrow(/metadata/);
  });

  it.each([
    { ...code, codeVersionId: 0 }, { ...code, codeVersionId: 2 },
    { ...code, codeVersionId: 1, pipelineId: 2 }, { ...code, codeVersionId: 1, tier: 1 },
  ])('rejects dangling or inconsistent code references %j', async args => {
    await expect(loadProfile(open(codeFixture(undefined, args)))).rejects.toThrow(/code.version/);
  });

  it('accepts empty captures and incomplete captures without assuming event-counter semantics', async () => {
    const profile = await loadProfile(open(fixture({ complete: false, droppedEvents: 7 }, trace([]))));
    expect(groupThreads(profile.trace)).toEqual([]);
    expect(captureStatus(profile)).toMatch(/Incomplete.*7 dropped/);
    expect(summaryText(profile)).toContain('Recorded events: 1');
    expect(summaryText(profile)).toContain('Trace events: 0');
    expect(captureStatus({ ...profile, manifest: { ...profile.manifest, complete: true } })).toContain('data loss');
  });

  it('groups spans by both process and thread and sorts by timestamp without converting units', () => {
    const result = groupThreads(trace([
      span({ ts: 20 }), span({ pid: 2, ts: 3 }), span({ ts: 2 }),
      span({ ph: 'M', name: 'thread_name', args: { name: '<worker>' } }),
    ]));
    expect(result).toHaveLength(2);
    expect(result[0].name).toBe('<worker>');
    expect(result[0].spans.map(event => event.ts)).toEqual([2, 20]);
    expect(result[1].pid).toBe(2);
  });

  it.each([
    { displayTimeUnit: 'ns', traceEvents: [] },
    trace([span({ dur: -1 })]),
    trace([span({ ts: NaN })]),
    trace([span({ args: null as never })]),
    trace([span({ ph: 'B' as never })]),
  ])('rejects invalid trace data %j', async document => {
    await expect(loadProfile(open(fixture({}, document)))).rejects.toThrow();
  });

  it('closes any transport on failure', async () => {
    const source: ProfileDataSource = {
      readManifest: async () => { throw new Error('transport failed'); },
      readEntry: vi.fn(),
      close: vi.fn(),
    };
    await expect(loadProfile(source)).rejects.toThrow('transport failed');
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('rejects malformed shader JSON', async () => {
    const source = open(fixture({}, trace(), entries => { entries['shaders.json'] = strToU8('no'); }));
    await expect(loadProfile(source)).rejects.toThrow();
  });
});
