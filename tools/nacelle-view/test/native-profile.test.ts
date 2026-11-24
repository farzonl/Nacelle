import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { FileProfileDataSource } from '../src/file-source';
import { loadProfile } from '../src/profile';

it.runIf(Boolean(process.env.NACELLE_NATIVE_COUNTER_PROFILE))(
  'loads an actual counter-enabled runtime capture with exact code attribution and availability',
  async () => {
    const bytes = readFileSync(process.env.NACELLE_NATIVE_COUNTER_PROFILE!);
    const profile = await loadProfile(FileProfileDataSource.fromBuffer(bytes.buffer.slice(
      bytes.byteOffset, bytes.byteOffset + bytes.byteLength,
    )));
    expect(profile.manifest.schemaVersion).toBe(4);
    expect(profile.counters?.attempted).toBeGreaterThan(0);
    expect(profile.counters?.records.length).toBe(profile.counters?.recorded);
    expect(profile.manifest.complete).toBe(profile.counters?.complete);
    expect(profile.trace.traceEvents.length).toBe(profile.manifest.recordedEvents);
    for (const record of profile.counters!.records) {
      expect(record.codeVersionId).toBeGreaterThan(0);
      const event = profile.trace.traceEvents.find(event => event.args.id === record.eventId);
      expect(event?.args.codeVersionId).toBe(record.codeVersionId);
    }
  },
);

it.runIf(Boolean(process.env.NACELLE_NATIVE_PROFILE))(
  'loads and exports an actual native runtime capture',
  async () => {
    const bytes = readFileSync(process.env.NACELLE_NATIVE_PROFILE!);
    const profile = await loadProfile(FileProfileDataSource.fromBuffer(bytes.buffer.slice(
      bytes.byteOffset, bytes.byteOffset + bytes.byteLength,
    )));
    expect(profile.manifest.complete).toBe(true);
    expect(profile.trace.traceEvents.length).toBe(profile.manifest.recordedEvents);
    expect(profile.trace.traceEvents.some(event => event.name === 'Compute dispatch')).toBe(true);
    expect(profile.trace.traceEvents.some(event => event.name === 'Graphics draw')).toBe(true);
    if (process.env.NACELLE_NATIVE_CODE_VERSIONS) {
      expect(profile.shaders.codeVersions.length).toBeGreaterThan(0);
      for (const event of profile.trace.traceEvents.filter(event =>
        event.name === 'Compute dispatch' || event.name === 'Compute worker')) {
        expect(event.args.pipelineId).toBeGreaterThan(0);
        expect(event.args.codeVersionId).toBeGreaterThan(0);
      }
      expect(profile.shaders.schemaVersion).toBe(2);
      const draw = profile.trace.traceEvents.find(event => event.name === 'Graphics draw');
      expect(draw?.args.codeVersionId).toBeUndefined();
      const selection = profile.shaders.codeSelections?.[(draw?.args.codeSelectionId as number) - 1];
      expect(selection?.codeVersionIds.map(id => profile.shaders.codeVersions[id - 1].stage).sort())
        .toEqual(['pixel', 'vertex']);
    }
    if (process.env.NACELLE_NATIVE_SHADER_DIR) {
      const files = { compute: 'vec_add.dxil', vertex: 'raster_basic_vs.dxil', pixel: 'raster_basic_ps.dxil' };
      for (const code of profile.shaders.codeVersions) {
        if (code.stage !== 'compute' && code.stage !== 'vertex' && code.stage !== 'pixel')
          throw new Error(`Unexpected native fixture stage ${code.stage}`);
        const file = files[code.stage];
        const digest = createHash('sha256')
          .update(readFileSync(join(process.env.NACELLE_NATIVE_SHADER_DIR, file))).digest('hex');
        expect(code.shaderId).toBe(digest);
      }
    }
    expect(JSON.parse(new TextDecoder().decode(profile.traceBytes))).toEqual(profile.trace);
  },
);

it.runIf(Boolean(process.env.NACELLE_NATIVE_ARTIFACT_PROFILE))(
  'loads real opt-in DXC source and cached native artifacts through the typed data-source boundary',
  async () => {
    const bytes = readFileSync(process.env.NACELLE_NATIVE_ARTIFACT_PROFILE!);
    const profile = await loadProfile(FileProfileDataSource.fromBuffer(bytes.buffer.slice(
      bytes.byteOffset, bytes.byteOffset + bytes.byteLength,
    )));
    expect(profile.manifest.schemaVersion).toBe(2);
    expect(profile.artifacts?.payloads.some(payload => payload.kind === 'hlsl')).toBe(true);
    expect(profile.artifacts?.payloads.some(payload => payload.kind === 'dxil' && payload.mappings.length > 0)).toBe(true);
    expect(profile.artifacts?.payloads.some(payload => payload.kind === 'assembly')).toBe(true);
    expect(profile.artifacts?.selections.every(selection => selection.nativeAvailability === 'available')).toBe(true);
  },
);
