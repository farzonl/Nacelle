import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { FileProfileDataSource } from '../src/file-source';
import { loadProfile } from '../src/profile';

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
    expect(JSON.parse(new TextDecoder().decode(profile.traceBytes))).toEqual(profile.trace);
  },
);
