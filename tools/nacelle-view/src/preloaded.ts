import { FileProfileDataSource } from './file-source';
import { LIMITS, object, requireValid, type ProfileDataSource } from './profile';

export interface PreloadedProfileEntry {
  id: string;
  name: string;
  category: string;
  file: string;
  simdWidth?: number;
  sizeBytes?: number;
}

export interface PreloadedProfileManifest {
  schemaVersion: 1;
  generatedAt?: string;
  commitSha?: string;
  shortSha?: string;
  os?: string;
  simdWidth?: number;
  profiles: PreloadedProfileEntry[];
}

const SAFE_PROFILE_FILE = /^[A-Za-z0-9_.-]+\.nacelle-profile$/;

export function isSafeProfileFile(file: string): boolean {
  return file.length <= 128 && SAFE_PROFILE_FILE.test(file) && !file.includes('..');
}

export function validatePreloadedManifest(value: unknown): PreloadedProfileManifest {
  requireValid(object(value) && value.schemaVersion === 1 && Array.isArray(value.profiles),
    'Invalid benchmark profile manifest.');
  const seenIds = new Set<string>();
  const profiles: PreloadedProfileEntry[] = [];
  for (const item of value.profiles) {
    requireValid(
      object(item)
        && typeof item.id === 'string' && item.id.length > 0 && item.id.length <= 128
        && typeof item.name === 'string' && item.name.length > 0
        && typeof item.category === 'string' && item.category.length > 0
        && typeof item.file === 'string' && isSafeProfileFile(item.file),
      'Invalid benchmark profile entry in manifest.',
    );
    requireValid(!seenIds.has(item.id), `Duplicate benchmark profile id: ${item.id}`);
    seenIds.add(item.id);
    profiles.push({
      id: item.id,
      name: item.name,
      category: item.category,
      file: item.file,
      simdWidth: typeof item.simdWidth === 'number' ? item.simdWidth : undefined,
      sizeBytes: typeof item.sizeBytes === 'number' ? item.sizeBytes : undefined,
    });
  }
  return {
    schemaVersion: 1,
    generatedAt: typeof value.generatedAt === 'string' ? value.generatedAt : undefined,
    commitSha: typeof value.commitSha === 'string' ? value.commitSha : undefined,
    shortSha: typeof value.shortSha === 'string' ? value.shortSha : undefined,
    os: typeof value.os === 'string' ? value.os : undefined,
    simdWidth: typeof value.simdWidth === 'number' ? value.simdWidth : undefined,
    profiles,
  };
}

export async function fetchPreloadedManifest(url = './profiles/manifest.json'): Promise<PreloadedProfileManifest> {
  const response = await fetch(url, { cache: 'no-cache' });
  requireValid(response.ok, `Unable to load benchmark profile index (${response.status}).`);
  return validatePreloadedManifest(await response.json());
}

export async function openPreloadedProfile(file: string): Promise<ProfileDataSource> {
  requireValid(isSafeProfileFile(file), 'Invalid preloaded profile filename.');
  const response = await fetch(`./profiles/${encodeURIComponent(file)}`);
  requireValid(response.ok, `Unable to load benchmark profile ${file} (${response.status}).`);
  const buffer = await response.arrayBuffer();
  requireValid(buffer.byteLength <= LIMITS.archive, 'Benchmark profile exceeds 256 MiB limit.');
  return FileProfileDataSource.fromBuffer(buffer);
}
