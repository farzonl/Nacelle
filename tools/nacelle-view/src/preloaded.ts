import { FileProfileDataSource } from './file-source';
import { decodeProfileFile } from './worker-loader';
import { LIMITS, object, requireValid, type Profile, type ProfileDataSource } from './profile';

export interface PreloadedProfileEntry {
  id: string;
  name: string;
  category: string;
  file: string;
  simdWidth?: number;
  sizeBytes?: number;
}

export interface PreloadedPlatformManifest {
  os: string;
  generatedAt?: string;
  commitSha?: string;
  shortSha?: string;
  simdWidth?: number;
  profiles: PreloadedProfileEntry[];
}

export interface LegacyPreloadedProfileManifest {
  schemaVersion: 1;
  generatedAt?: string;
  commitSha?: string;
  shortSha?: string;
  os?: string;
  simdWidth?: number;
  profiles: PreloadedProfileEntry[];
}

export interface PreloadedProfileManifest {
  schemaVersion: 2;
  defaultPlatform?: string;
  platforms: Record<string, PreloadedPlatformManifest>;
}

export type PreloadedManifestInput = PreloadedProfileManifest | LegacyPreloadedProfileManifest;

const SAFE_PLATFORM_KEY = /^[a-z0-9_-]+$/;
const SAFE_PROFILE_FILE = /^(?:[a-z0-9_-]+\/)?[A-Za-z0-9_.-]+\.nacelle-profile$/;

export function isSafeProfileFile(file: string): boolean {
  return file.length <= 160 && SAFE_PROFILE_FILE.test(file) && !file.includes('..');
}

function validateProfileEntries(items: unknown[]): PreloadedProfileEntry[] {
  const seenIds = new Set<string>();
  const profiles: PreloadedProfileEntry[] = [];
  for (const item of items) {
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
  return profiles;
}

function normalizePlatformKey(osName: string | undefined): string {
  const raw = (osName ?? 'default').trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
  return raw || 'default';
}

export function validatePreloadedManifest(value: unknown): PreloadedProfileManifest {
  requireValid(object(value), 'Invalid benchmark profile manifest.');
  if (value.schemaVersion === 1) {
    requireValid(Array.isArray(value.profiles), 'Invalid benchmark profile manifest.');
    const osLabel = typeof value.os === 'string' && value.os ? value.os : 'macOS';
    const key = normalizePlatformKey(osLabel);
    const profiles = validateProfileEntries(value.profiles);
    return {
      schemaVersion: 2,
      defaultPlatform: key,
      platforms: {
        [key]: {
          os: osLabel,
          generatedAt: typeof value.generatedAt === 'string' ? value.generatedAt : undefined,
          commitSha: typeof value.commitSha === 'string' ? value.commitSha : undefined,
          shortSha: typeof value.shortSha === 'string' ? value.shortSha : undefined,
          simdWidth: typeof value.simdWidth === 'number' ? value.simdWidth : undefined,
          profiles,
        },
      },
    };
  }

  requireValid(value.schemaVersion === 2 && object(value.platforms), 'Invalid benchmark profile manifest.');
  const platforms: Record<string, PreloadedPlatformManifest> = {};
  for (const [key, plat] of Object.entries(value.platforms)) {
    requireValid(SAFE_PLATFORM_KEY.test(key) && object(plat) && Array.isArray(plat.profiles),
      `Invalid platform manifest entry: ${key}`);
    const profiles = validateProfileEntries(plat.profiles);
    if (!profiles.length) continue;
    platforms[key] = {
      os: typeof plat.os === 'string' && plat.os ? plat.os : key,
      generatedAt: typeof plat.generatedAt === 'string' ? plat.generatedAt : undefined,
      commitSha: typeof plat.commitSha === 'string' ? plat.commitSha : undefined,
      shortSha: typeof plat.shortSha === 'string' ? plat.shortSha : undefined,
      simdWidth: typeof plat.simdWidth === 'number' ? plat.simdWidth : undefined,
      profiles,
    };
  }

  const defaultPlatform = typeof value.defaultPlatform === 'string' && value.defaultPlatform in platforms
    ? value.defaultPlatform
    : Object.keys(platforms)[0];

  return {
    schemaVersion: 2,
    defaultPlatform,
    platforms,
  };
}

export async function fetchPreloadedManifest(url = './profiles/manifest.json'): Promise<PreloadedProfileManifest> {
  const response = await fetch(url, { cache: 'no-cache' });
  requireValid(response.ok, `Unable to load benchmark profile index (${response.status}).`);
  return validatePreloadedManifest(await response.json());
}

export async function openPreloadedProfile(file: string): Promise<ProfileDataSource>;
export async function openPreloadedProfile(file: string, signal: AbortSignal): Promise<Profile>;
export async function openPreloadedProfile(file: string, signal?: AbortSignal): Promise<ProfileDataSource | Profile> {
  requireValid(isSafeProfileFile(file), 'Invalid preloaded profile filename.');
  const encodedPath = file.split('/').map(encodeURIComponent).join('/');
  const response = await fetch(`./profiles/${encodedPath}`, { signal });
  requireValid(response.ok, `Unable to load benchmark profile ${file} (${response.status}).`);
  requireValid(response.body, 'Benchmark profile has no response body.');
  const reader = response.body.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.length;
      requireValid(size <= LIMITS.archive, 'Benchmark profile exceeds 256 MiB limit.');
      chunks.push(value);
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
  const blob = new Blob(chunks);
  if (signal) return decodeProfileFile(blob, signal);
  return FileProfileDataSource.fromBuffer(await blob.arrayBuffer());
}
