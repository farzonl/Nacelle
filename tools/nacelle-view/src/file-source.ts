import { Decompress } from 'fzstd';
import xxhash from 'xxhash-wasm';
import {
  LIMITS, parseJson, requireValid, safePath, validateManifest,
  type ProfileDataSource, type ProfileManifest,
} from './profile';

interface ZipEntry {
  path: string;
  size: number;
  crc: number;
  start: number;
}

let checksum: ReturnType<typeof xxhash> | undefined;

const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0);
  return value >>> 0;
});

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of data) crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 255];
  return (crc ^ 0xffffffff) >>> 0;
}

// Inspect the directory before allowing the ZIP library to allocate entry buffers.
// The native writer uses stored ZIP entries; compression is inside trace.json.zst.
function inspectZip(bytes: Uint8Array): ZipEntry[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u16 = (offset: number) => view.getUint16(offset, true);
  const u32 = (offset: number) => view.getUint32(offset, true);
  let end = bytes.length - 22;
  const minimum = Math.max(0, end - 65535);
  while (end >= minimum && (u32(end) !== 0x06054b50 || end + 22 + u16(end + 20) !== bytes.length)) end--;
  requireValid(end >= minimum, 'Corrupt ZIP: missing end directory.');
  const count = u16(end + 10);
  requireValid(u16(end + 4) === 0 && u16(end + 6) === 0 && u16(end + 8) === count,
    'Multi-disk ZIP archives are not supported.');
  requireValid(count > 0 && count <= LIMITS.entries, 'ZIP entry count exceeds limit.');
  let cursor = u32(end + 16);
  const directory = cursor;
  requireValid(cursor + u32(end + 12) === end, 'Corrupt ZIP directory bounds.');
  const result: ZipEntry[] = [];
  const paths = new Set<string>();
  const ranges: [number, number][] = [];
  let total = 0;
  for (let i = 0; i < count; i++) {
    requireValid(cursor + 46 <= end && u32(cursor) === 0x02014b50, 'Corrupt ZIP directory entry.');
    const flags = u16(cursor + 8);
    const size = u32(cursor + 24);
    const nameSize = u16(cursor + 28);
    const next = cursor + 46 + nameSize + u16(cursor + 30) + u16(cursor + 32);
    requireValid(next <= end && u16(cursor + 34) === 0, 'Corrupt ZIP entry bounds.');
    requireValid((flags & ~0x808) === 0 && u16(cursor + 10) === 0
      && u32(cursor + 20) === size, 'Only unencrypted, stored ZIP entries are supported.');
    requireValid(size <= LIMITS.entry, 'ZIP entry exceeds size limit.');
    total += size;
    requireValid(total <= LIMITS.total, 'ZIP exceeds total size limit.');
    const nameBytes = bytes.subarray(cursor + 46, cursor + 46 + nameSize);
    const path = new TextDecoder('utf-8', { fatal: true }).decode(nameBytes);
    requireValid(safePath(path) && !paths.has(path), 'Unsafe or duplicate ZIP entry path.');
    paths.add(path);
    const local = u32(cursor + 42);
    requireValid(local + 30 <= directory && u32(local) === 0x04034b50
      && u16(local + 6) === flags && u16(local + 8) === 0
      && u16(local + 26) === nameSize, 'Corrupt ZIP local header.');
    requireValid(nameBytes.every((byte, index) => byte === bytes[local + 30 + index]),
      'ZIP entry names disagree.');
    const dataEnd = local + 30 + nameSize + u16(local + 28) + size;
    requireValid(dataEnd <= directory, 'Corrupt ZIP entry data bounds.');
    const crc = u32(cursor + 16);
    let rangeEnd = dataEnd;
    if (flags & 8) {
      requireValid((u32(local + 14) === 0 && u32(local + 18) === 0 && u32(local + 22) === 0)
        || (u32(local + 14) === crc && u32(local + 18) === size && u32(local + 22) === size),
      'ZIP local sizes or checksum conflict with data descriptor.');
      requireValid(dataEnd + 12 <= directory, 'Missing ZIP data descriptor.');
      const descriptor = u32(dataEnd) === 0x08074b50 ? dataEnd + 4 : dataEnd;
      requireValid(descriptor + 12 <= directory && u32(descriptor) === crc
        && u32(descriptor + 4) === size && u32(descriptor + 8) === size,
      'Corrupt ZIP data descriptor.');
      rangeEnd = descriptor + 12;
    } else {
      requireValid(u32(local + 14) === crc && u32(local + 18) === size && u32(local + 22) === size,
        'ZIP entry sizes or checksums disagree.');
    }
    requireValid(ranges.every(([start, stop]) => rangeEnd <= start || local >= stop),
      'Overlapping ZIP entries.');
    ranges.push([local, rangeEnd]);
    result.push({ path, size, crc, start: local + 30 + nameSize + u16(local + 28) });
    cursor = next;
  }
  requireValid(cursor === end, 'Corrupt ZIP directory size.');
  return result;
}

// Bound the decoder's window allocation and admit exactly one dictionary-free
// frame. Scan block boundaries so concatenated frames cannot bypass these limits.
function inspectZstd(data: Uint8Array, expected: number): void {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  requireValid(data.length >= 6 && view.getUint32(0, true) === 0xfd2fb528, 'Invalid zstd frame.');
  const flags = data[4];
  requireValid((flags & 0x1b) === 0, 'Unsupported zstd flags or dictionary.');
  const single = Boolean(flags & 0x20);
  let cursor = single ? 5 : 6;
  const sizeFlag = flags >>> 6;
  const sizeBytes = sizeFlag ? 2 ** sizeFlag : single ? 1 : 0;
  requireValid(cursor + sizeBytes <= data.length, 'Truncated zstd header.');
  let contentSize = 0;
  for (let i = 0; i < sizeBytes; i++) contentSize += data[cursor++] * 2 ** (8 * i);
  if (sizeFlag === 1) contentSize += 256;
  requireValid(!sizeBytes || contentSize === expected, 'Zstd content size disagrees with manifest.');
  const base = 2 ** (10 + (data[5] >>> 3));
  const windowSize = single ? contentSize : base + base / 8 * (data[5] & 7);
  requireValid(windowSize <= LIMITS.entry, 'Zstd window exceeds size limit.');
  let last = false;
  while (!last) {
    requireValid(cursor + 3 <= data.length, 'Truncated zstd block.');
    const header = data[cursor] | (data[cursor + 1] << 8) | (data[cursor + 2] << 16);
    cursor += 3;
    last = Boolean(header & 1);
    const type = (header >>> 1) & 3;
    requireValid(type !== 3 && (header >>> 3) <= 131072, 'Invalid zstd block.');
    cursor += type === 1 ? 1 : header >>> 3;
    requireValid(cursor <= data.length, 'Truncated zstd block data.');
  }
  requireValid(cursor + ((flags & 4) ? 4 : 0) === data.length,
    'Truncated or concatenated zstd frames are not supported.');
}

async function decodeZstd(data: Uint8Array, expected: number): Promise<Uint8Array> {
  inspectZstd(data, expected);
  const result = new Uint8Array(expected);
  let written = 0;
  const decoder = new Decompress(chunk => {
    requireValid(written + chunk.length <= expected, 'Zstd output exceeds declared size.');
    result.set(chunk, written);
    written += chunk.length;
  });
  decoder.push(data, true);
  requireValid(written === expected, 'Zstd decoded size disagrees with manifest.');
  if (data[4] & 4) {
    const hash = await (checksum ??= xxhash());
    const expectedChecksum = new DataView(data.buffer, data.byteOffset + data.length - 4, 4).getUint32(0, true);
    requireValid(Number(hash.h64Raw(result) & 0xffffffffn) === expectedChecksum, 'Corrupt zstd checksum.');
  }
  return result;
}

export class FileProfileDataSource implements ProfileDataSource {
  private entries: Record<string, Uint8Array> | null;
  private readonly manifest: ProfileManifest;

  private constructor(bytes: Uint8Array) {
    requireValid(bytes.length >= 22 && bytes.length <= LIMITS.archive, 'Archive exceeds size limit or is truncated.');
    const directory = inspectZip(bytes);
    const metadata = directory.find(entry => entry.path === 'manifest.json');
    requireValid(metadata && metadata.size <= LIMITS.manifest, 'Missing or oversized manifest.json.');
    this.entries = Object.create(null) as Record<string, Uint8Array>;
    for (const entry of directory) {
      const data = bytes.subarray(entry.start, entry.start + entry.size);
      requireValid(crc32(data) === entry.crc,
        `Corrupt ZIP entry: ${entry.path}.`);
      this.entries[entry.path] = data;
    }
    this.manifest = validateManifest(parseJson(this.entries['manifest.json']));
    const declared = new Set(this.manifest.entries.map(entry => entry.path));
    requireValid(directory.every(entry => entry.path === 'manifest.json' || declared.has(entry.path)),
      'ZIP contains undeclared entries.');
    for (const entry of this.manifest.entries) {
      const raw = this.entries[entry.path];
      requireValid(raw !== undefined, `Missing entry: ${entry.path}.`);
      requireValid(entry.encoding !== 'json' || raw.length === entry.uncompressedSize,
        `Entry size disagrees with manifest: ${entry.path}.`);
    }
  }

  static async open(file: Pick<File, 'size' | 'arrayBuffer'>): Promise<FileProfileDataSource> {
    requireValid(file.size <= LIMITS.archive, 'Archive exceeds 256 MiB limit.');
    return FileProfileDataSource.fromBuffer(await file.arrayBuffer());
  }

  static fromBuffer(buffer: ArrayBuffer): FileProfileDataSource {
    return new FileProfileDataSource(new Uint8Array(buffer));
  }

  async readManifest(): Promise<ProfileManifest> {
    requireValid(this.entries, 'Profile data source is closed.');
    return structuredClone(this.manifest);
  }

  async readEntry(path: string): Promise<ArrayBuffer> {
    requireValid(this.entries, 'Profile data source is closed.');
    const entry = this.manifest.entries.find(candidate => candidate.path === path);
    requireValid(entry, 'Entry is not declared in manifest.');
    const raw = this.entries[path];
    if (entry.encoding === 'zstd') return (await decodeZstd(raw, entry.uncompressedSize)).buffer as ArrayBuffer;
    return raw.slice().buffer as ArrayBuffer;
  }

  close(): void {
    this.entries = null;
  }
}
