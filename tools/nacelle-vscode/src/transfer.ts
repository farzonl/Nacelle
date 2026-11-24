import { EXPORT_CHUNK_BYTES, MAX_EXPORT_BYTES } from '../../nacelle-view/src/host-protocol.ts';

export class ExportBuffer {
  readonly bytes: Uint8Array;
  offset = 0;
  private finished = false;

  constructor(size: number) {
    if (!Number.isSafeInteger(size) || size < 0 || size > MAX_EXPORT_BYTES) {
      throw new Error('Export exceeds the 256 MiB limit.');
    }
    this.bytes = new Uint8Array(size);
  }

  append(offset: number, encoded: string): number {
    if (this.finished || offset !== this.offset || encoded.length > Math.ceil(EXPORT_CHUNK_BYTES / 3) * 4) {
      throw new Error('Export chunk is stale, oversized, or out of order.');
    }
    const bytes = Buffer.from(encoded, 'base64');
    if (!bytes.length || bytes.length > EXPORT_CHUNK_BYTES || bytes.toString('base64') !== encoded ||
        offset + bytes.length > this.bytes.length) {
      throw new Error('Export chunk is invalid or exceeds declared size.');
    }
    this.bytes.set(bytes, offset);
    this.offset += bytes.length;
    return this.offset;
  }

  finish(): Uint8Array {
    if (this.finished || this.offset !== this.bytes.length) throw new Error('Export is incomplete or already finished.');
    this.finished = true;
    return this.bytes;
  }
}
