import { LIMITS } from './profile';

export const HOST_VERSION = 1;
export const EXPORT_CHUNK_BYTES = 64 * 1024;
export const MAX_EXPORT_BYTES = LIMITS.entry;

export type ExportName = 'nacelle-trace.json' | 'nacelle-speedscope.json';

export interface DesktopCaptureMeta {
  id: number;
  name: string;
  size: number;
}

export type DesktopMenuAction = 'open' | 'reload' | 'export-trace' | 'export-speedscope';

export interface DesktopHostBridge {
  readonly version: 1;
  getInitialCapture(): Promise<DesktopCaptureMeta | null>;
  openDialog(): Promise<DesktopCaptureMeta | null>;
  reloadCapture(): Promise<DesktopCaptureMeta>;
  readChunk(id: number, offset: number): Promise<ArrayBuffer>;
  cancelRead(id: number): Promise<void>;
  exportStart(name: ExportName, size: number): Promise<number>;
  exportChunk(id: number, offset: number, bytes: Uint8Array): Promise<number>;
  exportFinish(id: number): Promise<void>;
  exportCancel(id: number): Promise<void>;
  onOpenCapture(listener: (capture: DesktopCaptureMeta) => void): Promise<() => void>;
  onMenuAction(listener: (action: DesktopMenuAction) => void): Promise<() => void>;
  onHostError?(listener: (message: string) => void): Promise<() => void>;
  reportSmokeEvent?(payload: Record<string, unknown>): Promise<void>;
}

declare global {
  interface Window {
    __NACELLE_DESKTOP_HOST__?: DesktopHostBridge;
  }
}
export type ExportMessage =
  | { version: 1; kind: 'export-start'; id: number; size: number; name: ExportName }
  | { version: 1; kind: 'export-chunk'; id: number; offset: number; data: string }
  | { version: 1; kind: 'export-end' | 'export-cancel'; id: number };
export interface ExportReply {
  version: 1;
  kind: 'export-reply';
  id: number;
  offset?: number;
  error?: string;
}
export type ReadMessage =
  | { version: 1; kind: 'read-start'; id: number }
  | { version: 1; kind: 'read-cancel'; id: number }
  | { version: 1; kind: 'read-chunk'; id: number; offset: number };

export function readMessage(value: unknown): ReadMessage {
  if (!isRecord(value) || value.version !== HOST_VERSION ||
      !Number.isSafeInteger(value.id) || Number(value.id) < 1) throw new Error('Invalid read identity.');
  const keys = value.kind === 'read-chunk' ? ['version', 'kind', 'id', 'offset'] : ['version', 'kind', 'id'];
  if (Object.keys(value).some(key => !keys.includes(key))) throw new Error('A read message cannot select a path or extra fields.');
  if (value.kind === 'read-start' || value.kind === 'read-cancel') {
    return { version: 1, kind: value.kind, id: Number(value.id) };
  }
  if (value.kind === 'read-chunk' && Number.isSafeInteger(value.offset) && Number(value.offset) >= 0) {
    return { version: 1, kind: value.kind, id: Number(value.id), offset: Number(value.offset) };
  }
  throw new Error('Invalid read message.');
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function exportMessage(value: unknown): ExportMessage {
  if (!isRecord(value) || value.version !== HOST_VERSION ||
      !Number.isSafeInteger(value.id) || Number(value.id) < 1) throw new Error('Invalid export message identity.');
  const keys = value.kind === 'export-start' ? ['version', 'kind', 'id', 'size', 'name'] :
    value.kind === 'export-chunk' ? ['version', 'kind', 'id', 'offset', 'data'] : ['version', 'kind', 'id'];
  if (Object.keys(value).some(key => !keys.includes(key))) throw new Error('Unexpected export message fields.');
  if (value.kind === 'export-start' && Number.isSafeInteger(value.size) &&
      Number(value.size) >= 0 && Number(value.size) <= MAX_EXPORT_BYTES &&
      (value.name === 'nacelle-trace.json' || value.name === 'nacelle-speedscope.json')) {
    return { version: 1, kind: value.kind, id: Number(value.id), size: Number(value.size), name: value.name };
  }
  if (value.kind === 'export-chunk' && Number.isSafeInteger(value.offset) &&
      Number(value.offset) >= 0 && typeof value.data === 'string' &&
      value.data.length > 0 && value.data.length <= Math.ceil(EXPORT_CHUNK_BYTES / 3) * 4) {
    return { version: 1, kind: value.kind, id: Number(value.id), offset: Number(value.offset), data: value.data };
  }
  if (value.kind === 'export-end' || value.kind === 'export-cancel') {
    return { version: 1, kind: value.kind, id: Number(value.id) };
  }
  throw new Error('Invalid or oversized export message.');
}
