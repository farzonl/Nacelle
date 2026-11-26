import InlineDecoder from './profile-worker?worker&inline';
import { EXPORT_CHUNK_BYTES, HOST_VERSION, MAX_EXPORT_BYTES, isRecord, type ExportMessage } from './host-protocol';
import { LIMITS, requireValid } from './profile';
import { createSwimlaneTimelineView } from './swimlane-timeline';
import { mountViewer } from './ui';
import { decodeProfileFile } from './worker-loader';

interface VsCodeApi {
  postMessage(message: unknown): void;
}
declare global {
  interface Window {
    acquireVsCodeApi?: () => VsCodeApi;
  }
}

export function mountVSCodeViewer(): void {
  requireValid(window.acquireVsCodeApi, 'VS Code API is unavailable.');
  const api = window.acquireVsCodeApi();
  const config: unknown = JSON.parse(document.querySelector('#nacelle-host-config')?.textContent ?? '');
  requireValid(isRecord(config) &&
    typeof config.name === 'string' && typeof config.frameHtml === 'string' &&
    typeof config.nonce === 'string' && /^[a-f0-9]{32}$/.test(config.nonce), 'Invalid VS Code viewer configuration.');
  const name = config.name;
  let nextId = 0;
  let exporting = false;
  let loadGeneration = 0;
  let readPending: { id: number; resolve: (reply: Record<string, unknown>) => void; reject: (error: Error) => void; timer: number } | undefined;
  let pending: { id: number; offset: number; resolve: () => void; reject: (error: Error) => void; timer: number } | undefined;

  function request(message: ExportMessage, offset: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        pending = undefined;
        reject(new Error('Export acknowledgement timed out.'));
      }, 120000);
      pending = { id: message.id, offset, resolve, reject, timer };
      api.postMessage(message);
    });
  }

  function readRequest(message: unknown, id: number, signal: AbortSignal): Promise<Record<string, unknown>> {
    signal.throwIfAborted();
    return new Promise((resolve, reject) => {
      const abort = () => {
        if (readPending?.id === id) {
          clearTimeout(readPending.timer);
          readPending = undefined;
        }
        api.postMessage({ version: 1, kind: 'read-cancel', id });
        reject(new Error('Profile loading cancelled.'));
      };
      signal.addEventListener('abort', abort, { once: true });
      const timer = window.setTimeout(() => {
        signal.removeEventListener('abort', abort);
        readPending = undefined;
        reject(new Error('Profile read acknowledgement timed out.'));
      }, 120000);
      readPending = {
        id, timer,
        resolve: reply => { signal.removeEventListener('abort', abort); resolve(reply); },
        reject: error => { signal.removeEventListener('abort', abort); reject(error); },
      };
      api.postMessage(message);
    });
  }

  async function saveFile(blob: Blob, filename: string): Promise<void> {
    requireValid(!exporting, 'Another export is in progress.');
    requireValid(blob.size <= MAX_EXPORT_BYTES &&
      (filename === 'nacelle-trace.json' || filename === 'nacelle-speedscope.json'), 'Invalid or oversized export.');
    const id = ++nextId;
    exporting = true;
    try {
      await request({ version: 1, kind: 'export-start', id, size: blob.size, name: filename }, 0);
      for (let offset = 0; offset < blob.size; offset += EXPORT_CHUNK_BYTES) {
        const bytes = new Uint8Array(await blob.slice(offset, offset + EXPORT_CHUNK_BYTES).arrayBuffer());
        let text = '';
        for (const byte of bytes) text += String.fromCharCode(byte);
        await request({ version: 1, kind: 'export-chunk', id, offset, data: btoa(text) }, offset + bytes.length);
      }
      await request({ version: 1, kind: 'export-end', id }, blob.size);
    } catch (error) {
      api.postMessage({ version: 1, kind: 'export-cancel', id });
      throw error;
    } finally {
      exporting = false;
    }
  }

  const viewer = mountViewer(document, undefined, {
    decodeFile: (file, signal) => decodeProfileFile(file, signal, () => new InlineDecoder()),
    saveFile,
    timelineView: createSwimlaneTimelineView(),
  });
  document.querySelector<HTMLInputElement>('#profile-file')!.disabled = true;
  document.querySelector<HTMLElement>('#dropzone')!.hidden = true;
  document.querySelectorAll('nav a').forEach(link => link.remove());

  async function load(): Promise<void> {
    const generation = ++loadGeneration;
    let sourceId = 0;
    await viewer.openSource(name, async signal => {
      const id = ++nextId;
      sourceId = id;
      try {
        const start = await readRequest({ version: 1, kind: 'read-start', id }, id, signal);
        requireValid(Number.isSafeInteger(start.size) && Number(start.size) >= 22 &&
          Number(start.size) <= LIMITS.archive, 'Selected capture is truncated or exceeds 256 MiB.');
        const bytes = new Uint8Array(Number(start.size));
        for (let offset = 0; offset < bytes.length;) {
          const reply = await readRequest({ version: 1, kind: 'read-chunk', id, offset }, id, signal);
          requireValid(reply.offset === offset && typeof reply.data === 'string' &&
            reply.data.length <= Math.ceil(EXPORT_CHUNK_BYTES / 3) * 4, 'Invalid read chunk.');
          const text = atob(reply.data);
          requireValid(text.length > 0 && text.length <= EXPORT_CHUNK_BYTES &&
            offset + text.length <= bytes.length, 'Read chunk exceeds the selected capture.');
          for (let index = 0; index < text.length; ++index) bytes[offset + index] = text.charCodeAt(index);
          offset += text.length;
        }
        return new Blob([bytes]);
      } finally {
        api.postMessage({ version: 1, kind: 'read-cancel', id });
      }
    });
    if (generation !== loadGeneration) return;
    const status = document.querySelector<HTMLElement>('#status')!;
    api.postMessage({ version: HOST_VERSION, kind: 'status', id: sourceId, state: status.dataset.state,
      message: status.textContent?.slice(0, 2048) });
  }

  function receive(event: MessageEvent<unknown>): void {
    const message = event.data;
    if (!isRecord(message) || message.version !== HOST_VERSION) return;
    if (message.kind === 'read-reply' && readPending && message.id === readPending.id) {
      const reply = readPending;
      readPending = undefined;
      clearTimeout(reply.timer);
      if (typeof message.error === 'string') reply.reject(new Error(message.error));
      else reply.resolve(message);
    } else if (message.kind === 'export-reply' && pending && message.id === pending.id) {
      const reply = pending;
      pending = undefined;
      clearTimeout(reply.timer);
      if (typeof message.error === 'string') reply.reject(new Error(message.error));
      else if (message.offset !== reply.offset) reply.reject(new Error('Export acknowledgement was out of order.'));
      else reply.resolve();
    } else if (message.kind === 'reload') {
      void load();
    } else if (message.kind === 'export-trace') {
      void viewer.exportTrace().catch(error => console.error('VS Code trace export failed.', error));
    }
  }
  window.addEventListener('message', receive);
  window.addEventListener('pagehide', () => {
    window.removeEventListener('message', receive);
    ++loadGeneration;
    if (pending) {
      clearTimeout(pending.timer);
      api.postMessage({ version: 1, kind: 'export-cancel', id: pending.id });
      pending.reject(new Error('Viewer closed.'));
      pending = undefined;
    }
    viewer.dispose();
  }, { once: true });
  void load();
}
