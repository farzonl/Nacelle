import InlineDecoder from './profile-worker?worker&inline';
import {
  EXPORT_CHUNK_BYTES,
  HOST_VERSION,
  MAX_EXPORT_BYTES,
  isRecord,
  type DesktopCaptureMeta,
  type DesktopHostBridge,
  type DesktopMenuAction,
} from './host-protocol';
import { LIMITS, requireValid } from './profile';
import { createSwimlaneTimelineView } from './swimlane-timeline';
import type { TimelineView } from './timeline';
import { mountViewer, type ViewerOptions } from './ui';
import { decodeProfileFile } from './worker-loader';

export interface DesktopViewerHandle {
  openDialog(): Promise<void>;
  reload(): Promise<void>;
  exportTrace(): Promise<void>;
  exportSpeedscope(): Promise<void>;
  dispose(): void;
}

export interface DesktopHostOptions {
  decodeFile?: ViewerOptions['decodeFile'];
  timelineView?: TimelineView;
}

function readBlobArrayBuffer(blob: Blob): Promise<ArrayBuffer> {
  if (typeof blob.arrayBuffer === 'function') return blob.arrayBuffer();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = () => reject(reader.error ?? new Error('Unable to read export blob.'));
    reader.readAsArrayBuffer(blob);
  });
}

function toChunkBytes(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer || Object.prototype.toString.call(value) === '[object ArrayBuffer]') {
    return new Uint8Array(value as ArrayBuffer);
  }
  throw new Error('Read chunk is not a binary buffer.');
}

function validateCaptureMeta(value: unknown): DesktopCaptureMeta {
  requireValid(isRecord(value) &&
    Number.isSafeInteger(value.id) && Number(value.id) >= 1 &&
    typeof value.name === 'string' && value.name.length > 0 && value.name.length <= 256 &&
    Number.isSafeInteger(value.size) && Number(value.size) >= 22 && Number(value.size) <= LIMITS.archive,
  'Selected capture is truncated, invalid, or exceeds 256 MiB.');
  return { id: Number(value.id), name: value.name, size: Number(value.size) };
}

export function mountDesktopViewer(
  host: DesktopHostBridge | undefined = window.__NACELLE_DESKTOP_HOST__,
  root: Document = document,
  options: DesktopHostOptions = {},
): DesktopViewerHandle {
  requireValid(host && host.version === HOST_VERSION, 'Desktop host bridge is unavailable.');
  const bridge = host;
  const win = root.defaultView;

  let disposed = false;
  let exporting = false;
  let activeExportId: number | undefined;
  let activeReadId: number | undefined;
  let loadGeneration = 0;
  const unsubscribers: (() => void)[] = [];

  const status = root.querySelector<HTMLElement>('#status')!;
  const modeBadge = root.querySelector<HTMLElement>('#mode-badge');
  if (modeBadge) modeBadge.textContent = 'Desktop · Offline';

  const subtitle = root.querySelector<HTMLElement>('header p');
  if (subtitle) {
    subtitle.textContent = 'Native offline .nacelle-profile trace & shader JIT inspector';
  }
  root.querySelectorAll('nav a').forEach(link => link.remove());

  const fileInput = root.querySelector<HTMLInputElement>('#profile-file');
  const dropzone = root.querySelector<HTMLElement>('#dropzone');
  const openButton = root.createElement('button');
  openButton.id = 'desktop-open';
  openButton.type = 'button';
  openButton.className = 'profiler-btn rounded-lg border border-[var(--border-strong)] bg-[var(--surface)] px-3.5 py-2 text-xs font-semibold text-[var(--text)] transition hover:border-[var(--accent)] hover:text-[var(--accent)]';
  openButton.textContent = 'Open .nacelle-profile…';

  if (fileInput) {
    fileInput.disabled = true;
    fileInput.hidden = true;
    fileInput.insertAdjacentElement('afterend', openButton);
  }
  if (dropzone) {
    const label = dropzone.querySelector<HTMLLabelElement>('label[for="profile-file"]');
    if (label) label.removeAttribute('for');
    const hint = dropzone.querySelector<HTMLElement>('p');
    if (hint) {
      hint.textContent = 'Drop a .nacelle-profile file onto this window or open via the native dialog. Offline only — no uploads, network requests, or telemetry. Profiles are not replayable.';
    }
  }

  function showHostError(message: string): void {
    status.dataset.state = 'invalid';
    status.textContent = message;
  }

  async function saveFile(blob: Blob, filename: string): Promise<void> {
    requireValid(!disposed, 'Desktop viewer was closed.');
    requireValid(!exporting, 'Another export is in progress.');
    requireValid(blob.size <= MAX_EXPORT_BYTES &&
      (filename === 'nacelle-trace.json' || filename === 'nacelle-speedscope.json'),
    'Invalid or oversized export.');

    exporting = true;
    let exportId: number | undefined;
    try {
      exportId = await bridge.exportStart(filename, blob.size);
      requireValid(Number.isSafeInteger(exportId) && exportId >= 1, 'Invalid export transaction.');
      activeExportId = exportId;

      for (let offset = 0; offset < blob.size; offset += EXPORT_CHUNK_BYTES) {
        requireValid(!disposed, 'Desktop viewer was closed.');
        const slice = new Uint8Array(await readBlobArrayBuffer(blob.slice(offset, offset + EXPORT_CHUNK_BYTES)));
        const nextOffset = await bridge.exportChunk(exportId, offset, slice);
        requireValid(nextOffset === offset + slice.byteLength, 'Export acknowledgement was out of order.');
      }
      await bridge.exportFinish(exportId);
      activeExportId = undefined;
    } catch (error) {
      if (exportId !== undefined) {
        activeExportId = undefined;
        await bridge.exportCancel(exportId).catch(() => {});
      }
      throw error;
    } finally {
      exporting = false;
    }
  }

  const viewer = mountViewer(root, undefined, {
    decodeFile: options.decodeFile ?? ((file, signal) => decodeProfileFile(file, signal, () => new InlineDecoder())),
    saveFile,
    timelineView: options.timelineView ?? createSwimlaneTimelineView(),
  });

  async function reportSmokeState(generation: number): Promise<void> {
    if (!bridge.reportSmokeEvent || disposed || generation !== loadGeneration) return;
    const state = status.dataset.state ?? '';
    const message = status.textContent ?? '';
    const spans = root.querySelectorAll('tbody tr').length;
    if (state !== 'complete' && state !== 'incomplete') {
      await bridge.reportSmokeEvent({ stage: 'loaded', state, message, spans, timelineReady: false });
      return;
    }
    const deadline = Date.now() + 15000;
    let timelineReady = false;
    while (!disposed && generation === loadGeneration && Date.now() < deadline) {
      const renderer = root.querySelector<HTMLElement>('#timeline-renderer')?.textContent?.trim();
      const canvas = root.querySelector<HTMLCanvasElement>('#timeline-host canvas');
      if (renderer === 'Nacelle Swimlanes' && canvas && !canvas.hidden && canvas.width > 0 && canvas.height > 0) {
        timelineReady = true;
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (!disposed && generation === loadGeneration) {
      await bridge.reportSmokeEvent({ stage: 'loaded', state, message, spans, timelineReady });
    }
  }

  async function loadCapture(rawMeta: DesktopCaptureMeta): Promise<void> {
    if (disposed) return;
    let meta: DesktopCaptureMeta;
    try {
      meta = validateCaptureMeta(rawMeta);
    } catch (error) {
      showHostError(`Invalid profile: ${error instanceof Error ? error.message : String(error)}`);
      return;
    }

    const generation = ++loadGeneration;
    await viewer.openSource(meta.name, async signal => {
      activeReadId = meta.id;
      const onAbort = () => {
        void bridge.cancelRead(meta.id).catch(() => {});
      };
      signal.addEventListener('abort', onAbort, { once: true });
      try {
        const bytes = new Uint8Array(meta.size);
        for (let offset = 0; offset < bytes.length;) {
          signal.throwIfAborted();
          const chunk = toChunkBytes(await bridge.readChunk(meta.id, offset));
          signal.throwIfAborted();
          requireValid(chunk.byteLength > 0 && chunk.byteLength <= EXPORT_CHUNK_BYTES &&
            offset + chunk.byteLength <= bytes.length,
          'Read chunk is empty, oversized, or exceeds declared capture size.');
          bytes.set(chunk, offset);
          offset += chunk.byteLength;
        }
        return new Blob([bytes]);
      } finally {
        signal.removeEventListener('abort', onAbort);
        if (activeReadId === meta.id) activeReadId = undefined;
        await bridge.cancelRead(meta.id).catch(() => {});
      }
    });
    if (generation === loadGeneration && !disposed) {
      void reportSmokeState(generation).catch(error => console.error('Desktop smoke status failed.', error));
    }
  }

  async function openDialog(): Promise<void> {
    if (disposed) return;
    try {
      const selected = await bridge.openDialog();
      if (selected && !disposed) await loadCapture(selected);
    } catch (error) {
      if (!disposed) {
        showHostError(`Unable to open profile: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }

  async function reload(): Promise<void> {
    if (disposed) return;
    try {
      const selected = await bridge.reloadCapture();
      if (!disposed) await loadCapture(selected);
    } catch (error) {
      if (!disposed) {
        showHostError(`Unable to reload profile: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
  }

  function handleMenuAction(action: DesktopMenuAction): void {
    if (disposed) return;
    if (action === 'open') void openDialog();
    else if (action === 'reload') void reload();
    else if (action === 'export-trace') {
      void viewer.exportTrace().catch(error => console.error('Desktop trace export failed.', error));
    } else if (action === 'export-speedscope') {
      void viewer.exportSpeedscope().catch(error => console.error('Desktop speedscope export failed.', error));
    }
  }

  openButton.addEventListener('click', () => {
    void openDialog();
  });

  const blockDrag = (event: DragEvent) => {
    event.preventDefault();
  };
  win?.addEventListener('dragover', blockDrag);
  win?.addEventListener('drop', blockDrag);

  function registerUnsub(promise: Promise<() => void>): void {
    void promise.then(unsub => {
      if (disposed) unsub();
      else unsubscribers.push(unsub);
    }).catch(error => {
      if (!disposed) console.error('Desktop event subscription failed.', error);
    });
  }

  registerUnsub(bridge.onOpenCapture(capture => {
    void loadCapture(capture);
  }));
  registerUnsub(bridge.onMenuAction(handleMenuAction));
  if (bridge.onHostError) {
    registerUnsub(bridge.onHostError(message => {
      if (!disposed) showHostError(message);
    }));
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    ++loadGeneration;
    win?.removeEventListener('dragover', blockDrag);
    win?.removeEventListener('drop', blockDrag);
    for (const unsub of unsubscribers.splice(0)) {
      try { unsub(); } catch { /* ignore teardown listener error */ }
    }
    if (activeReadId !== undefined) {
      const readId = activeReadId;
      activeReadId = undefined;
      void bridge.cancelRead(readId).catch(() => {});
    }
    if (activeExportId !== undefined) {
      const exportId = activeExportId;
      activeExportId = undefined;
      void bridge.exportCancel(exportId).catch(() => {});
    }
    viewer.dispose();
  }

  win?.addEventListener('pagehide', dispose, { once: true });

  if (bridge.reportSmokeEvent) {
    win?.addEventListener('error', event => {
      void bridge.reportSmokeEvent?.({ stage: 'error', message: event.message }).catch(() => {});
    });
    win?.addEventListener('unhandledrejection', event => {
      void bridge.reportSmokeEvent?.({ stage: 'rejection', message: String(event.reason) }).catch(() => {});
    });
  }

  void bridge.getInitialCapture().then(initial => {
    if (initial && !disposed) return loadCapture(initial);
    return undefined;
  }).catch(error => {
    if (!disposed) {
      showHostError(`Unable to open initial profile: ${error instanceof Error ? error.message : String(error)}`);
    }
  });

  return {
    openDialog,
    reload,
    exportTrace: () => viewer.exportTrace(),
    exportSpeedscope: () => viewer.exportSpeedscope(),
    dispose,
  };
}
