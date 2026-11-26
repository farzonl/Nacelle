// @vitest-environment jsdom
import html from '../index.html?raw';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mountDesktopViewer } from '../src/desktop-host';
import { FileProfileDataSource } from '../src/file-source';
import { EXPORT_CHUNK_BYTES, type DesktopCaptureMeta, type DesktopHostBridge, type DesktopMenuAction } from '../src/host-protocol';
import { loadProfile } from '../src/profile';
import type { TimelineView } from '../src/timeline';
import { fixture, span, trace } from './fixtures';

interface MockHost {
  bridge: DesktopHostBridge;
  emitOpen(capture: DesktopCaptureMeta): void;
  emitMenu(action: DesktopMenuAction): void;
  emitError(message: string): void;
  exportedFiles: Map<string, Uint8Array>;
  cancelledReads: number[];
  cancelledExports: number[];
}

function createMockHost(captures: Map<number, { name: string; bytes: Uint8Array }>, initialId?: number): MockHost {
  let openListener: ((capture: DesktopCaptureMeta) => void) | undefined;
  let menuListener: ((action: DesktopMenuAction) => void) | undefined;
  let errorListener: ((message: string) => void) | undefined;
  let nextExportId = 1;
  const activeExports = new Map<number, { name: string; size: number; bytes: Uint8Array; offset: number }>();
  const exportedFiles = new Map<string, Uint8Array>();
  const cancelledReads: number[] = [];
  const cancelledExports: number[] = [];

  const metaFor = (id: number): DesktopCaptureMeta => {
    const entry = captures.get(id);
    if (!entry) throw new Error(`Unknown capture ${id}`);
    return { id, name: entry.name, size: entry.bytes.byteLength };
  };

  const bridge: DesktopHostBridge = {
    version: 1,
    getInitialCapture: vi.fn(async () => (initialId !== undefined ? metaFor(initialId) : null)),
    openDialog: vi.fn(async () => {
      const first = captures.keys().next().value;
      return first !== undefined ? metaFor(first) : null;
    }),
    reloadCapture: vi.fn(async () => {
      const first = captures.keys().next().value;
      if (first === undefined) throw new Error('No capture to reload.');
      return metaFor(first);
    }),
    readChunk: vi.fn(async (id: number, offset: number) => {
      const entry = captures.get(id);
      if (!entry) throw new Error('Unknown read handle.');
      const slice = entry.bytes.slice(offset, offset + EXPORT_CHUNK_BYTES);
      return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
    }),
    cancelRead: vi.fn(async (id: number) => {
      cancelledReads.push(id);
    }),
    exportStart: vi.fn(async (name, size) => {
      const id = nextExportId++;
      activeExports.set(id, { name, size, bytes: new Uint8Array(size), offset: 0 });
      return id;
    }),
    exportChunk: vi.fn(async (id, offset, bytes) => {
      const tx = activeExports.get(id);
      if (!tx || tx.offset !== offset) throw new Error('Out-of-order export chunk.');
      tx.bytes.set(bytes, offset);
      tx.offset += bytes.byteLength;
      return tx.offset;
    }),
    exportFinish: vi.fn(async id => {
      const tx = activeExports.get(id);
      if (!tx || tx.offset !== tx.size) throw new Error('Incomplete export.');
      exportedFiles.set(tx.name, tx.bytes);
      activeExports.delete(id);
    }),
    exportCancel: vi.fn(async id => {
      cancelledExports.push(id);
      activeExports.delete(id);
    }),
    onOpenCapture: vi.fn(async listener => {
      openListener = listener;
      return () => { openListener = undefined; };
    }),
    onMenuAction: vi.fn(async listener => {
      menuListener = listener;
      return () => { menuListener = undefined; };
    }),
    onHostError: vi.fn(async listener => {
      errorListener = listener;
      return () => { errorListener = undefined; };
    }),
  };

  return {
    bridge,
    emitOpen: capture => openListener?.(capture),
    emitMenu: action => menuListener?.(action),
    emitError: message => errorListener?.(message),
    exportedFiles,
    cancelledReads,
    cancelledExports,
  };
}

function noopTimeline(): TimelineView {
  return {
    capabilities: { rendererName: 'Mock', multiThreadSwimlanes: false, supportsSelectionFilter: false },
    mount: vi.fn(),
    render: vi.fn(),
    clear: vi.fn(),
    dispose: vi.fn(),
  };
}

async function decodeDirect(blob: Blob, signal: AbortSignal) {
  signal.throwIfAborted();
  const buffer = typeof blob.arrayBuffer === 'function'
    ? await blob.arrayBuffer()
    : await new Promise<ArrayBuffer>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as ArrayBuffer);
        reader.onerror = () => reject(reader.error ?? new Error('Failed to read blob.'));
        reader.readAsArrayBuffer(blob);
      });
  return loadProfile(FileProfileDataSource.fromBuffer(buffer));
}

describe('desktop host adapter', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '/');
    document.documentElement.innerHTML = html;
  });

  it('loads initial capture via chunked binary IPC, updates chrome, and exports Chrome trace and Speedscope', async () => {
    const tracePayload = trace([span({ name: 'desktop_dispatch', ts: 100, dur: 45 })]);
    const archive = new Uint8Array(fixture({}, tracePayload));
    const mock = createMockHost(new Map([[1, { name: 'capture.nacelle-profile', bytes: archive }]]), 1);

    const handle = mountDesktopViewer(mock.bridge, document, {
      decodeFile: decodeDirect,
      timelineView: noopTimeline(),
    });

    expect(document.querySelector('#mode-badge')!.textContent).toBe('Desktop · Offline');
    expect(document.querySelectorAll('nav a')).toHaveLength(0);
    expect(document.querySelector<HTMLInputElement>('#profile-file')!.disabled).toBe(true);
    expect(document.querySelector<HTMLButtonElement>('#desktop-open')).not.toBeNull();

    await vi.waitFor(() => {
      const el = document.querySelector('#status')!;
      expect(`${el.getAttribute('data-state')}: ${el.textContent}`).toMatch(/^complete:/);
    });
    expect(document.querySelector('#threads')!.textContent).toContain('desktop_dispatch');
    expect(mock.cancelledReads).toContain(1);

    await handle.exportTrace();
    const exportedTrace = mock.exportedFiles.get('nacelle-trace.json');
    expect(exportedTrace).toBeDefined();
    expect(JSON.parse(new TextDecoder().decode(exportedTrace))).toEqual(tracePayload);

    await handle.exportSpeedscope();
    const exportedSpeedscope = mock.exportedFiles.get('nacelle-speedscope.json');
    expect(exportedSpeedscope).toBeDefined();
    expect(JSON.parse(new TextDecoder().decode(exportedSpeedscope)).exporter).toContain('nacelle');

    handle.dispose();
  });

  it('opens captures via native dialog button and OS drag-drop events, and rejects oversized chunks', async () => {
    const archive = new Uint8Array(fixture({}, trace([span({ name: 'dropped_dispatch' })])));
    const mock = createMockHost(new Map([[7, { name: 'dropped.nacelle-profile', bytes: archive }]]));

    const handle = mountDesktopViewer(mock.bridge, document, {
      decodeFile: decodeDirect,
      timelineView: noopTimeline(),
    });

    document.querySelector<HTMLButtonElement>('#desktop-open')!.click();
    await vi.waitFor(() => {
      const el = document.querySelector('#status')!;
      expect(`${el.getAttribute('data-state')}: ${el.textContent}`).toMatch(/^complete:/);
    });
    expect(document.querySelector('#threads')!.textContent).toContain('dropped_dispatch');

    // Simulate oversized chunk from a misbehaving host.
    vi.mocked(mock.bridge.readChunk).mockResolvedValueOnce(new ArrayBuffer(EXPORT_CHUNK_BYTES + 1));
    mock.emitOpen({ id: 8, name: 'bad.nacelle-profile', size: 100 });
    await vi.waitFor(() => expect(document.querySelector('#status')!.getAttribute('data-state')).toBe('invalid'));
    expect(document.querySelector('#status')!.textContent).toContain('Read chunk is empty, oversized, or exceeds declared capture size.');

    mock.emitError('Only .nacelle-profile files are accepted.');
    expect(document.querySelector('#status')!.textContent).toBe('Only .nacelle-profile files are accepted.');

    handle.dispose();
  });

  it('cancels export transactions cleanly when chunk delivery fails', async () => {
    const archive = new Uint8Array(fixture({}, trace([span({ name: 'export_fail_dispatch' })])));
    const mock = createMockHost(new Map([[1, { name: 'capture.nacelle-profile', bytes: archive }]]), 1);

    const handle = mountDesktopViewer(mock.bridge, document, {
      decodeFile: decodeDirect,
      timelineView: noopTimeline(),
    });
    await vi.waitFor(() => {
      const el = document.querySelector('#status')!;
      expect(`${el.getAttribute('data-state')}: ${el.textContent}`).toMatch(/^complete:/);
    });

    vi.mocked(mock.bridge.exportChunk).mockRejectedValueOnce(new Error('Disk full'));
    await expect(handle.exportTrace()).rejects.toThrow('Disk full');
    expect(mock.cancelledExports).toEqual([1]);
    expect(document.querySelector('#export-status')!.textContent).toContain('Disk full');

    handle.dispose();
  });
});
