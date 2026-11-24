import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type {
  DesktopCaptureMeta,
  DesktopHostBridge,
  DesktopMenuAction,
  ExportName,
} from '../../nacelle-view/src/host-protocol';

declare const __NACELLE_SMOKE_HARNESS__: boolean;

const bridge: DesktopHostBridge = {
  version: 1,
  getInitialCapture: () => invoke<DesktopCaptureMeta | null>('desktop_get_initial_capture'),
  openDialog: () => invoke<DesktopCaptureMeta | null>('desktop_open_dialog'),
  reloadCapture: () => invoke<DesktopCaptureMeta>('desktop_reload_capture'),
  readChunk: (id: number, offset: number) =>
    invoke<ArrayBuffer>('desktop_read_chunk', { id, offset }),
  cancelRead: (id: number) => invoke<void>('desktop_cancel_read', { id }),
  exportStart: (name: ExportName, size: number) =>
    invoke<number>('desktop_export_start', { name, size }),
  exportChunk: (id: number, offset: number, bytes: Uint8Array) =>
    invoke<number>('desktop_export_chunk', bytes, {
      headers: {
        'x-nacelle-export-id': String(id),
        'x-nacelle-export-offset': String(offset),
      },
    }),
  exportFinish: (id: number) => invoke<void>('desktop_export_finish', { id }),
  exportCancel: (id: number) => invoke<void>('desktop_export_cancel', { id }),
  onOpenCapture: async (listener: (capture: DesktopCaptureMeta) => void) =>
    listen<DesktopCaptureMeta>('nacelle://open-capture', event => {
      listener(event.payload);
    }),
  onMenuAction: async (listener: (action: DesktopMenuAction) => void) =>
    listen<DesktopMenuAction>('nacelle://menu-action', event => {
      listener(event.payload);
    }),
  onHostError: async (listener: (message: string) => void) =>
    listen<string>('nacelle://host-error', event => {
      listener(event.payload);
    }),
};

if (__NACELLE_SMOKE_HARNESS__) {
  bridge.reportSmokeEvent = (payload: Record<string, unknown>) =>
    invoke<void>('desktop_smoke_report', { payload });
}

Object.freeze(bridge);

Object.defineProperty(window, '__NACELLE_DESKTOP_HOST__', {
  value: bridge,
  writable: false,
  configurable: false,
  enumerable: false,
});
