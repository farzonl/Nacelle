import * as vscode from 'vscode';
import { randomBytes } from 'node:crypto';
import { posix } from 'node:path';
import { EXPORT_CHUNK_BYTES, HOST_VERSION, MAX_EXPORT_BYTES, exportMessage, readMessage, isRecord, type ExportReply } from '../../nacelle-view/src/host-protocol.ts';
import { configurationJson, rewriteAssets } from './html.ts';
import { ExportBuffer } from './transfer.ts';

export interface EditorStatus { state: string; message: string }
interface Transfer {
  id: number;
  target?: vscode.Uri;
  buffer?: ExportBuffer;
  finalizing?: boolean;
}
interface ReadTransfer { id: number; bytes?: Uint8Array; offset: number }

export class ProfileEditor implements vscode.CustomReadonlyEditorProvider {
  private readonly context: vscode.ExtensionContext;
  private readonly output: vscode.OutputChannel;
  private readonly panels = new Map<string, vscode.WebviewPanel>();
  private readonly statuses = new Map<string, EditorStatus>();

  constructor(context: vscode.ExtensionContext, output: vscode.OutputChannel) {
    this.context = context;
    this.output = output;
  }

  getStatus(uri: vscode.Uri): EditorStatus | undefined { return this.statuses.get(uri.toString()); }

  async exportTrace(): Promise<void> {
    if (!vscode.workspace.isTrusted) throw new Error('Export requires workspace trust.');
    const selected = [...this.panels.entries()].find(([, candidate]) => candidate.active);
    if (!selected) throw new Error('Open and focus a Nacelle profile editor first.');
    const [key, panel] = selected;
    if (!['complete', 'incomplete'].includes(this.statuses.get(key)?.state ?? '')) {
      throw new Error('No validated capture is ready to export.');
    }
    if (!await panel.webview.postMessage({ version: HOST_VERSION, kind: 'export-trace' })) {
      throw new Error('The profile editor is no longer available.');
    }
  }

  async openCustomDocument(uri: vscode.Uri): Promise<vscode.CustomDocument> {
    if (uri.scheme !== 'file' && uri.scheme !== 'vscode-remote') throw new Error('Unsupported profile URI scheme.');
    const stat = await vscode.workspace.fs.stat(uri);
    if (stat.type !== vscode.FileType.File || stat.size > MAX_EXPORT_BYTES) {
      throw new Error('Profile is not a regular file or exceeds the 256 MiB limit.');
    }
    return { uri, dispose() {} };
  }

  async resolveCustomEditor(document: vscode.CustomDocument, panel: vscode.WebviewPanel,
                            token: vscode.CancellationToken): Promise<void> {
    const key = document.uri.toString();
    const assets = vscode.Uri.joinPath(this.context.extensionUri, 'dist', 'viewer');
    const nonce = randomBytes(16).toString('hex');
    panel.webview.options = { enableScripts: true, localResourceRoots: [assets] };
    const assetUri = (path: string) => panel.webview.asWebviewUri(vscode.Uri.joinPath(assets, path)).toString();
    const decoder = new TextDecoder();
    const html = decoder.decode(await vscode.workspace.fs.readFile(vscode.Uri.joinPath(assets, 'index.html')));
    const frame = decoder.decode(await vscode.workspace.fs.readFile(vscode.Uri.joinPath(assets, 'speedscope', 'editor.html')));
    const csp = `default-src 'none'; script-src 'nonce-${nonce}' ${panel.webview.cspSource} 'wasm-unsafe-eval'; worker-src blob:; style-src ${panel.webview.cspSource} 'nonce-${nonce}' 'nonce-nacelle-editor'; style-src-attr 'unsafe-inline'; connect-src ${panel.webview.cspSource} blob:; img-src ${panel.webview.cspSource} data: blob:; font-src ${panel.webview.cspSource} data:; frame-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'none'`;
    const frameHtml = frame.replaceAll('__NACELLE_WEBVIEW_NONCE__', nonce)
      .replace('<head>', `<head><meta http-equiv="Content-Security-Policy" content="${csp}">`);
    const config = configurationJson({ name: posix.basename(document.uri.path), nonce, frameHtml });
    if (token.isCancellationRequested) return;
    let disposed = false;
    let transfer: Transfer | undefined;
    let reading: ReadTransfer | undefined;
    let readJob: Promise<Uint8Array> | undefined;
    let lastReadId = 0;
    let statusReadId: number | undefined;
    let lastId = 0;
    let idleTimer: ReturnType<typeof setTimeout> | undefined;
    const touch = () => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        const readId = reading?.id;
        const exportId = transfer?.id;
        reading = undefined;
        transfer = undefined;
        if (readId === undefined && exportId === undefined) return;
        this.output.appendLine('Idle profile transfers expired after 120 seconds.');
        for (const [kind, id] of [['read-reply', readId], ['export-reply', exportId]] as const) {
          if (id !== undefined && !disposed) {
            void panel.webview.postMessage({ version: 1, kind, id, error: 'Profile transfer expired.' });
          }
        }
      }, 120000);
    };
    this.panels.set(key, panel);
    this.statuses.set(key, { state: 'loading', message: 'Reading selected capture.' });

    async function reply(value: ExportReply): Promise<void> {
      if (!disposed && !await panel.webview.postMessage(value)) throw new Error('Export reply could not reach the viewer.');
    }

    const messages = panel.webview.onDidReceiveMessage(async (raw: unknown) => {
      try {
        if (isRecord(raw) && typeof raw.kind === 'string' && raw.kind.startsWith('read-')) {
          const request = readMessage(raw);
          touch();
          if (request.kind === 'read-cancel') {
            if (reading?.id === request.id) reading = undefined;
            return;
          }
          if (request.kind === 'read-start') {
            if (reading || request.id <= lastReadId) throw new Error('Read request is stale or already active.');
            lastReadId = request.id;
            statusReadId = request.id;
            const current: ReadTransfer = { id: request.id, offset: 0 };
            reading = current;
            if (readJob) {
              try { await readJob; }
              catch (error) { this.output.appendLine(`Earlier cancelled read failed: ${String(error)}`); }
            }
            if (disposed || reading !== current) throw new Error('Profile read cancelled.');
            const stat = await vscode.workspace.fs.stat(document.uri);
            if (disposed || reading !== current) throw new Error('Profile read cancelled.');
            if (stat.type !== vscode.FileType.File || stat.size > MAX_EXPORT_BYTES) throw new Error('Selected profile is not a regular bounded file.');
            const job = Promise.resolve(vscode.workspace.fs.readFile(document.uri));
            readJob = job;
            let bytes: Uint8Array;
            try { bytes = await job; }
            finally { if (readJob === job) readJob = undefined; }
            if (disposed || reading !== current) throw new Error('Profile read cancelled.');
            if (bytes.length > MAX_EXPORT_BYTES) throw new Error('Selected profile grew beyond 256 MiB.');
            current.bytes = bytes;
            if (!await panel.webview.postMessage({ version: 1, kind: 'read-reply', id: request.id, size: bytes.length })) {
              throw new Error('Read reply could not reach the viewer.');
            }
          } else {
            const current = reading;
            if (!current || current.id !== request.id || !current.bytes ||
                current.offset !== request.offset || request.offset >= current.bytes.length) {
              throw new Error('Read chunk is stale or out of order.');
            }
            const bytes = current.bytes.subarray(request.offset, request.offset + EXPORT_CHUNK_BYTES);
            current.offset += bytes.length;
            if (!await panel.webview.postMessage({ version: 1, kind: 'read-reply', id: request.id,
              offset: request.offset, data: Buffer.from(bytes).toString('base64') })) {
              throw new Error('Read chunk could not reach the viewer.');
            }
            if (current.offset === current.bytes.length) reading = undefined;
          }
          return;
        }
        if (isRecord(raw) && raw.version === HOST_VERSION && raw.kind === 'status' && Number.isSafeInteger(raw.id) &&
            typeof raw.state === 'string' && ['complete', 'incomplete', 'invalid', 'cancelled'].includes(raw.state) &&
            typeof raw.message === 'string' && raw.message.length <= 2048) {
          if (raw.id === statusReadId) this.statuses.set(key, { state: raw.state, message: raw.message });
          else this.output.appendLine('Stale capture status ignored after replacement.');
          return;
        }
        const message = exportMessage(raw);
        touch();
        if (message.kind === 'export-cancel') {
          if (transfer?.id === message.id) transfer = undefined;
          this.output.appendLine(`Export ${message.id} cancelled.`);
          return;
        }
        if (!vscode.workspace.isTrusted) throw new Error('Export requires workspace trust.');
        if (disposed) throw new Error('The viewer was closed.');
        if (message.kind === 'export-start') {
          if (!['complete', 'incomplete'].includes(this.statuses.get(key)?.state ?? '')) throw new Error('No validated capture is ready to export.');
          if (transfer || message.id <= lastId) throw new Error('An export is already active or the request is stale.');
          lastId = message.id;
          const current: Transfer = { id: message.id };
          transfer = current;
          const target = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.joinPath(document.uri, '..', message.name),
            filters: { JSON: ['json'] }, saveLabel: 'Export Nacelle profile data',
          });
          if (disposed || transfer !== current || !target) throw new Error('Export cancelled.');
          if (target.scheme !== 'file' && target.scheme !== 'vscode-remote') throw new Error('Unsupported export URI scheme.');
          current.target = target;
          current.buffer = new ExportBuffer(message.size);
          await reply({ version: 1, kind: 'export-reply', id: message.id, offset: 0 });
        } else {
          const current = transfer;
          if (!current || current.id !== message.id || !current.buffer || !current.target || current.finalizing) {
            throw new Error('Export is stale, not authorized, or already finishing.');
          }
          if (message.kind === 'export-chunk') {
            const offset = current.buffer.append(message.offset, message.data);
            await reply({ version: 1, kind: 'export-reply', id: message.id, offset });
          } else {
            current.finalizing = true;
            const bytes = current.buffer.finish();
            const temporary = vscode.Uri.joinPath(current.target, '..', `.nacelle-export-${randomBytes(16).toString('hex')}.tmp`);
            try {
              await vscode.workspace.fs.writeFile(temporary, bytes);
              if (disposed || transfer !== current || !vscode.workspace.isTrusted) throw new Error('Export cancelled before commit.');
              await vscode.workspace.fs.rename(temporary, current.target, { overwrite: true });
            } finally {
              try { await vscode.workspace.fs.delete(temporary); }
              catch (error) {
                if (!(error instanceof vscode.FileSystemError && error.code === 'FileNotFound')) {
                  this.output.appendLine(`Export staging cleanup failed: ${String(error)}`);
                  void vscode.window.showErrorMessage(`Nacelle export staging cleanup failed: ${String(error)}`);
                }
              }
            }
            transfer = undefined;
            await reply({ version: 1, kind: 'export-reply', id: message.id, offset: bytes.length });
          }
        }
      } catch (error) {
        if (isRecord(raw) && typeof raw.id === 'number' && transfer?.id === raw.id) transfer = undefined;
        const reason = error instanceof Error ? error.message : String(error);
        this.output.appendLine(`Profile IPC failed: ${reason}`);
        if (isRecord(raw) && typeof raw.kind === 'string' && raw.kind.startsWith('read-') &&
            Number.isSafeInteger(raw.id)) {
          if (reading?.id === raw.id) reading = undefined;
          if (!disposed && !await panel.webview.postMessage({ version: 1, kind: 'read-reply', id: raw.id, error: reason })) {
            this.output.appendLine('Read error could not reach the closed viewer.');
          }
          return;
        }
        if (isRecord(raw) && Number.isSafeInteger(raw.id)) {
          try { await reply({ version: 1, kind: 'export-reply', id: Number(raw.id), error: reason }); }
          catch (replyError) {
            this.output.appendLine(`Export error delivery failed: ${String(replyError)}`);
            if (!disposed) void vscode.window.showErrorMessage(`Nacelle export failed: ${reason}`);
          }
        } else if (!disposed) void vscode.window.showErrorMessage(`Nacelle profile message rejected: ${reason}`);
      }
    });
    const watcher = vscode.workspace.createFileSystemWatcher(
      new vscode.RelativePattern(vscode.Uri.joinPath(document.uri, '..'), posix.basename(document.uri.path)));
    const reload = async () => {
      if (disposed) return;
      transfer = undefined;
      statusReadId = undefined;
      try {
        const stat = await vscode.workspace.fs.stat(document.uri);
        if (stat.type !== vscode.FileType.File || stat.size > MAX_EXPORT_BYTES) throw new Error('Changed profile is not a regular file or exceeds 256 MiB.');
        this.statuses.set(key, { state: 'loading', message: 'Reloading changed capture.' });
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        panel.webview.options = { enableScripts: true, localResourceRoots: [assets] };
        this.statuses.set(key, { state: 'invalid', message: reason });
        this.output.appendLine(`Profile reload failed: ${reason}`);
        void vscode.window.showErrorMessage(`Nacelle profile reload failed: ${reason}`);
      }
      await panel.webview.postMessage({ version: 1, kind: 'reload' });
    };
    const changes = [watcher.onDidChange(reload), watcher.onDidCreate(reload), watcher.onDidDelete(reload)];
    panel.onDidDispose(() => {
      disposed = true;
      if (idleTimer) clearTimeout(idleTimer);
      transfer = undefined;
      reading = undefined;
      messages.dispose();
      changes.forEach(change => change.dispose());
      watcher.dispose();
      if (this.panels.get(key) === panel) {
        this.panels.delete(key);
        this.statuses.delete(key);
      }
    });
    panel.webview.html = rewriteAssets(html, assetUri, nonce)
      .replace(/<meta http-equiv="Content-Security-Policy"[^>]*>/, `<meta http-equiv="Content-Security-Policy" content="${csp}">`)
      .replace('</head>', () => `<script id="nacelle-host-config" type="application/json" nonce="${nonce}">${config}</script></head>`);
  }
}
