import * as vscode from 'vscode';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { isRecord } from '../../nacelle-view/src/host-protocol.ts';
import { checkDebugResult, commandList, debugEnvironment, debugInteger, nativeDebugSupported } from './debug-policy.ts';

interface Stop { session: vscode.DebugSession; generation: number; thread: number; frame?: number; name?: string }

export class ShaderDebugger {
  private readonly context: vscode.ExtensionContext;
  private readonly output: vscode.OutputChannel;
  private generation = 0;
  private stop: Stop | undefined;

  constructor(context: vscode.ExtensionContext, output: vscode.OutputChannel) {
    this.context = context;
    this.output = output;
  }

  register(): void {
    this.context.subscriptions.push(
      vscode.debug.registerDebugConfigurationProvider('nacelle-lldb', {
        resolveDebugConfigurationWithSubstitutedVariables: async (_folder, config) => {
          try {
            this.authorize();
            if (config.request !== 'launch' || typeof config.program !== 'string' || !config.program) {
              throw new Error('Only explicit configured-application launches are supported.');
            }
            const mode: unknown = config.mode ?? 'native';
            if (mode !== 'native' && mode !== 'hlsl' && mode !== 'optimized-hlsl') throw new Error('Unsupported shader debug mode.');
            if (config.disableASLR !== undefined && typeof config.disableASLR !== 'boolean') {
              throw new Error('disableASLR must be an explicit boolean.');
            }
            const program = await vscode.workspace.fs.stat(vscode.Uri.file(config.program));
            if (program.type !== vscode.FileType.File) throw new Error('Debug program is not a regular file.');
            await this.adapter();
            const settings = [
              'settings set symbols.enable-external-lookup false',
              'settings set symbols.enable-lldb-index-cache false',
              'settings set target.disable-aslr false',
              'settings set plugin.jit-loader.gdb.enable on',
            ];
            const commands = mode === 'native' ? ['breakpoint set --func-regex ^nacelle_(cs|vs|ps)_'] :
              [`command source "${vscode.Uri.joinPath(this.context.extensionUri, 'dist', 'debug',
                mode === 'hlsl' ? 'source-debug.lldb' : 'optimized-source-debug.lldb').fsPath.replace(/["\\]/g, '\\$&')}"`];
            return { ...config, mode, disableASLR: config.disableASLR ?? false, env: debugEnvironment(config.env),
              initCommands: [...settings, ...commandList(config.initCommands, 'initCommands')],
              preRunCommands: [...commands, ...commandList(config.preRunCommands, 'preRunCommands')] };
          } catch (error) {
            this.report(error);
            return undefined;
          }
        },
      }),
      vscode.debug.registerDebugAdapterDescriptorFactory('nacelle-lldb', {
        createDebugAdapterDescriptor: async () => {
          this.authorize();
          return new vscode.DebugAdapterExecutable(await this.adapter(), ['--repl-mode', 'command']);
        },
      }),
      vscode.debug.registerDebugAdapterTrackerFactory('nacelle-lldb', {
        createDebugAdapterTracker: session => {
          let ended = false;
          return {
          onWillReceiveMessage: (message: unknown) => {
            if (isRecord(message) && ['continue', 'next', 'stepIn', 'stepOut', 'disconnect', 'terminate'].includes(String(message.command))) {
              this.clear(session.id);
            }
          },
          onDidSendMessage: (message: unknown) => {
            if (!isRecord(message) || message.type !== 'event') return;
            if (message.event === 'continued' || message.event === 'terminated' || message.event === 'exited') {
              if (message.event !== 'continued') ended = true;
              this.clear(session.id);
            } else if (message.event === 'stopped' && isRecord(message.body) &&
                       Number.isSafeInteger(message.body.threadId)) {
              const stop: Stop = { session, generation: ++this.generation, thread: Number(message.body.threadId) };
              this.stop = stop;
              void this.resolveFrame(stop).catch(error => {
                if (this.stop === stop) this.report(error);
                else this.output.appendLine(`Stopped-frame resolution cancelled after resume: ${String(error)}`);
              });
            }
          },
          onError: error => {
            this.clear(session.id);
            if (ended) this.output.appendLine(`Adapter connection closed after workload exit: ${String(error)}`);
            else this.report(error);
          },
          onExit: () => this.clear(session.id),
        };
        },
      }),
      vscode.debug.onDidTerminateDebugSession(session => this.clear(session.id)),
    );
  }

  capabilities(): string {
    return `Extension host: ${process.platform}/${process.arch}; remote: ${vscode.env.remoteName ?? 'local'}.\n` +
      `Workspace trust: ${vscode.workspace.isTrusted ? 'trusted' : 'untrusted; execution disabled'}.\n` +
      `Native/HLSL launch: ${nativeDebugSupported(process.platform, process.arch) ? 'supported with a configured runtime and lldb-dap' : 'unsupported'}.\n` +
      'HLSL stepping observes the application-selected invocation; lane inspection does not retarget it.\n' +
      'HLSL initial breakpoint selection is application-controlled. Editor-line breakpoint insertion, HLSL step-in/out, attach, and #226 remote RPC are not claimed.\n' +
      'Profile editing is independent of debugger availability. Language intelligence belongs to upstream clangd.';
  }

  async inspect(command: 'hlsl-location' | 'hlsl-values' | 'hlsl-code'): Promise<void> {
    const stop = await this.sourceStop();
    const result = await this.evaluate(stop, command, 'repl');
    if (!isRecord(result) || typeof result.result !== 'string') throw new Error('Debugger inspection is unavailable.');
    this.output.appendLine(result.result);
    this.output.show(true);
  }

  async navigateSource(): Promise<void> {
    const stop = await this.sourceStop();
    await this.inspect('hlsl-location');
    const base = this.base(stop);
    const line = debugInteger(await this.evaluate(stop, `${base}->site.line`, 'watch'));
    const identity = await this.evaluate(stop, `(const char*)${base}->site.source_id`, 'watch');
    if (!isRecord(identity) || typeof identity.result !== 'string') throw new Error('Source identity is unavailable.');
    const hash = /"([a-f0-9]{64})"/.exec(identity.result)?.[1];
    if (!hash || !line) throw new Error('Source identity or mapped line is unavailable.');
    const selected = await vscode.window.showOpenDialog({ canSelectMany: false,
      title: 'Select the exact HLSL source (content identity must match the stopped shader)',
      filters: { HLSL: ['hlsl', 'hlsli'] } });
    if (!selected?.[0]) {
      this.output.appendLine('HLSL source navigation cancelled by user.');
      return;
    }
    const uri = selected[0];
    if (uri.scheme !== 'file' && uri.scheme !== 'vscode-remote') throw new Error('Unsupported source URI scheme.');
    const stat = await vscode.workspace.fs.stat(uri);
    if (stat.type !== vscode.FileType.File || stat.size > 16 * 1024 * 1024) {
      throw new Error('Source is not a regular file or exceeds the supported 16 MiB limit.');
    }
    const bytes = await vscode.workspace.fs.readFile(uri);
    if (bytes.length > 16 * 1024 * 1024) throw new Error('Source grew beyond the supported 16 MiB limit.');
    if (createHash('sha256').update(bytes).digest('hex') !== hash) throw new Error('Selected source does not match the exact stopped source identity.');
    this.current(stop);
    const document = await vscode.workspace.openTextDocument(uri);
    if (line > document.lineCount) throw new Error('Mapped source line is outside the selected source.');
    this.current(stop);
    const editor = await vscode.window.showTextDocument(document);
    const range = new vscode.Range(line - 1, 0, line - 1, 0);
    editor.selection = new vscode.Selection(range.start, range.end);
    editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
  }

  async inspectLane(): Promise<void> {
    const stop = await this.sourceStop();
    const base = this.base(stop);
    const width = debugInteger(await this.evaluate(stop, `${base}->width`, 'watch'), 16);
    const active = debugInteger(await this.evaluate(stop, `${base}->active_mask`, 'watch'), 65535);
    const real = debugInteger(await this.evaluate(stop, `${base}->real_mask`, 'watch'), 65535);
    if (![4, 8, 16].includes(width)) throw new Error('Unsupported snapshot lane width.');
    if ((active >>> width) || (real >>> width)) throw new Error('Snapshot masks exceed the supported lane width.');
    const picked = await vscode.window.showQuickPick(Array.from({ length: width }, (_, lane) => ({
      label: `Lane ${lane}`, lane, description: !(active & (1 << lane)) ? 'Inactive: values unavailable' :
        real & (1 << lane) ? 'Active submitted lane' : 'Active helper/companion lane',
    })), { title: 'Inspect a captured SIMD slot (does not change the selected invocation)' });
    if (!picked) return;
    this.current(stop);
    this.output.appendLine(`Lane ${picked.lane}: ${picked.description}`);
    if (!(active & (1 << picked.lane))) {
      this.output.show(true);
      return;
    }
    const coordinates = await this.evaluate(stop, `${base}->xyz[${picked.lane}]`, 'watch');
    this.output.appendLine(JSON.stringify(coordinates));
    const count = debugInteger(await this.evaluate(stop, `${base}->site.local_count`, 'watch'), 64);
    for (let local = 0; local < count; ++local) {
      const metadata = await this.evaluate(stop, `${base}->locals[${local}]`, 'watch');
      const kind = debugInteger(await this.evaluate(stop, `${base}->locals[${local}].kind`, 'watch'), 4);
      let valid = Boolean(kind);
      let availability = '';
      if (stop.session.configuration.mode === 'optimized-hlsl') {
        const mask = debugInteger(await this.evaluate(stop, `snapshot->valid_mask[${local}]`, 'watch'), 65535);
        const reason = debugInteger(await this.evaluate(stop, `snapshot->availability[${local}]`, 'watch'), 6);
        valid &&= Boolean(mask & (1 << picked.lane));
        availability = ` availability=${reason}`;
      }
      if (!valid) {
        this.output.appendLine(`Local ${local}: unavailable${availability}; metadata=${JSON.stringify(metadata)}`);
        continue;
      }
      const value = await this.evaluate(stop, `${base}->values[${local}][${picked.lane}]`, 'watch');
      this.output.appendLine(`Local ${local}: metadata=${JSON.stringify(metadata)} raw bits=${JSON.stringify(value)}`);
    }
    if (stop.session.configuration.mode === 'optimized-hlsl') await this.inspect('hlsl-values');
    await this.inspect('hlsl-code');
    this.output.show(true);
  }

  async control(action: 'step' | 'continue' | 'cancel'): Promise<void> {
    const stop = await this.sourceStop();
    await this.evaluate(stop, `hlsl-arm-${action}`, 'repl');
    this.current(stop);
    await stop.session.customRequest('continue', { threadId: stop.thread, singleThread: false });
  }

  private authorize(): void {
    if (!vscode.workspace.isTrusted) throw new Error('Shader debugger execution requires workspace trust.');
    if (!nativeDebugSupported(process.platform, process.arch)) throw new Error('Nacelle runtime shader debugging currently requires Linux x86_64; offline profile viewing remains available.');
  }

  private async adapter(): Promise<string> {
    const path = vscode.workspace.getConfiguration('nacelle.debug').get<string>('lldbDapPath', 'lldb-dap');
    const { stdout } = await promisify(execFile)(path, ['--version'], { timeout: 5000, maxBuffer: 65536 });
    if (!stdout.includes('lldb-dap')) throw new Error('Configured executable is not standard lldb-dap.');
    return path;
  }

  private clear(sessionId: string): void {
    if (this.stop?.session.id === sessionId) {
      ++this.generation;
      this.stop = undefined;
    }
  }

  private async resolveFrame(stop: Stop): Promise<void> {
    const response: unknown = await stop.session.customRequest('stackTrace', { threadId: stop.thread, startFrame: 0, levels: 1 });
    this.current(stop);
    if (!isRecord(response) || !Array.isArray(response.stackFrames) || !isRecord(response.stackFrames[0])) {
      throw new Error('Stopped debugger frame is unavailable.');
    }
    const frame = response.stackFrames[0];
    if (!Number.isSafeInteger(frame.id) || typeof frame.name !== 'string') throw new Error('Invalid stopped frame.');
    stop.frame = Number(frame.id);
    stop.name = frame.name;
  }

  private current(stop: Stop): void {
    this.authorize();
    if (this.stop !== stop || stop.generation !== this.generation) throw new Error('Debugger resumed, disconnected, or changed code version; stale inspection cancelled.');
  }

  private async sourceStop(): Promise<Stop> {
    this.authorize();
    const stop = this.stop;
    if (!stop) throw new Error('No supported stopped shader session is active.');
    if (!['hlsl', 'optimized-hlsl'].includes(stop.session.configuration.mode)) throw new Error('Select an explicit HLSL launch mode for HLSL controls.');
    if (stop.frame === undefined) await this.resolveFrame(stop);
    this.current(stop);
    const expected = stop.session.configuration.mode === 'optimized-hlsl' ?
      'nacelle_optimized_source_debug_stop' : 'nacelle_source_debug_stop';
    if (!stop.name?.includes(expected)) throw new Error('HLSL controls require a genuine source-snapshot stop, not a host/native frame.');
    return stop;
  }

  private base(stop: Stop): string {
    return stop.session.configuration.mode === 'optimized-hlsl' ? '(&snapshot->snapshot)' : 'snapshot';
  }

  private async evaluate(stop: Stop, expression: string, context: 'watch' | 'repl'): Promise<unknown> {
    this.current(stop);
    if (context === 'repl') {
      for (const selection of [`thread select --thread-id ${stop.thread}`, 'frame select 0']) {
        const selected: unknown = await stop.session.customRequest('evaluate', { expression: selection, context: 'repl' });
        this.current(stop);
        checkDebugResult(selected);
      }
    }
    const result: unknown = await stop.session.customRequest('evaluate', { expression, context, frameId: stop.frame });
    this.current(stop);
    checkDebugResult(result);
    return result;
  }

  private report(error: unknown): void {
    const reason = error instanceof Error ? error.message : String(error);
    this.output.appendLine(`Shader debugger: ${reason}`);
    void vscode.window.showErrorMessage(`Nacelle shader debugger: ${reason}`);
  }
}
