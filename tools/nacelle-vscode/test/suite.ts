import * as vscode from 'vscode';
import assert from 'node:assert/strict';
import { appendFileSync, existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

async function until(predicate: () => boolean | Promise<boolean>, label: string, timeout = 120000): Promise<void> {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out: ${label}`);
}

export async function run(): Promise<void> {
  const work = process.env.NACELLE_EXTENSION_SMOKE_DIR;
  const program = process.env.NACELLE_EXTENSION_DEBUG_PROGRAM;
  if (!work || !program) throw new Error('Smoke directory and real native fixture are required.');
  const extension = vscode.extensions.getExtension<{
    getEditorStatus(uri: vscode.Uri): { state: string; message: string } | undefined;
  }>('nacelle.nacelle-vscode');
  assert.ok(extension);
  const api = await extension.activate();
  await vscode.workspace.getConfiguration('files').update('simpleDialog.enable', true, vscode.ConfigurationTarget.Global);
  const uri = vscode.Uri.file(join(work, 'capture.nacelle-profile'));
  await vscode.commands.executeCommand('vscode.openWith', uri, 'nacelle.profile');
  await until(() => {
    const status = api.getEditorStatus(uri);
    if (status?.state === 'invalid') throw new Error(`Shared viewer failed: ${status.message}`);
    return status?.state === 'complete';
  }, 'actual shared viewer render');
  await writeFile(join(work, 'stage.json'), JSON.stringify({ stage: 'render' }));
  await until(() => existsSync(join(work, 'render-verified')), 'real swimlane timeline rendering');
  await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(join(work, 'expected-trace.json')));
  await vscode.commands.executeCommand('vscode.openWith', uri, 'nacelle.profile');
  if (process.env.NACELLE_EXTENSION_SMOKE_UNTRUSTED === '1') {
    assert.equal(vscode.workspace.isTrusted, false);
    await assert.rejects(async () => { await vscode.commands.executeCommand('nacelle.exportTrace'); }, /workspace trust/);
    for (const command of ['nacelle.hlslStep', 'nacelle.hlslLane', 'nacelle.hlslValues']) {
      await assert.rejects(async () => { await vscode.commands.executeCommand(command); }, /workspace trust/);
    }
    assert.equal(vscode.debug.activeDebugSession, undefined);
    assert.equal(existsSync(join(work, 'nacelle-trace.json')), false);
    await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
    await until(() => !api.getEditorStatus(uri), 'untrusted editor teardown');
    await writeFile(join(work, 'stage.json'), JSON.stringify({ stage: 'complete' }));
    return;
  }
  await writeFile(join(work, 'stage.json'), JSON.stringify({ stage: 'export' }));
  await vscode.commands.executeCommand('nacelle.exportTrace');
  const exported = join(work, 'nacelle-trace.json');
  await until(() => existsSync(exported), 'actual webview-to-host export');
  assert.deepEqual(await readFile(exported), await readFile(join(work, 'expected-trace.json')));
  const original = await readFile(uri.fsPath);
  await writeFile(uri.fsPath, 'invalid capture');
  await until(() => api.getEditorStatus(uri)?.state === 'invalid', 'changed invalid capture clears old viewer state');
  await assert.rejects(async () => { await vscode.commands.executeCommand('nacelle.exportTrace'); }, /No validated capture/);
  await writeFile(uri.fsPath, original);
  await until(() => api.getEditorStatus(uri)?.state === 'complete', 'restored capture reload');
  await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
  await until(() => !api.getEditorStatus(uri), 'editor teardown');

  let session: vscode.DebugSession | undefined;
  let stopped: { thread: number; sequence: number } | undefined;
  let sequence = 0;
  let output = '';
  let exited = false;
  function currentStop(): { thread: number; sequence: number } {
    assert.ok(stopped, 'Expected a real stopped shader event.');
    return stopped;
  }
  const tracker = vscode.debug.registerDebugAdapterTrackerFactory('nacelle-lldb', {
    createDebugAdapterTracker: active => {
      session = active;
      return {
        onWillReceiveMessage(message) {
          appendFileSync(join(work, 'dap.jsonl'), `${JSON.stringify({ direction: 'request', message }).slice(0, 8192)}\n`);
        },
        onDidSendMessage(message) {
          appendFileSync(join(work, 'dap.jsonl'), `${JSON.stringify({ direction: 'response', message }).slice(0, 8192)}\n`);
          if (message.type !== 'event') return;
          if (message.event === 'stopped') stopped = { thread: message.body.threadId, sequence: ++sequence };
          if (message.event === 'output') output += message.body.output;
          if (message.event === 'terminated') exited = true;
        },
      };
    },
  });
  try {
    assert.ok(await vscode.debug.startDebugging(undefined, {
      type: 'nacelle-lldb', request: 'launch', name: 'Native shader smoke', mode: 'native',
      program, args: ['unused', 'native-compute'], cwd: work,
    }));
    await until(() => Boolean(stopped && session), 'genuine shader breakpoint');
    assert.ok(stopped && session);
    const stack = await session.customRequest('stackTrace', { threadId: stopped.thread, levels: 1 });
    assert.match(stack.stackFrames[0].name, /nacelle_cs_/);
    const first = stopped.sequence;
    await session.customRequest('stepIn', { threadId: stopped.thread, granularity: 'instruction' });
    await until(() => Boolean(stopped && stopped.sequence > first), 'real shader instruction step');
    assert.ok(stopped);
    await session.customRequest('evaluate', { expression: 'breakpoint modify --auto-continue true 1', context: 'repl' });
    await session.customRequest('continue', { threadId: stopped.thread, singleThread: false });
    await until(() => exited, 'native workload completion');
    assert.match(output, /NATIVE DEBUG OUTPUT PASS/);
    for (const mode of ['hlsl', 'optimized-hlsl']) {
      stopped = undefined;
      session = undefined;
      output = '';
      exited = false;
      assert.ok(await vscode.debug.startDebugging(undefined, {
        type: 'nacelle-lldb', request: 'launch', name: `${mode} shader smoke`, mode,
        program, args: ['unused', 'source-compute'], cwd: work,
        env: mode === 'optimized-hlsl' ? { NACELLE_TEST_SOURCE_OPTIMIZED_MODE: '1', NACELLE_DISABLE_TIER2: '0' } : {},
      }));
      await until(() => Boolean(stopped && session), `${mode} genuine source snapshot`);
      assert.ok(stopped && session);
      await vscode.commands.executeCommand('nacelle.hlslValues');
      await writeFile(join(work, 'stage.json'), JSON.stringify({ stage: 'source', mode }));
      await vscode.commands.executeCommand('nacelle.hlslLocation');
      assert.equal(
        vscode.window.activeTextEditor?.document.uri.fsPath,
        vscode.Uri.file(process.env.NACELLE_EXTENSION_SOURCE_FILE ?? '').fsPath,
      );
      await writeFile(join(work, 'stage.json'), JSON.stringify({ stage: 'lane', mode }));
      await vscode.commands.executeCommand('nacelle.hlslLane');
      await writeFile(join(work, 'stage.json'), JSON.stringify({ stage: 'debug', mode }));
      const steps = mode === 'hlsl' ? 2 : 1;
      for (let index = 0; index < steps; ++index) {
        const previous = currentStop().sequence;
        await vscode.commands.executeCommand('nacelle.hlslStep');
        await until(() => Boolean(stopped && stopped.sequence > previous), `${mode} real mapped source step`);
        assert.ok(stopped);
      }
      await vscode.commands.executeCommand('nacelle.hlslCancel');
      await until(() => exited, `${mode} preserved output after cancellation`);
      assert.match(output, mode === 'hlsl' ? /SOURCE DEBUG STEP PASS stops=3/ : /OPTIMIZED SOURCE TIER2 LOCAL STEP PASS/);
      assert.match(output, /SOURCE DEBUG OUTPUT PASS/);
      await assert.rejects(async () => { await vscode.commands.executeCommand('nacelle.hlslStep'); }, /No supported stopped shader session/);
    }
    await writeFile(join(work, 'stage.json'), JSON.stringify({ stage: 'complete' }));
  } finally {
    tracker.dispose();
    if (session && !exited) await vscode.debug.stopDebugging(session);
  }
}
