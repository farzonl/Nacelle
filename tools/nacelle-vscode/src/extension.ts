import * as vscode from 'vscode';
import { ProfileEditor } from './editor.ts';
import { ShaderDebugger } from './debugger.ts';

export function activate(context: vscode.ExtensionContext) {
  const output = vscode.window.createOutputChannel('Nacelle');
  const editor = new ProfileEditor(context, output);
  const debuggerController = new ShaderDebugger(context, output);
  context.subscriptions.push(output,
    vscode.window.registerCustomEditorProvider('nacelle.profile', editor, {
      supportsMultipleEditorsPerDocument: false,
      webviewOptions: { retainContextWhenHidden: true },
    }));
  debuggerController.register();
  const commands: Record<string, () => void | Promise<void>> = {
    'nacelle.exportTrace': () => editor.exportTrace(),
    'nacelle.debugCapabilities': () => { output.appendLine(debuggerController.capabilities()); output.show(true); },
    'nacelle.hlslLocation': () => debuggerController.navigateSource(),
    'nacelle.hlslLane': () => debuggerController.inspectLane(),
    'nacelle.hlslValues': async () => { await debuggerController.inspect('hlsl-values'); await debuggerController.inspect('hlsl-code'); },
    'nacelle.hlslStep': () => debuggerController.control('step'),
    'nacelle.hlslContinue': () => debuggerController.control('continue'),
    'nacelle.hlslCancel': () => debuggerController.control('cancel'),
  };
  for (const [name, command] of Object.entries(commands)) {
    context.subscriptions.push(vscode.commands.registerCommand(name, async () => {
      try { await command(); }
      catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        output.appendLine(`${name}: ${reason}`);
        void vscode.window.showErrorMessage(`Nacelle: ${reason}`);
        throw error;
      }
    }));
  }
  return { getEditorStatus: (uri: vscode.Uri) => editor.getStatus(uri) };
}
