import { EditorState } from '@codemirror/state';
import { EditorView, lineNumbers, highlightActiveLine } from '@codemirror/view';
import type { Profile } from './profile';
import type { ArtifactKind, TextArtifact } from './artifacts';

export function createArtifactView(root: Document): { show(profile?: Profile): void; select(codeVersionId: number): void } {
  const section = root.querySelector<HTMLElement>('#shader-artifacts')!;
  const selector = root.querySelector<HTMLSelectElement>('#artifact-version')!;
  const status = root.querySelector<HTMLElement>('#artifact-status')!;
  const panes = root.querySelector<HTMLElement>('#artifact-panes')!;
  let current: Profile | undefined;
  let editors: EditorView[] = [];
  let selections: { payload: TextArtifact; editor: EditorView }[] = [];
  let navigating = false;

  function selectLine(editor: EditorView, line: number): void {
    if (line > editor.state.doc.lines) return;
    const position = editor.state.doc.line(line).from;
    editor.dispatch({ selection: { anchor: position }, effects: EditorView.scrollIntoView(position, { y: 'center' }) });
  }
  function navigate(payload: TextArtifact, line: number): void {
    if (navigating) return;
    navigating = true;
    try {
      const mapping = payload.mappings.find(item => item.line === line && item.source);
      const sourceId = payload.kind === 'hlsl' ? payload.id : mapping?.source;
      const sourceLine = payload.kind === 'hlsl' ? line : mapping?.sourceLine;
      if (!sourceId || !sourceLine) {
        status.textContent = 'No preserved source mapping for this line. LLVM/assembly and lane mappings may be unavailable.';
        return;
      }
      for (const item of selections) {
        if (item.payload.id === sourceId) selectLine(item.editor, sourceLine);
        else {
          const target = item.payload.mappings.find(candidate => candidate.source === sourceId && candidate.sourceLine === sourceLine);
          if (target) selectLine(item.editor, target.line);
        }
      }
      status.textContent = `Preserved HLSL/DXIL mapping: source line ${sourceLine}. Unmapped panes are unchanged; lane data unavailable.`;
    } finally {
      navigating = false;
    }
  }
  function render(): void {
    for (const editor of editors) editor.destroy();
    editors = [];
    selections = [];
    panes.replaceChildren();
    const metadata = current?.artifacts;
    const selection = metadata?.selections[Number(selector.value)];
    if (!selection || !metadata) return;
    status.textContent = `Shader: ${selection.shaderAvailability}; native: ${selection.nativeAvailability}. Mappings: partial; lanes unavailable.`;
    for (const kind of ['hlsl', 'dxil', 'mlir', 'llvm', 'assembly'] satisfies ArtifactKind[]) {
      const payloads = metadata.payloads.filter(payload => payload.kind === kind && selection.payloadIds.includes(payload.id));
      const pane = root.createElement('section');
      pane.dataset.artifactKind = kind;
      pane.className = 'min-w-0 rounded border border-[var(--border)] p-2';
      const heading = root.createElement('h3');
      heading.textContent = kind.toUpperCase();
      pane.append(heading);
      if (!payloads.length) {
        const message = root.createElement('p');
        message.textContent = kind === 'hlsl' ? 'Source not embedded (stripped, not retained, or consent disabled).' :
          'Unavailable for this version. No mapping is inferred.';
        pane.append(message);
      }
      for (const payload of payloads) {
        const label = root.createElement('p');
        label.textContent = payload.label;
        pane.append(label);
        const host = root.createElement('div');
        pane.append(host);
        const editor = new EditorView({
          parent: host,
          state: EditorState.create({ doc: payload.text, extensions: [
            EditorState.readOnly.of(true), EditorView.editable.of(false),
            EditorView.cspNonce.of('nacelle-editor'), lineNumbers(), highlightActiveLine(),
            EditorView.theme({ '&': { maxHeight: '360px' }, '.cm-scroller': { overflow: 'auto' } }),
            EditorView.updateListener.of(update => {
              if (update.selectionSet) navigate(payload, update.state.doc.lineAt(update.state.selection.main.head).number);
            }),
          ] }),
        });
        editors.push(editor);
        selections.push({ payload, editor });
      }
      panes.append(pane);
    }
  }
  selector.addEventListener('change', render);
  return {
    select(codeVersionId) {
      if (!current?.artifacts) return;
      selector.value = String(codeVersionId - 1);
      render();
      section.scrollIntoView?.({ block: 'nearest' });
    },
    show(profile) {
      current = profile;
      selector.replaceChildren();
      section.hidden = !profile?.artifacts;
      for (const [index, code] of (profile?.shaders.codeVersions ?? []).entries()) {
        const option = root.createElement('option');
        option.value = String(index);
        option.textContent = `Pipeline ${code.pipelineId} · ${code.entryRole ?? code.stage} · version ${code.version} · Tier ${code.tier} · ${code.cache} · SIMD ${code.simdWidth}`;
        selector.append(option);
      }
      render();
    },
  };
}
