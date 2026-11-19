import { FileProfileDataSource } from './file-source';
import { loadProfile, type Profile, type ProfileDataSource } from './profile';
import { captureStatus, groupThreads, summaryText, type ThreadGroup } from './presentation';

export function mountViewer(
  root: Document,
  open: (file: File) => Promise<ProfileDataSource> = FileProfileDataSource.open,
): void {
  const input = root.querySelector<HTMLInputElement>('#profile-file')!;
  const status = root.querySelector<HTMLElement>('#status')!;
  const panel = root.querySelector<HTMLElement>('#profile')!;
  const summary = root.querySelector<HTMLElement>('#summary')!;
  const threads = root.querySelector<HTMLElement>('#threads')!;
  const exportButton = root.querySelector<HTMLButtonElement>('#export')!;
  const more = root.querySelector<HTMLButtonElement>('#more')!;
  let current: Profile | undefined;
  let groups: ThreadGroup[] = [];
  let generation = 0;
  let shown = 0;
  let total = 0;

  const element = <K extends keyof HTMLElementTagNameMap>(tag: K, text: string) => {
    const node = root.createElement(tag);
    node.textContent = text;
    return node;
  };

  function showNext(): void {
    const start = shown;
    shown = Math.min(shown + 250, total);
    let offset = 0;
    for (const group of groups) {
      const from = Math.max(0, start - offset);
      const to = Math.min(group.spans.length, shown - offset);
      offset += group.spans.length;
      if (from >= to) continue;
      const section = root.createElement('section');
      section.append(element('h3',
        `Process ${group.pid} / Thread ${group.tid}${group.name ? ` — ${group.name}` : ''}`));
      const table = root.createElement('table');
      table.append(element('caption', `Spans ${from + 1}–${to} of ${group.spans.length}`));
      const head = root.createElement('thead');
      const heading = root.createElement('tr');
      for (const label of ['Name', 'Category', 'Start (µs)', 'Duration (µs)', 'Arguments']) {
        const cell = element('th', label);
        cell.scope = 'col';
        heading.append(cell);
      }
      head.append(heading);
      table.append(head);
      const body = root.createElement('tbody');
      for (const event of group.spans.slice(from, to)) {
        const row = root.createElement('tr');
        for (const text of [event.name, event.cat, String(event.ts), String(event.dur), JSON.stringify(event.args)]) {
          row.append(element('td', text));
        }
        body.append(row);
      }
      table.append(body);
      section.append(table);
      threads.append(section);
    }
    more.hidden = shown >= total;
    more.textContent = `Show more spans (${shown} of ${total} shown)`;
  }

  input.addEventListener('change', async () => {
    const file = input.files?.[0];
    if (!file) return;
    const request = ++generation;
    current = undefined;
    groups = [];
    shown = 0;
    total = 0;
    exportButton.disabled = true;
    panel.hidden = true;
    more.hidden = true;
    summary.textContent = '';
    threads.replaceChildren();
    status.textContent = 'Reading local profile…';
    try {
      const source = await open(file);
      if (request !== generation) {
        source.close();
        return;
      }
      const profile = await loadProfile(source);
      if (request !== generation) return;
      current = profile;
      groups = groupThreads(profile.trace);
      total = groups.reduce((count, group) => count + group.spans.length, 0);
      summary.textContent = summaryText(profile);
      status.textContent = captureStatus(profile);
      status.dataset.state = profile.manifest.complete && !profile.manifest.droppedEvents ? 'complete' : 'incomplete';
      panel.hidden = false;
      exportButton.disabled = false;
      if (!total) threads.append(element('p', 'No timing spans in this capture.'));
      showNext();
    } catch (error) {
      if (request !== generation) return;
      status.textContent = `Invalid profile: ${error instanceof Error ? error.message : 'Unable to decode archive.'}`;
      status.dataset.state = 'invalid';
    } finally {
      if (request === generation) input.value = '';
    }
  });

  more.addEventListener('click', showNext);
  exportButton.addEventListener('click', () => {
    if (!current) return;
    const url = URL.createObjectURL(new Blob([current.traceBytes], { type: 'application/json' }));
    const link = root.createElement('a');
    link.href = url;
    link.download = 'nacelle-trace.json';
    root.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
