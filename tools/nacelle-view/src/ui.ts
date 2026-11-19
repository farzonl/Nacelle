import { FileProfileDataSource } from './file-source';
import { LIMITS, loadProfile, requireValid, type Profile, type ProfileDataSource } from './profile';
import { captureStatus, filterThreads, groupThreads, summaryText, type SpanFilter, type ThreadGroup } from './presentation';
import { speedscopeProfile } from './speedscope';

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
  const speedscopeButton = root.querySelector<HTMLButtonElement>('#export-speedscope')!;
  const exportStatus = root.querySelector<HTMLElement>('#export-status')!;
  const filterForm = root.querySelector<HTMLFormElement>('#filters')!;
  const threadFilter = root.querySelector<HTMLSelectElement>('#thread-filter')!;
  const psoFilter = root.querySelector<HTMLInputElement>('#pso-filter')!;
  const stageFilter = root.querySelector<HTMLSelectElement>('#stage-filter')!;
  const startFilter = root.querySelector<HTMLInputElement>('#start-filter')!;
  const endFilter = root.querySelector<HTMLInputElement>('#end-filter')!;
  const filterStatus = root.querySelector<HTMLElement>('#filter-status')!;
  const more = root.querySelector<HTMLButtonElement>('#more')!;
  let current: Profile | undefined;
  let allGroups: ThreadGroup[] = [];
  let groups: ThreadGroup[] = [];
  let filtered = false;
  let generation = 0;
  let shown = 0;
  let total = 0;

  const element = <K extends keyof HTMLElementTagNameMap>(tag: K, text: string) => {
    const node = root.createElement(tag);
    node.textContent = text;
    return node;
  };

  function option(select: HTMLSelectElement, value: string, label: string): void {
    const item = element('option', label);
    item.value = value;
    select.append(item);
  }

  function applyFilters(): void {
    if (!current) return;
    threads.replaceChildren();
    shown = 0;
    total = 0;
    groups = [];
    more.hidden = true;
    speedscopeButton.disabled = true;
    exportStatus.textContent = '';
    try {
      const filter: SpanFilter = {
        thread: threadFilter.value, pso: psoFilter.value.trim(), stage: stageFilter.value,
        start: startFilter.value === '' ? undefined : Number(startFilter.value),
        end: endFilter.value === '' ? undefined : Number(endFilter.value),
      };
      groups = filterThreads(allGroups, filter);
      filtered = Boolean(filter.thread || filter.pso || filter.stage
        || filter.start !== undefined || filter.end !== undefined);
      total = groups.reduce((count, group) => count + group.spans.length, 0);
      const original = allGroups.reduce((count, group) => count + group.spans.length, 0);
      filterStatus.textContent = `${total} of ${original} timing spans match. Durations remain inclusive wall time, not CPU time.`;
      speedscopeButton.disabled = total === 0;
      if (!total) threads.append(element('p', 'No timing spans match the current filters.'));
      showNext();
    } catch (error) {
      filterStatus.textContent = `Invalid filters: ${error instanceof Error ? error.message : 'Unable to filter spans.'}`;
    }
  }

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
    allGroups = [];
    groups = [];
    shown = 0;
    total = 0;
    exportButton.disabled = true;
    speedscopeButton.disabled = true;
    exportStatus.textContent = '';
    filterStatus.textContent = '';
    filterForm.reset();
    threadFilter.replaceChildren();
    stageFilter.replaceChildren();
    option(threadFilter, '', 'All threads');
    option(stageFilter, '', 'All stages');
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
      allGroups = groupThreads(profile.trace);
      const stages = new Set<string>();
      for (const group of allGroups) {
        option(threadFilter, `${group.pid}:${group.tid}`,
          `Process ${group.pid} / Thread ${group.tid}${group.name ? ` — ${group.name}` : ''}`);
        for (const span of group.spans) {
          if (typeof span.args.stage === 'string' && span.args.stage) stages.add(span.args.stage);
        }
      }
      for (const stage of [...stages].sort()) option(stageFilter, stage, stage);
      summary.textContent = summaryText(profile);
      status.textContent = captureStatus(profile);
      status.dataset.state = profile.manifest.complete && !profile.manifest.droppedEvents ? 'complete' : 'incomplete';
      panel.hidden = false;
      exportButton.disabled = false;
      applyFilters();
    } catch (error) {
      if (request !== generation) return;
      status.textContent = `Invalid profile: ${error instanceof Error ? error.message : 'Unable to decode archive.'}`;
      status.dataset.state = 'invalid';
    } finally {
      if (request === generation) input.value = '';
    }
  });

  filterForm.addEventListener('submit', event => {
    event.preventDefault();
    applyFilters();
  });
  root.querySelector<HTMLButtonElement>('#clear-filters')!.addEventListener('click', () => {
    filterForm.reset();
    applyFilters();
  });
  more.addEventListener('click', showNext);
  function download(blob: Blob, name: string): void {
    const url = URL.createObjectURL(blob);
    const link = root.createElement('a');
    link.href = url;
    link.download = name;
    root.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  exportButton.addEventListener('click', () => {
    if (!current) return;
    download(new Blob([current.traceBytes], { type: 'application/json' }), 'nacelle-trace.json');
  });
  speedscopeButton.addEventListener('click', () => {
    if (!current || !total) return;
    exportStatus.textContent = '';
    try {
      const json = JSON.stringify(speedscopeProfile(current, groups, filtered));
      const blob = new Blob([json], { type: 'application/json' });
      requireValid(blob.size <= LIMITS.entry, 'Speedscope export exceeds the 256 MiB size limit.');
      download(blob, 'nacelle-speedscope.json');
      exportStatus.textContent = 'Saved a local Speedscope profile. Open it in your locally installed Speedscope viewer; nothing was uploaded.';
    } catch (error) {
      exportStatus.textContent = `Speedscope export failed: ${error instanceof Error ? error.message : 'Unable to export spans.'}`;
    }
  });
}
