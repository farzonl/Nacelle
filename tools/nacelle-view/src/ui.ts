import { decodeProfileFile } from './worker-loader';
import { createArtifactView } from './artifact-view';
import { createSpatialView } from './spatial-view';
import {
  fetchPreloadedManifest,
  openPreloadedProfile,
  validatePreloadedManifest,
  type PreloadedManifestInput,
  type PreloadedPlatformManifest,
  type PreloadedProfileEntry,
} from './preloaded';
import { LIMITS, loadProfile, requireValid, type Profile, type ProfileDataSource } from './profile';
import { captureStatus, filterThreads, groupThreads, summaryText, type SpanFilter, type ThreadGroup } from './presentation';
import { speedscopeProfile } from './speedscope';
import type { TimelineSelectionEvent, TimelineView } from './timeline';

export interface ViewerOptions {
  preloadedEnabled?: boolean;
  loadPreloadedManifest?: () => Promise<PreloadedManifestInput>;
  openPreloadedProfile?: (file: string) => Promise<ProfileDataSource>;
  timelineView?: TimelineView;
  decodeFile?: typeof decodeProfileFile;
  saveFile?: (blob: Blob, name: string) => Promise<void>;
}

export interface ViewerController {
  openSource(name: string, read: (signal: AbortSignal) => Promise<Blob>): Promise<void>;
  exportTrace(): Promise<void>;
  exportSpeedscope(): Promise<void>;
  cancel(): void;
  dispose(): void;
}
const THEME_STORAGE_KEY = 'nacelle-viewer-theme';

function systemPrefersDark(win: Window | null): boolean {
  return Boolean(win?.matchMedia?.('(prefers-color-scheme: dark)').matches);
}

function getStoredTheme(): 'light' | 'dark' | null {
  try {
    const val = localStorage.getItem(THEME_STORAGE_KEY);
    return val === 'light' || val === 'dark' ? val : null;
  } catch {
    return null;
  }
}

function applyThemeMeta(root: Document, scheme: 'light' | 'dark' | null): void {
  const meta = root.querySelector<HTMLMetaElement>('meta[name="color-scheme"]');
  if (meta) meta.content = scheme ?? 'light dark';
}

export function setupThemeToggle(root: Document): void {
  const win = root.defaultView;
  applyThemeMeta(root, getStoredTheme());
  const toggle = root.querySelector<HTMLButtonElement>('#theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const stored = getStoredTheme();
      const prefersDark = systemPrefersDark(win);
      const rendered = stored ?? (prefersDark ? 'dark' : 'light');
      const target: 'light' | 'dark' = rendered === 'dark' ? 'light' : 'dark';
      const matchesSystem = target === (prefersDark ? 'dark' : 'light');
      try {
        if (matchesSystem) localStorage.removeItem(THEME_STORAGE_KEY);
        else localStorage.setItem(THEME_STORAGE_KEY, target);
      } catch {
        // Apply in-memory even if storage is blocked.
      }
      applyThemeMeta(root, matchesSystem ? null : target);
    });
  }
  win?.addEventListener('storage', event => {
    if (event.key === THEME_STORAGE_KEY) {
      applyThemeMeta(root, getStoredTheme());
    }
  });
}

function formatDurationUs(us: number): string {
  if (!Number.isFinite(us)) return '0';
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms`;
  return `${us.toFixed(1)} µs`;
}

function categoryTone(cat: string): string {
  const lower = cat.toLowerCase();
  if (lower.includes('compile') || lower.includes('jit') || lower.includes('mlir') || lower.includes('dxil')) {
    return 'accent';
  }
  if (lower.includes('compute') || lower.includes('dispatch')) return 'violet';
  if (lower.includes('graphics') || lower.includes('draw') || lower.includes('mesh') || lower.includes('raster')) {
    return 'ok';
  }
  if (lower.includes('worker') || lower.includes('async')) return 'warn';
  return 'neutral';
}

export function mountViewer(
  root: Document,
  open?: (file: File) => Promise<ProfileDataSource>,
  options: ViewerOptions = {},
): ViewerController {
  setupThemeToggle(root);
  const artifacts = createArtifactView(root);

  const input = root.querySelector<HTMLInputElement>('#profile-file')!;
  const dropzone = root.querySelector<HTMLElement>('#dropzone');
  const status = root.querySelector<HTMLElement>('#status')!;
  const cancel = root.querySelector<HTMLButtonElement>('#cancel-load')!;
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

  const modeBadge = root.querySelector<HTMLElement>('#mode-badge');
  const benchmarkBar = root.querySelector<HTMLElement>('#benchmark-bar');
  const benchmarkOsToggle = root.querySelector<HTMLElement>('#benchmark-os-toggle');
  const benchmarkSelect = root.querySelector<HTMLSelectElement>('#benchmark-select');
  const benchmarkPrev = root.querySelector<HTMLButtonElement>('#benchmark-prev');
  const benchmarkNext = root.querySelector<HTMLButtonElement>('#benchmark-next');
  const benchmarkMeta = root.querySelector<HTMLElement>('#benchmark-meta');

  const kpiSpans = root.querySelector<HTMLElement>('#kpi-spans');
  const kpiWindow = root.querySelector<HTMLElement>('#kpi-window');
  const kpiEvents = root.querySelector<HTMLElement>('#kpi-events');
  const kpiDropped = root.querySelector<HTMLElement>('#kpi-dropped');
  const kpiThreads = root.querySelector<HTMLElement>('#kpi-threads');
  const kpiClock = root.querySelector<HTMLElement>('#kpi-clock');
  const kpiShaders = root.querySelector<HTMLElement>('#kpi-shaders');
  const kpiTiers = root.querySelector<HTMLElement>('#kpi-tiers');

  const timelineSection = root.querySelector<HTMLElement>('#timeline-section');
  const timelineHost = root.querySelector<HTMLElement>('#timeline-host');
  const timelineRenderer = root.querySelector<HTMLElement>('#timeline-renderer');
  const timelineExpand = root.querySelector<HTMLButtonElement>('#timeline-expand');
  const timelineView = options.timelineView;

  let current: Profile | undefined;
  let activeTitle = 'Nacelle Profile';
  let allGroups: ThreadGroup[] = [];
  let groups: ThreadGroup[] = [];
  let filtered = false;
  let generation = 0;
  let decoding: AbortController | undefined;
  let shown = 0;
  let total = 0;
  let preloadedPlatforms: Record<string, PreloadedPlatformManifest> = {};
  let activePlatformKey = '';
  let preloadedEntries: PreloadedProfileEntry[] = [];
  let tableCounter = 0;
  const spatial = createSpatialView(root, selected => {
    if (!current) return;
    artifacts.select(selected.codeVersionId);
    const span = current.trace.traceEvents.find(event => event.args.id === selected.eventId);
    startFilter.value = '';
    endFilter.value = '';
    applySelectionFilter({ pso: selected.pso, stage: selected.stage, thread: span ? `${span.pid}:${span.tid}` : '',
      start: span?.ts, end: span ? span.ts + span.dur : undefined });
  });

  function applySelectionFilter(selection: TimelineSelectionEvent): void {
    if (selection.reset) {
      filterForm.reset();
      applyFilters();
      return;
    }
    if (selection.thread !== undefined) threadFilter.value = selection.thread;
    if (selection.pso !== undefined) psoFilter.value = selection.pso;
    if (selection.stage !== undefined) stageFilter.value = selection.stage;
    if (selection.start !== undefined) startFilter.value = String(selection.start);
    if (selection.end !== undefined) endFilter.value = String(selection.end);
    applyFilters();
  }

  if (timelineView && timelineSection && timelineHost) {
    timelineView.mount(timelineHost, { onSelectFilter: applySelectionFilter });
    if (timelineRenderer) timelineRenderer.textContent = timelineView.capabilities.rendererName;
    timelineSection.hidden = false;
    timelineExpand?.addEventListener('click', () => {
      const expanded = timelineExpand.getAttribute('aria-pressed') === 'true';
      const next = !expanded;
      timelineExpand.setAttribute('aria-pressed', String(next));
      timelineExpand.textContent = next ? 'Compact height' : 'Expand height';
      timelineHost.classList.toggle('h-[520px]', !next);
      timelineHost.classList.toggle('h-[78vh]', next);
    });
  }

  const element = <K extends keyof HTMLElementTagNameMap>(tag: K, text?: string, className?: string) => {
    const node = root.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };

  function option(parent: HTMLSelectElement | HTMLOptGroupElement, value: string, label: string): void {
    const item = element('option', label);
    item.value = value;
    parent.append(item);
  }

  function updateKpis(profile: Profile, threadGroups: ThreadGroup[]): void {
    const timingSpans = profile.trace.traceEvents.filter(e => e.ph === 'X');
    let minTs = Infinity;
    let maxEnd = -Infinity;
    for (const span of timingSpans) {
      if (span.ts < minTs) minTs = span.ts;
      if (span.ts + span.dur > maxEnd) maxEnd = span.ts + span.dur;
    }
    const spanWindow = timingSpans.length > 0 && Number.isFinite(minTs) && Number.isFinite(maxEnd)
      ? `Window: ${formatDurationUs(Math.max(0, maxEnd - minTs))}`
      : 'No timing spans';

    if (kpiSpans) kpiSpans.textContent = timingSpans.length.toLocaleString();
    if (kpiWindow) kpiWindow.textContent = spanWindow;

    if (kpiEvents) kpiEvents.textContent = profile.manifest.recordedEvents.toLocaleString();
    if (kpiDropped) {
      kpiDropped.textContent = profile.manifest.droppedEvents > 0
        ? `${profile.manifest.droppedEvents.toLocaleString()} dropped`
        : profile.manifest.complete ? '0 dropped · Complete' : '0 event drops · Spatial data incomplete';
    }

    const pids = new Set(threadGroups.map(g => g.pid));
    if (kpiThreads) kpiThreads.textContent = `${pids.size} / ${threadGroups.length}`;
    if (kpiClock) kpiClock.textContent = `Clock: ${profile.manifest.clock} (${profile.manifest.timestampUnit})`;

    const codes = profile.shaders.codeVersions;
    if (kpiShaders) kpiShaders.textContent = codes.length.toLocaleString();
    if (kpiTiers) {
      if (codes.length === 0) {
        kpiTiers.textContent = 'No code versions';
      } else {
        const tier1 = codes.filter(c => c.tier === 1).length;
        const tier2 = codes.filter(c => c.tier === 2).length;
        const hits = codes.filter(c => c.cache === 'hit').length;
        kpiTiers.textContent = `T1: ${tier1} · T2: ${tier2} · Cache hits: ${hits}`;
      }
    }
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
      if (!total) {
        const empty = element('div', undefined, 'panel rounded-xl p-6 text-center text-sm text-[var(--muted)]');
        empty.append(element('p', 'No timing spans match the current filters.'));
        threads.append(empty);
      }
      timelineView?.render({ profile: current, groups, filtered, title: activeTitle });
      showNext();
    } catch (error) {
      filterStatus.textContent = `Invalid filters: ${error instanceof Error ? error.message : 'Unable to filter spans.'}`;
      timelineView?.clear();
      artifacts.show();
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

      const slice = group.spans.slice(from, to);
      const maxDur = group.spans.reduce((m, s) => (s.dur > m ? s.dur : m), 0);
      const captionId = `thread-caption-${++tableCounter}`;

      const section = element('section', undefined, 'panel overflow-hidden rounded-xl shadow-sm');

      const header = element('div', undefined,
        'flex flex-wrap items-center justify-between gap-2 bg-[var(--surface-elevated)]/70 px-4 py-3');
      const titleGroup = element('div', undefined, 'flex flex-wrap items-center gap-2');
      titleGroup.append(element('span', `PID ${group.pid} · TID ${group.tid}`,
        'font-mono-tabular rounded-md border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5 text-[11px] font-semibold text-[var(--accent)]'));
      titleGroup.append(element('h3',
        `Process ${group.pid} / Thread ${group.tid}${group.name ? ` — ${group.name}` : ''}`,
        'text-sm font-semibold text-[var(--text)]'));
      header.append(titleGroup);

      const metaBadges = element('div', undefined, 'flex items-center gap-2');
      metaBadges.append(element('span', `${group.spans.length.toLocaleString()} spans`,
        'font-mono-tabular rounded-full border border-[var(--border)] bg-[var(--surface)] px-2.5 py-0.5 text-[11px] text-[var(--muted)]'));
      if (maxDur > 0) {
        metaBadges.append(element('span', `Peak ${formatDurationUs(maxDur)}`,
          'font-mono-tabular rounded-full border border-[var(--border)] bg-[var(--surface)] px-2.5 py-0.5 text-[11px] text-[var(--muted)]'));
      }
      header.append(metaBadges);
      section.append(header);

      const wrapper = element('div', undefined, 'table-wrapper');
      wrapper.tabIndex = 0;
      wrapper.setAttribute('role', 'region');
      wrapper.setAttribute('aria-labelledby', captionId);

      const table = element('table', undefined, 'span-table');
      const caption = element('caption', `Spans ${from + 1}–${to} of ${group.spans.length}`);
      caption.id = captionId;
      table.append(caption);

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
      for (const event of slice) {
        const row = root.createElement('tr');

        const nameCell = element('th', event.name);
        nameCell.scope = 'row';
        row.append(nameCell);

        const catCell = root.createElement('td');
        const badge = element('span', event.cat || 'default', 'cat-badge');
        badge.dataset.tone = categoryTone(event.cat);
        catCell.append(badge);
        row.append(catCell);

        const startCell = element('td', String(event.ts), 'font-mono-tabular text-[var(--muted)] whitespace-nowrap');
        row.append(startCell);

        const durCell = element('td', undefined, 'duration-cell font-mono-tabular');
        durCell.append(element('div', String(event.dur), 'font-semibold text-[var(--text)]'));
        if (maxDur > 0) {
          const track = element('div', undefined, 'duration-bar-track');
          const fill = element('div', undefined, 'duration-bar-fill');
          const pct = Math.max(2, Math.min(100, (event.dur / maxDur) * 100));
          fill.style.setProperty('--bar-pct', `${pct.toFixed(1)}%`);
          track.append(fill);
          durCell.append(track);
        }
        row.append(durCell);

        const argsCell = element('td', JSON.stringify(event.args), 'args-cell');
        row.append(argsCell);

        body.append(row);
      }
      table.append(body);
      wrapper.append(table);
      section.append(wrapper);
      threads.append(section);
    }
    more.hidden = shown >= total;
    more.textContent = `Show more spans (${shown} of ${total} shown)`;
  }

  async function loadFromSource(
    title: string,
    loadingMessage: string,
    getProfile: (signal: AbortSignal) => Promise<Profile>,
    onFinally?: () => void,
  ): Promise<void> {
    const request = ++generation;
    decoding?.abort();
    const controller = new AbortController();
    decoding = controller;
    cancel.hidden = false;
    current = undefined;
    spatial.show();
    artifacts.show();
    activeTitle = title;
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
    timelineView?.clear();
    status.textContent = loadingMessage;
    status.dataset.state = 'loading';
    try {
      const profile = await getProfile(controller.signal);
      if (request !== generation) return;
      current = profile;
      artifacts.show(profile);
      spatial.show(profile);
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
      updateKpis(profile, allGroups);
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
      if (request === generation) {
        decoding = undefined;
        cancel.hidden = true;
        onFinally?.();
      }
    }
  }

  function loadLocalFile(file: File): void {
    if (benchmarkSelect) benchmarkSelect.value = '';
    updateBenchmarkNavButtons();
    void loadFromSource(file.name || 'Local Profile', 'Reading local profile…', async signal => {
      if (!open) return (options.decodeFile ?? decodeProfileFile)(file, signal);
      const source = await open(file);
      if (signal.aborted) {
        source.close();
        signal.throwIfAborted();
      }
      return loadProfile(source);
    }, () => {
      input.value = '';
    });
  }

  function cancelLoad(): void {
    ++generation;
    decoding?.abort();
    decoding = undefined;
    cancel.hidden = true;
    input.value = '';
    status.textContent = 'Profile loading cancelled.';
    status.dataset.state = 'cancelled';
  }
  cancel.addEventListener('click', cancelLoad);

  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (!file) return;
    loadLocalFile(file);
  });

  if (dropzone) {
    const setDragging = (active: boolean) => {
      dropzone.dataset.dragging = active ? 'true' : 'false';
    };
    dropzone.addEventListener('dragover', event => {
      event.preventDefault();
      setDragging(true);
    });
    dropzone.addEventListener('dragleave', () => setDragging(false));
    dropzone.addEventListener('drop', event => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer?.files?.[0];
      if (file) loadLocalFile(file);
    });
  }

  function updateBenchmarkNavButtons(): void {
    if (!benchmarkSelect || !benchmarkPrev || !benchmarkNext) return;
    const idx = preloadedEntries.findIndex(entry => entry.id === benchmarkSelect.value);
    benchmarkPrev.disabled = idx <= 0;
    benchmarkNext.disabled = idx < 0 || idx >= preloadedEntries.length - 1;
  }

  function syncQueryParams(shaderId: string, platformKey?: string): void {
    const win = root.defaultView;
    if (!win?.history?.replaceState || !win.location) return;
    try {
      const url = new URL(win.location.href);
      if (platformKey) url.searchParams.set('os', platformKey);
      url.searchParams.set('shader', shaderId);
      win.history.replaceState(null, '', url.toString());
    } catch {
      // Ignore in opaque/non-standard origins (e.g., jsdom tests).
    }
  }

  function updateOsToggleButtons(): void {
    if (!benchmarkOsToggle) return;
    const buttons = benchmarkOsToggle.querySelectorAll<HTMLButtonElement>('button[data-platform]');
    for (const btn of buttons) {
      const active = btn.dataset.platform === activePlatformKey;
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
      btn.className = active
        ? 'profiler-btn px-3 py-2 text-xs font-semibold bg-[var(--accent-soft)] text-[var(--accent-strong)] transition'
        : 'profiler-btn px-3 py-2 text-xs font-medium text-[var(--muted)] transition hover:bg-[var(--surface-hover)] hover:text-[var(--text)]';
    }
  }

  function selectBenchmarkShader(id: string): void {
    const entry = preloadedEntries.find(item => item.id === id);
    if (!entry) return;
    if (benchmarkSelect) benchmarkSelect.value = entry.id;
    updateBenchmarkNavButtons();
    syncQueryParams(entry.id, activePlatformKey || undefined);
    const plat = activePlatformKey ? preloadedPlatforms[activePlatformKey] : undefined;
    const title = plat?.os ? `${entry.name} (${plat.os})` : entry.name;
    const openPreloaded = options.openPreloadedProfile;
    void loadFromSource(title, `Loading benchmark profile: ${title}…`, async signal => {
      if (openPreloaded) {
        const source = await openPreloaded(entry.file);
        if (signal.aborted) {
          source.close();
          signal.throwIfAborted();
        }
        return loadProfile(source);
      }
      return openPreloadedProfile(entry.file, signal);
    });
  }

  function selectBenchmarkPlatform(platformKey: string, preferredShaderId?: string | null): void {
    const plat = preloadedPlatforms[platformKey];
    if (!plat || !benchmarkSelect) return;
    const prevShaderId = preferredShaderId ?? (benchmarkSelect.value || null);
    activePlatformKey = platformKey;
    preloadedEntries = plat.profiles;
    updateOsToggleButtons();

    benchmarkSelect.replaceChildren();
    option(benchmarkSelect, '', '— Local file —');
    const byCategory = new Map<string, PreloadedProfileEntry[]>();
    for (const entry of preloadedEntries) {
      const list = byCategory.get(entry.category) ?? [];
      list.push(entry);
      byCategory.set(entry.category, list);
    }
    for (const [category, entries] of byCategory) {
      const group = root.createElement('optgroup');
      group.label = category;
      for (const entry of entries) {
        option(group, entry.id, `${entry.name} (${entry.id})`);
      }
      benchmarkSelect.append(group);
    }

    if (benchmarkMeta) {
      const parts: string[] = [];
      if (plat.os) parts.push(plat.os);
      if (plat.simdWidth) parts.push(`SIMD-${plat.simdWidth}`);
      if (plat.shortSha) parts.push(`commit ${plat.shortSha}`);
      parts.push(`${preloadedEntries.length} shaders`);
      benchmarkMeta.textContent = parts.join(' · ');
    }

    const initial = (prevShaderId && preloadedEntries.find(e => e.id === prevShaderId))
      ?? preloadedEntries.find(e => e.id === 'bench_raymarch')
      ?? preloadedEntries[0];
    if (initial) selectBenchmarkShader(initial.id);
    else updateBenchmarkNavButtons();
  }

  async function initPreloadedProfiles(): Promise<void> {
    if (!options.preloadedEnabled || !benchmarkBar || !benchmarkSelect) return;
    const loadManifest = options.loadPreloadedManifest ?? fetchPreloadedManifest;
    try {
      const manifest = validatePreloadedManifest(await loadManifest());
      const keys = Object.keys(manifest.platforms).sort((a, b) => {
        const order = (k: string) => (k === 'windows' ? 0 : k === 'macos' ? 1 : 2);
        const diff = order(a) - order(b);
        return diff !== 0 ? diff : a.localeCompare(b);
      });
      if (!keys.length) return;
      preloadedPlatforms = manifest.platforms;

      if (benchmarkOsToggle) {
        benchmarkOsToggle.replaceChildren();
        keys.forEach((key, index) => {
          if (index > 0) {
            const divider = root.createElement('span');
            divider.className = 'h-4 w-px bg-[var(--border)]';
            divider.setAttribute('aria-hidden', 'true');
            benchmarkOsToggle.append(divider);
          }
          const btn = element('button', preloadedPlatforms[key].os);
          btn.type = 'button';
          btn.dataset.platform = key;
          btn.addEventListener('click', () => {
            if (activePlatformKey !== key) selectBenchmarkPlatform(key);
          });
          benchmarkOsToggle.append(btn);
        });
        benchmarkOsToggle.hidden = keys.length === 0;
      }

      if (modeBadge) modeBadge.textContent = 'Benchmark Suite';
      benchmarkBar.hidden = false;

      benchmarkSelect.addEventListener('change', () => {
        if (benchmarkSelect.value) selectBenchmarkShader(benchmarkSelect.value);
        else updateBenchmarkNavButtons();
      });
      benchmarkPrev?.addEventListener('click', () => {
        const idx = preloadedEntries.findIndex(e => e.id === benchmarkSelect.value);
        if (idx > 0) selectBenchmarkShader(preloadedEntries[idx - 1].id);
      });
      benchmarkNext?.addEventListener('click', () => {
        const idx = preloadedEntries.findIndex(e => e.id === benchmarkSelect.value);
        if (idx >= 0 && idx < preloadedEntries.length - 1) selectBenchmarkShader(preloadedEntries[idx + 1].id);
      });

      const params = root.defaultView?.location
        ? new URLSearchParams(root.defaultView.location.search)
        : null;
      const requestedOs = params?.get('os')?.trim().toLowerCase() ?? '';
      const requestedShader = params?.get('shader') ?? null;
      const initialPlatform = (requestedOs && requestedOs in preloadedPlatforms)
        ? requestedOs
        : (manifest.defaultPlatform && manifest.defaultPlatform in preloadedPlatforms ? manifest.defaultPlatform : keys[0]);
      selectBenchmarkPlatform(initialPlatform, requestedShader);
    } catch {
      // Preloaded profiles are optional CI-staged assets; keep local viewer usable if unavailable.
      benchmarkBar.hidden = true;
    }
  }

  void initPreloadedProfiles();

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

  async function exportTrace(): Promise<void> {
    if (!current) return;
    const blob = new Blob([current.traceBytes], { type: 'application/json' });
    try {
      if (options.saveFile) await options.saveFile(blob, 'nacelle-trace.json');
      else download(blob, 'nacelle-trace.json');
    } catch (error) {
      exportStatus.textContent = `Trace export failed: ${error instanceof Error ? error.message : 'Unable to export trace.'}`;
      throw error;
    }
  }
  exportButton.addEventListener('click', () => {
    void exportTrace().catch(error => console.error('Trace export failed.', error));
  });

  async function exportSpeedscope(): Promise<void> {
    if (!current || !total) return;
    exportStatus.textContent = '';
    try {
      const json = JSON.stringify(speedscopeProfile(current, groups, filtered));
      const blob = new Blob([json], { type: 'application/json' });
      requireValid(blob.size <= LIMITS.entry, 'Speedscope export exceeds the 256 MiB size limit.');
      if (options.saveFile) await options.saveFile(blob, 'nacelle-speedscope.json');
      else download(blob, 'nacelle-speedscope.json');
      exportStatus.textContent = 'Saved a local Speedscope profile. Open it in your locally installed Speedscope viewer; nothing was uploaded.';
    } catch (error) {
      exportStatus.textContent = `Speedscope export failed: ${error instanceof Error ? error.message : 'Unable to export spans.'}`;
      throw error;
    }
  }
  speedscopeButton.addEventListener('click', () => {
    void exportSpeedscope().catch(error => console.error('Speedscope export failed.', error));
  });
  return {
    openSource: (name, read) => loadFromSource(name, 'Reading selected profile...', async signal => {
      const blob = await read(signal);
      signal.throwIfAborted();
      return (options.decodeFile ?? decodeProfileFile)(blob, signal);
    }),
    exportTrace,
    exportSpeedscope,
    cancel: cancelLoad,
    dispose: () => {
      cancelLoad();
      current = undefined;
      artifacts.show();
      spatial.show();
      timelineView?.dispose();
    },
  };
}
