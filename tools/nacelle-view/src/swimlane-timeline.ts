import type { ThreadGroup } from './presentation';
import type { TraceEvent } from './profile';
import type {
  TimelineRenderInput,
  TimelineView,
  TimelineViewCallbacks,
  TimelineViewCapabilities,
} from './timeline';

export interface PlacedSpan {
  span: TraceEvent;
  groupIndex: number;
  depth: number;
  y: number;
  height: number;
}

export interface SwimlaneLane {
  group: ThreadGroup;
  groupIndex: number;
  label: string;
  top: number;
  headerHeight: number;
  height: number;
  rowCount: number;
  spans: PlacedSpan[];
}

export interface SwimlaneLayout {
  lanes: SwimlaneLane[];
  minTs: number;
  maxEnd: number;
  totalDuration: number;
  totalHeight: number;
  totalSpans: number;
}

const RULER_HEIGHT = 28;
const LANE_HEADER_HEIGHT = 24;
const ROW_HEIGHT = 20;
const ROW_GAP = 2;
const LANE_PADDING_BOTTOM = 8;

export function formatTimeUs(us: number): string {
  if (!Number.isFinite(us)) return '0 µs';
  const abs = Math.abs(us);
  if (abs >= 1_000_000) return `${(us / 1_000_000).toFixed(2)} s`;
  if (abs >= 1_000) return `${(us / 1_000).toFixed(2)} ms`;
  if (abs >= 10) return `${us.toFixed(1)} µs`;
  return `${us.toFixed(2)} µs`;
}

export function computeSwimlaneLayout(groups: ThreadGroup[]): SwimlaneLayout {
  let minTs = Infinity;
  let maxEnd = -Infinity;
  let totalSpans = 0;

  for (const group of groups) {
    for (const span of group.spans) {
      totalSpans++;
      if (span.ts < minTs) minTs = span.ts;
      const end = span.ts + span.dur;
      if (end > maxEnd) maxEnd = end;
    }
  }

  if (totalSpans === 0 || !Number.isFinite(minTs) || !Number.isFinite(maxEnd)) {
    return { lanes: [], minTs: 0, maxEnd: 0, totalDuration: 0, totalHeight: RULER_HEIGHT, totalSpans: 0 };
  }

  const totalDuration = Math.max(maxEnd - minTs, 1);
  const lanes: SwimlaneLane[] = [];
  let cursorY = RULER_HEIGHT;

  for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
    const group = groups[groupIndex];
    if (group.spans.length === 0) continue;

    // Assign nesting/overlap depth greedily in O(N * activeDepth):
    // handles both nested parent/child scopes and crossing overlaps cleanly.
    const rowEnds: number[] = [];
    const placed: PlacedSpan[] = new Array(group.spans.length);

    for (let i = 0; i < group.spans.length; i++) {
      const span = group.spans[i];
      let depth = 0;
      while (depth < rowEnds.length && rowEnds[depth] > span.ts) {
        depth++;
      }
      const end = span.ts + span.dur;
      if (depth === rowEnds.length) rowEnds.push(end);
      else rowEnds[depth] = end;

      const y = cursorY + LANE_HEADER_HEIGHT + depth * (ROW_HEIGHT + ROW_GAP);
      placed[i] = { span, groupIndex, depth, y, height: ROW_HEIGHT };
    }

    const rowCount = Math.max(1, rowEnds.length);
    const height = LANE_HEADER_HEIGHT + rowCount * (ROW_HEIGHT + ROW_GAP) + LANE_PADDING_BOTTOM;
    const label = `PID ${group.pid} / TID ${group.tid}${group.name ? ` — ${group.name}` : ''}`;

    lanes.push({
      group,
      groupIndex,
      label,
      top: cursorY,
      headerHeight: LANE_HEADER_HEIGHT,
      height,
      rowCount,
      spans: placed,
    });
    cursorY += height;
  }

  return {
    lanes,
    minTs,
    maxEnd: minTs + totalDuration,
    totalDuration,
    totalHeight: cursorY,
    totalSpans,
  };
}

export function hitTestSwimlane(
  layout: SwimlaneLayout,
  x: number,
  y: number,
  width: number,
  viewStart: number,
  viewEnd: number,
  scrollY: number,
): PlacedSpan | null {
  if (width <= 0 || y < RULER_HEIGHT) return null;
  const contentY = y + scrollY;
  const spanDuration = viewEnd - viewStart;
  if (spanDuration <= 0) return null;
  const pxPerUs = width / spanDuration;

  for (const lane of layout.lanes) {
    if (contentY < lane.top + lane.headerHeight || contentY >= lane.top + lane.height) continue;
    // Check in reverse order so narrower/later spans on top win ties
    for (let i = lane.spans.length - 1; i >= 0; i--) {
      const item = lane.spans[i];
      if (contentY < item.y || contentY > item.y + item.height) continue;
      const sx = (item.span.ts - viewStart) * pxPerUs;
      const sw = Math.max(2, item.span.dur * pxPerUs);
      if (x >= sx - 1 && x <= sx + sw + 1) {
        return item;
      }
    }
    break;
  }
  return null;
}

function niceTickStep(rangeUs: number, targetTicks: number): number {
  const rough = rangeUs / Math.max(1, targetTicks);
  const exp = Math.floor(Math.log10(rough));
  const pow = 10 ** exp;
  const frac = rough / pow;
  if (frac <= 1.5) return 1 * pow;
  if (frac <= 3) return 2 * pow;
  if (frac <= 7) return 5 * pow;
  return 10 * pow;
}

function spanToneKey(cat: string, name: string): 'accent' | 'violet' | 'ok' | 'warn' | 'neutral' {
  const text = `${cat} ${name}`.toLowerCase();
  if (text.includes('compile') || text.includes('jit') || text.includes('mlir') || text.includes('dxil') || text.includes('optimize') || text.includes('lower')) {
    return 'accent';
  }
  if (text.includes('compute') || text.includes('dispatch')) return 'violet';
  if (text.includes('graphics') || text.includes('draw') || text.includes('mesh') || text.includes('raster')) return 'ok';
  if (text.includes('worker') || text.includes('async')) return 'warn';
  return 'neutral';
}

interface Palette {
  bg: string;
  surface: string;
  surfaceAlt: string;
  border: string;
  text: string;
  muted: string;
  accent: string;
  tones: Record<'accent' | 'violet' | 'ok' | 'warn' | 'neutral', { fill: string; border: string; text: string }>;
}

function readPalette(doc: Document): Palette {
  const win = doc.defaultView;
  const meta = doc.querySelector<HTMLMetaElement>('meta[name="color-scheme"]')?.content ?? 'light dark';
  const isDark = meta === 'dark'
    || (meta !== 'light' && Boolean(win?.matchMedia?.('(prefers-color-scheme: dark)').matches));

  if (isDark) {
    return {
      bg: '#0b111e',
      surface: '#101726',
      surfaceAlt: '#141e30',
      border: '#24324a',
      text: '#f1f5f9',
      muted: '#94a3b8',
      accent: '#38bdf8',
      tones: {
        accent: { fill: '#0284c7', border: '#38bdf8', text: '#f0f9ff' },
        violet: { fill: '#6d28d9', border: '#a78bfa', text: '#f5f3ff' },
        ok: { fill: '#047857', border: '#34d399', text: '#ecfdf5' },
        warn: { fill: '#b45309', border: '#fbbf24', text: '#fffbeb' },
        neutral: { fill: '#334155', border: '#64748b', text: '#f8fafc' },
      },
    };
  }
  return {
    bg: '#f8fafc',
    surface: '#ffffff',
    surfaceAlt: '#f1f5f9',
    border: '#cbd5e1',
    text: '#0f172a',
    muted: '#475569',
    accent: '#0284c7',
    tones: {
      accent: { fill: '#0284c7', border: '#0369a1', text: '#ffffff' },
      violet: { fill: '#7c3aed', border: '#5b21b6', text: '#ffffff' },
      ok: { fill: '#059669', border: '#047857', text: '#ffffff' },
      warn: { fill: '#d97706', border: '#b45309', text: '#ffffff' },
      neutral: { fill: '#475569', border: '#334155', text: '#ffffff' },
    },
  };
}

export class SwimlaneTimelineView implements TimelineView {
  readonly capabilities: TimelineViewCapabilities = {
    rendererName: 'Nacelle Swimlanes',
    multiThreadSwimlanes: true,
    supportsSelectionFilter: true,
  };

  private container: HTMLElement | null = null;
  private callbacks: TimelineViewCallbacks = {};
  private canvas: HTMLCanvasElement | null = null;
  private canvasWrap: HTMLElement | null = null;
  private tooltip: HTMLElement | null = null;
  private messageEl: HTMLElement | null = null;
  private infoEl: HTMLElement | null = null;
  private resetFilterBtn: HTMLButtonElement | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private themeObserver: MutationObserver | null = null;
  private mediaQuery: MediaQueryList | null = null;
  private mediaListener: (() => void) | null = null;

  private layout: SwimlaneLayout = computeSwimlaneLayout([]);
  private isFiltered = false;
  private viewStart = 0;
  private viewEnd = 1;
  private scrollY = 0;
  private hovered: PlacedSpan | null = null;
  private selected: PlacedSpan | null = null;
  private dragState: { startX: number; startY: number; origStart: number; origEnd: number; origScrollY: number; moved: boolean } | null = null;
  private rafId: number | null = null;

  mount(container: HTMLElement, callbacks: TimelineViewCallbacks = {}): void {
    this.dispose();
    this.container = container;
    this.callbacks = callbacks;
    container.replaceChildren();

    const doc = container.ownerDocument;
    const root = doc.createElement('div');
    root.className = 'flex h-full flex-col bg-[var(--surface)]';

    // Compact controls toolbar
    const toolbar = doc.createElement('div');
    toolbar.className = 'flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] bg-[var(--surface-elevated)]/60 px-3 py-1.5 text-xs';

    const leftControls = doc.createElement('div');
    leftControls.className = 'flex flex-wrap items-center gap-1.5';

    const mkBtn = (label: string, title: string, onClick: () => void) => {
      const btn = doc.createElement('button');
      btn.type = 'button';
      btn.textContent = label;
      btn.title = title;
      btn.className = 'profiler-btn rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5 text-[11px] font-medium text-[var(--text)] transition hover:border-[var(--accent)] hover:text-[var(--accent)]';
      btn.addEventListener('click', onClick);
      return btn;
    };

    const zoomInBtn = mkBtn('+', 'Zoom in (scroll wheel)', () => this.zoomBy(0.6));
    zoomInBtn.setAttribute('aria-label', 'Zoom in timeline');
    const zoomOutBtn = mkBtn('−', 'Zoom out (scroll wheel)', () => this.zoomBy(1.65));
    zoomOutBtn.setAttribute('aria-label', 'Zoom out timeline');
    const resetZoomBtn = mkBtn('Reset zoom', 'Fit entire capture window', () => this.resetViewport());
    const filterWindowBtn = mkBtn('Filter to window', 'Filter span tables to the visible time window', () => {
      if (this.layout.totalSpans === 0) return;
      this.callbacks.onSelectFilter?.({
        start: Math.round(this.viewStart * 100) / 100,
        end: Math.round(this.viewEnd * 100) / 100,
      });
    });

    const resetFilterBtn = mkBtn('Show all threads', 'Clear active table/timeline filters', () => {
      this.selected = null;
      this.callbacks.onSelectFilter?.({ reset: true });
    });
    resetFilterBtn.hidden = true;
    this.resetFilterBtn = resetFilterBtn;

    leftControls.append(zoomInBtn, zoomOutBtn, resetZoomBtn, filterWindowBtn, resetFilterBtn);

    const info = doc.createElement('div');
    info.className = 'font-mono-tabular text-[11px] text-[var(--muted)]';
    info.textContent = 'Drag to pan · Scroll to zoom · Click span to filter table · Double-click to zoom span';
    this.infoEl = info;

    toolbar.append(leftControls, info);

    // Canvas viewport wrapper
    const wrap = doc.createElement('div');
    wrap.className = 'relative flex-1 overflow-hidden select-none';
    this.canvasWrap = wrap;

    const canvas = doc.createElement('canvas');
    canvas.className = 'block h-full w-full cursor-grab active:cursor-grabbing';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', 'Interactive multi-thread swimlane timeline');
    this.canvas = canvas;

    const message = doc.createElement('div');
    message.className = 'absolute inset-0 flex items-center justify-center p-6 text-center text-xs text-[var(--muted)]';
    message.hidden = true;
    this.messageEl = message;

    const tooltip = doc.createElement('div');
    tooltip.className = 'pointer-events-none absolute z-20 max-w-xs rounded-lg border border-[var(--border-strong)] bg-[var(--surface)]/95 px-3 py-2 text-xs shadow-lg backdrop-blur-sm';
    tooltip.hidden = true;
    this.tooltip = tooltip;

    wrap.append(canvas, message, tooltip);
    root.append(toolbar, wrap);
    container.append(root);

    this.attachEvents(canvas, doc);
  }

  render(input: TimelineRenderInput): void {
    if (!this.container || !this.canvas || !this.messageEl) return;
    this.isFiltered = input.filtered;
    if (this.resetFilterBtn) this.resetFilterBtn.hidden = !input.filtered;

    this.layout = computeSwimlaneLayout(input.groups);
    this.hovered = null;
    this.selected = null;
    this.hideTooltip();

    if (this.layout.totalSpans === 0) {
      this.canvas.hidden = true;
      this.messageEl.textContent = 'No timing spans match the current filters.';
      this.messageEl.hidden = false;
      this.updateInfo();
      return;
    }

    this.canvas.hidden = false;
    this.messageEl.hidden = true;
    this.viewStart = this.layout.minTs;
    this.viewEnd = this.layout.maxEnd;
    this.scrollY = 0;
    this.updateInfo();
    this.scheduleDraw();
  }

  clear(): void {
    this.layout = computeSwimlaneLayout([]);
    this.isFiltered = false;
    this.hovered = null;
    this.selected = null;
    this.hideTooltip();
    if (this.resetFilterBtn) this.resetFilterBtn.hidden = true;
    if (this.canvas) this.canvas.hidden = true;
    if (this.messageEl) {
      this.messageEl.hidden = true;
      this.messageEl.textContent = '';
    }
  }

  dispose(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.themeObserver?.disconnect();
    this.themeObserver = null;
    if (this.mediaQuery && this.mediaListener) {
      this.mediaQuery.removeEventListener('change', this.mediaListener);
      this.mediaQuery = null;
      this.mediaListener = null;
    }
    this.clear();
    if (this.container) this.container.replaceChildren();
    this.container = null;
    this.canvas = null;
    this.canvasWrap = null;
    this.tooltip = null;
    this.messageEl = null;
    this.infoEl = null;
    this.resetFilterBtn = null;
  }

  private attachEvents(canvas: HTMLCanvasElement, doc: Document): void {
    const win = doc.defaultView;

    canvas.addEventListener('wheel', event => {
      if (this.layout.totalSpans === 0) return;
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, rect.width);
      const x = Math.max(0, Math.min(width, event.clientX - rect.left));

      // Horizontal pan with Shift+wheel or horizontal trackpad swipe
      if (event.shiftKey || (Math.abs(event.deltaX) > Math.abs(event.deltaY) && Math.abs(event.deltaX) > 0)) {
        const deltaPx = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
        const range = this.viewEnd - this.viewStart;
        this.panBy((deltaPx / width) * range);
        return;
      }

      // Zoom around pointer timestamp
      const factor = event.deltaY > 0 ? 1.25 : 0.8;
      this.zoomAround(x / width, factor);
    }, { passive: false });

    canvas.addEventListener('pointerdown', event => {
      if (event.button !== 0 || this.layout.totalSpans === 0) return;
      canvas.setPointerCapture?.(event.pointerId);
      this.dragState = {
        startX: event.clientX,
        startY: event.clientY,
        origStart: this.viewStart,
        origEnd: this.viewEnd,
        origScrollY: this.scrollY,
        moved: false,
      };
    });

    canvas.addEventListener('pointermove', event => {
      if (this.layout.totalSpans === 0) return;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, rect.width);
      const height = Math.max(1, rect.height);

      if (this.dragState) {
        const dx = event.clientX - this.dragState.startX;
        const dy = event.clientY - this.dragState.startY;
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) this.dragState.moved = true;
        if (this.dragState.moved) {
          const origRange = this.dragState.origEnd - this.dragState.origStart;
          const dt = -(dx / width) * origRange;
          this.setViewWindow(this.dragState.origStart + dt, this.dragState.origEnd + dt);
          const maxScroll = Math.max(0, this.layout.totalHeight - height);
          this.scrollY = Math.max(0, Math.min(maxScroll, this.dragState.origScrollY - dy));
          this.hideTooltip();
          this.updateInfo();
          this.scheduleDraw();
          return;
        }
      }

      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const hit = hitTestSwimlane(this.layout, x, y, width, this.viewStart, this.viewEnd, this.scrollY);
      if (hit !== this.hovered) {
        this.hovered = hit;
        canvas.style.cursor = hit ? 'pointer' : 'grab';
        this.scheduleDraw();
      }
      if (hit) this.showTooltip(hit, x, y, width, height);
      else this.hideTooltip();
    });

    canvas.addEventListener('pointerup', event => {
      if (!this.dragState) return;
      const wasDrag = this.dragState.moved;
      this.dragState = null;
      canvas.releasePointerCapture?.(event.pointerId);
      if (wasDrag) return;

      const rect = canvas.getBoundingClientRect();
      const hit = hitTestSwimlane(
        this.layout,
        event.clientX - rect.left,
        event.clientY - rect.top,
        Math.max(1, rect.width),
        this.viewStart,
        this.viewEnd,
        this.scrollY,
      );
      if (hit) {
        this.selected = hit;
        const group = this.layout.lanes.find(l => l.groupIndex === hit.groupIndex)?.group;
        this.scheduleDraw();
        this.callbacks.onSelectFilter?.({
          thread: group ? `${group.pid}:${group.tid}` : undefined,
          start: hit.span.ts,
          end: hit.span.ts + hit.span.dur,
        });
      }
    });

    canvas.addEventListener('pointerleave', () => {
      if (!this.dragState) {
        this.hovered = null;
        this.hideTooltip();
        this.scheduleDraw();
      }
    });

    canvas.addEventListener('dblclick', event => {
      if (this.layout.totalSpans === 0) return;
      const rect = canvas.getBoundingClientRect();
      const hit = hitTestSwimlane(
        this.layout,
        event.clientX - rect.left,
        event.clientY - rect.top,
        Math.max(1, rect.width),
        this.viewStart,
        this.viewEnd,
        this.scrollY,
      );
      if (!hit) {
        this.resetViewport();
        return;
      }
      const dur = Math.max(hit.span.dur, 1);
      const pad = dur * 0.15;
      this.setViewWindow(hit.span.ts - pad, hit.span.ts + dur + pad);
      this.updateInfo();
      this.scheduleDraw();
    });

    if (typeof ResizeObserver !== 'undefined' && this.canvasWrap) {
      this.resizeObserver = new ResizeObserver(() => this.scheduleDraw());
      this.resizeObserver.observe(this.canvasWrap);
    }

    const meta = doc.querySelector<HTMLMetaElement>('meta[name="color-scheme"]');
    if (meta && typeof MutationObserver !== 'undefined') {
      this.themeObserver = new MutationObserver(() => this.scheduleDraw());
      this.themeObserver.observe(meta, { attributes: true, attributeFilter: ['content'] });
    }
    if (win?.matchMedia) {
      this.mediaQuery = win.matchMedia('(prefers-color-scheme: dark)');
      this.mediaListener = () => this.scheduleDraw();
      this.mediaQuery.addEventListener('change', this.mediaListener);
    }
  }

  private resetViewport(): void {
    if (this.layout.totalSpans === 0) return;
    this.viewStart = this.layout.minTs;
    this.viewEnd = this.layout.maxEnd;
    this.scrollY = 0;
    this.updateInfo();
    this.scheduleDraw();
  }

  private zoomBy(factor: number): void {
    this.zoomAround(0.5, factor);
  }

  private zoomAround(anchorRatio: number, factor: number): void {
    if (this.layout.totalSpans === 0) return;
    const range = this.viewEnd - this.viewStart;
    const minRange = Math.max(0.1, this.layout.totalDuration / 100_000);
    const nextRange = Math.max(minRange, Math.min(this.layout.totalDuration, range * factor));
    const anchorTs = this.viewStart + range * anchorRatio;
    this.setViewWindow(anchorTs - nextRange * anchorRatio, anchorTs + nextRange * (1 - anchorRatio));
    this.updateInfo();
    this.scheduleDraw();
  }

  private panBy(deltaUs: number): void {
    this.setViewWindow(this.viewStart + deltaUs, this.viewEnd + deltaUs);
    this.updateInfo();
    this.scheduleDraw();
  }

  private setViewWindow(start: number, end: number): void {
    const total = this.layout.totalDuration;
    const range = Math.min(total, Math.max(0.1, end - start));
    let nextStart = start;
    if (nextStart < this.layout.minTs) nextStart = this.layout.minTs;
    if (nextStart + range > this.layout.maxEnd) nextStart = this.layout.maxEnd - range;
    this.viewStart = nextStart;
    this.viewEnd = nextStart + range;
  }

  private updateInfo(): void {
    if (!this.infoEl) return;
    if (this.layout.totalSpans === 0) {
      this.infoEl.textContent = 'No spans';
      return;
    }
    const range = this.viewEnd - this.viewStart;
    const zoom = this.layout.totalDuration / Math.max(0.001, range);
    const relStart = this.viewStart - this.layout.minTs;
    const relEnd = this.viewEnd - this.layout.minTs;
    const laneCount = this.layout.lanes.length;
    this.infoEl.textContent = `${formatTimeUs(relStart)} – ${formatTimeUs(relEnd)} (${zoom.toFixed(1)}×) · ${laneCount} ${laneCount === 1 ? 'thread' : 'threads'} · ${this.layout.totalSpans.toLocaleString()} spans`;
  }

  private showTooltip(hit: PlacedSpan, x: number, y: number, width: number, height: number): void {
    if (!this.tooltip) return;
    const doc = this.tooltip.ownerDocument;
    const lane = this.layout.lanes.find(l => l.groupIndex === hit.groupIndex);
    const { span } = hit;

    this.tooltip.replaceChildren();
    const title = doc.createElement('div');
    title.className = 'font-semibold text-[var(--text)] break-words';
    title.textContent = span.name;
    this.tooltip.append(title);

    const meta = doc.createElement('div');
    meta.className = 'font-mono-tabular mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-[var(--muted)]';
    const durPct = this.layout.totalDuration > 0 ? ((span.dur / this.layout.totalDuration) * 100).toFixed(1) : '0.0';
    const items = [
      `Dur: ${formatTimeUs(span.dur)} (${durPct}%)`,
      `Start: +${formatTimeUs(span.ts - this.layout.minTs)}`,
      span.cat ? `Cat: ${span.cat}` : '',
      lane ? lane.label : '',
    ].filter(Boolean);
    for (const line of items) {
      const el = doc.createElement('span');
      el.textContent = line;
      meta.append(el);
    }
    this.tooltip.append(meta);

    const argKeys = ['stage', 'pso', 'simdWidth', 'tier', 'cache'].filter(k => span.args[k] !== undefined);
    if (argKeys.length > 0) {
      const argsEl = doc.createElement('div');
      argsEl.className = 'font-mono-tabular mt-1 border-t border-[var(--border)] pt-1 text-[10px] text-[var(--muted)] break-all';
      argsEl.textContent = argKeys.map(k => `${k}=${String(span.args[k])}`).join(' · ');
      this.tooltip.append(argsEl);
    }

    const hint = doc.createElement('div');
    hint.className = 'mt-1 text-[10px] text-[var(--accent)]';
    hint.textContent = 'Click to filter table · Double-click to zoom';
    this.tooltip.append(hint);

    this.tooltip.hidden = false;
    const tipX = x + 14 + 260 > width ? Math.max(8, x - 268) : x + 14;
    const tipY = y + 14 + 110 > height ? Math.max(8, y - 110) : y + 14;
    this.tooltip.style.left = `${tipX}px`;
    this.tooltip.style.top = `${tipY}px`;
  }

  private hideTooltip(): void {
    if (this.tooltip) this.tooltip.hidden = true;
  }

  private scheduleDraw(): void {
    if (this.rafId !== null) return;
    const schedule = typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame
      : (cb: FrameRequestCallback) => window.setTimeout(() => cb(0), 0);
    this.rafId = schedule(() => {
      this.rafId = null;
      this.draw();
    });
  }

  private draw(): void {
    if (!this.canvas || !this.canvasWrap || this.layout.totalSpans === 0) return;
    const rect = this.canvasWrap.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    const dpr = this.canvas.ownerDocument.defaultView?.devicePixelRatio || 1;

    if (this.canvas.width !== Math.floor(width * dpr) || this.canvas.height !== Math.floor(height * dpr)) {
      this.canvas.width = Math.floor(width * dpr);
      this.canvas.height = Math.floor(height * dpr);
    }

    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    ctx.save();
    ctx.scale(dpr, dpr);

    const palette = readPalette(this.canvas.ownerDocument);
    const range = Math.max(0.001, this.viewEnd - this.viewStart);
    const pxPerUs = width / range;

    // Background
    ctx.fillStyle = palette.bg;
    ctx.fillRect(0, 0, width, height);

    // Compute ruler ticks
    const tickStep = niceTickStep(range, Math.max(4, Math.floor(width / 110)));
    const firstTickRel = Math.ceil((this.viewStart - this.layout.minTs) / tickStep) * tickStep;

    // Vertical gridlines behind lanes
    ctx.strokeStyle = palette.border;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    for (let rel = firstTickRel; this.layout.minTs + rel <= this.viewEnd; rel += tickStep) {
      const gx = Math.round((this.layout.minTs + rel - this.viewStart) * pxPerUs) + 0.5;
      if (gx < 0 || gx > width) continue;
      ctx.moveTo(gx, RULER_HEIGHT);
      ctx.lineTo(gx, height);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Clip and render thread swimlanes
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, RULER_HEIGHT, width, Math.max(0, height - RULER_HEIGHT));
    ctx.clip();

    for (let laneIdx = 0; laneIdx < this.layout.lanes.length; laneIdx++) {
      const lane = this.layout.lanes[laneIdx];
      const laneScreenTop = lane.top - this.scrollY;
      if (laneScreenTop + lane.height < RULER_HEIGHT || laneScreenTop > height) continue;

      // Lane background stripe
      ctx.fillStyle = laneIdx % 2 === 0 ? palette.surface : palette.surfaceAlt;
      ctx.fillRect(0, laneScreenTop, width, lane.height);

      // Lane header strip
      ctx.fillStyle = palette.surfaceAlt;
      ctx.fillRect(0, laneScreenTop, width, lane.headerHeight);
      ctx.strokeStyle = palette.border;
      ctx.beginPath();
      ctx.moveTo(0, laneScreenTop + 0.5);
      ctx.lineTo(width, laneScreenTop + 0.5);
      ctx.moveTo(0, laneScreenTop + lane.headerHeight + 0.5);
      ctx.lineTo(width, laneScreenTop + lane.headerHeight + 0.5);
      ctx.stroke();

      ctx.font = '600 11px ui-sans-serif, system-ui, -apple-system, sans-serif';
      ctx.fillStyle = palette.text;
      ctx.textBaseline = 'middle';
      ctx.fillText(lane.label, 10, laneScreenTop + lane.headerHeight / 2);

      const countLabel = `${lane.spans.length.toLocaleString()} spans`;
      ctx.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = palette.muted;
      const countWidth = ctx.measureText(countLabel).width;
      ctx.fillText(countLabel, width - countWidth - 10, laneScreenTop + lane.headerHeight / 2);

      // Draw spans in lane with viewport culling
      ctx.font = '500 11px ui-monospace, SFMono-Regular, Menlo, monospace';
      for (let i = 0; i < lane.spans.length; i++) {
        const item = lane.spans[i];
        const sEnd = item.span.ts + item.span.dur;
        if (sEnd < this.viewStart) continue;
        if (item.span.ts > this.viewEnd) break; // spans are sorted by ts

        const sy = item.y - this.scrollY;
        if (sy + item.height < RULER_HEIGHT || sy > height) continue;

        const rawX = (item.span.ts - this.viewStart) * pxPerUs;
        const rawW = item.span.dur * pxPerUs;
        const sx = Math.max(0, rawX);
        const ex = Math.min(width, rawX + Math.max(1.5, rawW));
        const sw = Math.max(1.5, ex - sx);

        const tone = palette.tones[spanToneKey(item.span.cat, item.span.name)];
        const isHot = item === this.hovered || item === this.selected;

        ctx.fillStyle = tone.fill;
        ctx.fillRect(sx, sy, sw, item.height);

        if (sw >= 3 || isHot) {
          ctx.strokeStyle = isHot ? palette.text : tone.border;
          ctx.lineWidth = isHot ? 1.5 : 1;
          ctx.strokeRect(sx + 0.5, sy + 0.5, Math.max(1, sw - 1), item.height - 1);
          ctx.lineWidth = 1;
        }

        if (sw >= 36) {
          const text = sw >= 120
            ? `${item.span.name} (${formatTimeUs(item.span.dur)})`
            : item.span.name;
          ctx.save();
          ctx.beginPath();
          ctx.rect(sx + 4, sy, sw - 8, item.height);
          ctx.clip();
          ctx.fillStyle = tone.text;
          ctx.textBaseline = 'middle';
          ctx.fillText(text, sx + 5, sy + item.height / 2 + 0.5);
          ctx.restore();
        }
      }
    }
    ctx.restore();

    // Sticky top time ruler
    ctx.fillStyle = palette.surface;
    ctx.fillRect(0, 0, width, RULER_HEIGHT);
    ctx.strokeStyle = palette.border;
    ctx.beginPath();
    ctx.moveTo(0, RULER_HEIGHT - 0.5);
    ctx.lineTo(width, RULER_HEIGHT - 0.5);
    ctx.stroke();

    ctx.font = '500 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = palette.muted;
    ctx.textBaseline = 'middle';
    for (let rel = firstTickRel; this.layout.minTs + rel <= this.viewEnd; rel += tickStep) {
      const gx = Math.round((this.layout.minTs + rel - this.viewStart) * pxPerUs) + 0.5;
      if (gx < 0 || gx > width) continue;
      ctx.strokeStyle = palette.border;
      ctx.beginPath();
      ctx.moveTo(gx, RULER_HEIGHT - 7);
      ctx.lineTo(gx, RULER_HEIGHT);
      ctx.stroke();
      if (gx + 54 < width) {
        ctx.fillText(`+${formatTimeUs(rel)}`, gx + 4, RULER_HEIGHT / 2 - 1);
      }
    }

    ctx.restore();
  }
}

export function createSwimlaneTimelineView(): TimelineView {
  return new SwimlaneTimelineView();
}
