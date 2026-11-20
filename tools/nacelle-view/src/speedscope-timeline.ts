import { LIMITS, requireValid } from './profile';
import { speedscopeProfile } from './speedscope';
import type {
  TimelineRenderInput,
  TimelineView,
  TimelineViewCallbacks,
  TimelineViewCapabilities,
} from './timeline';

export interface SpeedscopeTimelineOptions {
  /** Relative URL to the locally bundled Speedscope entry page. */
  baseUrl?: string;
}

export class SpeedscopeTimelineView implements TimelineView {
  readonly capabilities: TimelineViewCapabilities = {
    rendererName: 'Speedscope 1.25.0',
    multiThreadSwimlanes: false,
    supportsSelectionFilter: false,
  };

  private readonly baseUrl: string;
  private container: HTMLElement | null = null;
  private iframe: HTMLIFrameElement | null = null;
  private messageEl: HTMLElement | null = null;
  private activeBlobUrl: string | null = null;

  constructor(options: SpeedscopeTimelineOptions = {}) {
    this.baseUrl = options.baseUrl ?? './speedscope/index.html';
  }

  mount(container: HTMLElement, _callbacks?: TimelineViewCallbacks): void {
    this.dispose();
    this.container = container;
    container.replaceChildren();

    const doc = container.ownerDocument;
    const message = doc.createElement('div');
    message.className = 'flex h-full items-center justify-center p-6 text-center text-xs text-[var(--muted)]';
    message.hidden = true;
    this.messageEl = message;

    const iframe = doc.createElement('iframe');
    iframe.title = 'Interactive Speedscope flamegraph viewer';
    iframe.className = 'h-full w-full border-0 bg-[var(--surface)]';
    iframe.hidden = true;
    this.iframe = iframe;

    container.append(message, iframe);
  }

  render(input: TimelineRenderInput): void {
    if (!this.container || !this.iframe || !this.messageEl) return;
    this.revokeBlob();

    const spanCount = input.groups.reduce((sum, g) => sum + g.spans.length, 0);
    if (spanCount === 0) {
      this.showMessage('No timing spans match the current filters.');
      return;
    }

    try {
      const payload = JSON.stringify(speedscopeProfile(input.profile, input.groups, input.filtered));
      const blob = new Blob([payload], { type: 'application/json' });
      requireValid(blob.size <= LIMITS.entry, 'Flamegraph payload exceeds the 256 MiB size limit.');
      const blobUrl = URL.createObjectURL(blob);
      this.activeBlobUrl = blobUrl;

      const hash = `profileURL=${encodeURIComponent(blobUrl)}&title=${encodeURIComponent(input.title)}`;
      this.messageEl.hidden = true;
      this.iframe.hidden = false;
      // Assign via src with unique blob URL so Speedscope reloads cleanly on every profile/filter update.
      this.iframe.src = `${this.baseUrl}#${hash}`;
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'Unable to render flamegraph.';
      this.showMessage(`Flamegraph unavailable for this selection: ${reason} The span tables and Chrome trace export remain available below.`);
    }
  }

  clear(): void {
    this.revokeBlob();
    if (this.iframe) {
      this.iframe.hidden = true;
      this.iframe.removeAttribute('src');
    }
    if (this.messageEl) {
      this.messageEl.hidden = true;
      this.messageEl.textContent = '';
    }
  }

  dispose(): void {
    this.clear();
    if (this.container) this.container.replaceChildren();
    this.container = null;
    this.iframe = null;
    this.messageEl = null;
  }

  private showMessage(text: string): void {
    if (this.iframe) {
      this.iframe.hidden = true;
      this.iframe.removeAttribute('src');
    }
    if (this.messageEl) {
      this.messageEl.textContent = text;
      this.messageEl.hidden = false;
    }
  }

  private revokeBlob(): void {
    if (this.activeBlobUrl) {
      URL.revokeObjectURL(this.activeBlobUrl);
      this.activeBlobUrl = null;
    }
  }
}

export function createSpeedscopeTimelineView(options?: SpeedscopeTimelineOptions): TimelineView {
  return new SpeedscopeTimelineView(options);
}
