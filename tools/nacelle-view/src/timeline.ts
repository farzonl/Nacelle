import type { ThreadGroup } from './presentation';
import type { Profile } from './profile';

export interface TimelineRenderInput {
  profile: Profile;
  groups: ThreadGroup[];
  filtered: boolean;
  title: string;
}

export interface TimelineSelectionEvent {
  thread?: string;
  pso?: string;
  stage?: string;
  start?: number;
  end?: number;
}

export interface TimelineViewCapabilities {
  /** Human-readable name displayed in the viewer chrome (e.g. "Speedscope 1.25.0"). */
  readonly rendererName: string;
  /** Whether the renderer displays all threads in simultaneous swimlanes. */
  readonly multiThreadSwimlanes: boolean;
  /** Whether clicking spans in the timeline can update table filters. */
  readonly supportsSelectionFilter: boolean;
}

export interface TimelineViewCallbacks {
  /** Optional callback for interactive renderers to filter the span table from a timeline selection. */
  onSelectFilter?: (selection: TimelineSelectionEvent) => void;
}

/**
 * Renderer-agnostic contract for interactive timeline/flamegraph surfaces.
 * Implementations (e.g., embedded Speedscope iframe or a future custom multi-thread
 * swimlane canvas) can be swapped without modifying viewer state or filter orchestration.
 */
export interface TimelineView {
  readonly capabilities: TimelineViewCapabilities;
  mount(container: HTMLElement, callbacks?: TimelineViewCallbacks): void;
  render(input: TimelineRenderInput): void;
  clear(): void;
  dispose(): void;
}
