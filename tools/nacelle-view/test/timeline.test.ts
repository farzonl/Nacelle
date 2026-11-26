// @vitest-environment jsdom
import html from '../index.html?raw';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { FileProfileDataSource } from '../src/file-source';
import { loadProfile, type Profile, type ProfileDataSource } from '../src/profile';
import { SpeedscopeTimelineView } from '../src/speedscope-timeline';
import { computeSwimlaneLayout, createSwimlaneTimelineView, hitTestSwimlane } from '../src/swimlane-timeline';
import type { TimelineRenderInput, TimelineView, TimelineViewCallbacks } from '../src/timeline';
import { mountViewer } from '../src/ui';
import { fixture, span, trace } from './fixtures';

function source(profile: Profile): ProfileDataSource {
  return {
    readManifest: async () => profile.manifest,
    readEntry: async path => path === 'trace.json.zst' ? profile.traceBytes : new TextEncoder().encode('[]').buffer,
    close: vi.fn(),
  };
}

async function makeProfile(events = [span()]): Promise<Profile> {
  return loadProfile(FileProfileDataSource.fromBuffer(fixture({}, trace(events))));
}

function chooseFile(): void {
  const input = document.querySelector<HTMLInputElement>('#profile-file')!;
  Object.defineProperty(input, 'files', { configurable: true, value: [new File([], 'capture.nacelle-profile')] });
  input.dispatchEvent(new Event('change'));
}

describe('SpeedscopeTimelineView', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="host"></div>';
  });

  it('loads a blob URL into a same-origin Speedscope iframe and revokes old blobs', async () => {
    const create = vi.fn()
      .mockReturnValueOnce('blob:profile-1')
      .mockReturnValueOnce('blob:profile-2');
    const revoke = vi.fn();
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: create });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke });

    const host = document.querySelector<HTMLElement>('#host')!;
    const view = new SpeedscopeTimelineView();
    view.mount(host);

    const loaded = await makeProfile([span({ name: 'Compute dispatch', ts: 10, dur: 25 })]);
    const group = { pid: 1, tid: 2, name: 'main', spans: loaded.trace.traceEvents.filter(e => e.ph === 'X') };
    view.render({ profile: loaded, groups: [group], filtered: false, title: 'Raymarcher' });

    const iframe = host.querySelector('iframe')!;
    expect(iframe.hidden).toBe(false);
    expect(iframe.src).toContain('speedscope/index.html#profileURL=blob%3Aprofile-1&title=Raymarcher');

    view.render({ profile: loaded, groups: [group], filtered: true, title: 'Raymarcher' });
    expect(revoke).toHaveBeenCalledWith('blob:profile-1');
    expect(iframe.src).toContain('blob%3Aprofile-2');

    view.dispose();
    expect(revoke).toHaveBeenCalledWith('blob:profile-2');
  });

  it('shows a graceful message instead of throwing when spans have crossing overlaps', async () => {
    const host = document.querySelector<HTMLElement>('#host')!;
    const view = new SpeedscopeTimelineView();
    view.mount(host);

    const loaded = await makeProfile([span({ ts: 0, dur: 10 }), span({ ts: 5, dur: 10 })]);
    const group = { pid: 1, tid: 2, name: 'main', spans: loaded.trace.traceEvents.filter(e => e.ph === 'X') };
    view.render({ profile: loaded, groups: [group], filtered: false, title: 'Crossing' });

    expect(host.querySelector('iframe')!.hidden).toBe(true);
    expect(host.textContent).toContain('crossing overlaps');
  });

  it('imports into a nonce-controlled embedded viewer and clears its source on invalid selection', async () => {
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:embedded') });
    const revoke = vi.fn();
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke });
    const host = document.querySelector<HTMLElement>('#host')!;
    const view = new SpeedscopeTimelineView({ frameHtml: '<html><head></head><body></body></html>', frameNonce: 'test-nonce' });
    view.mount(host);
    const loaded = await makeProfile();
    const groups = [{ pid: 1, tid: 2, name: 'thread', spans: loaded.trace.traceEvents.filter(e => e.ph === 'X') }];
    view.render({ profile: loaded, groups, filtered: false, title: '<capture>' });
    const iframe = host.querySelector('iframe')!;
    expect(iframe.srcdoc).toContain('nonce="test-nonce"');
    expect(iframe.srcdoc).not.toContain('profileURL=');
    expect(iframe.onload).toBeTypeOf('function');
    view.render({ profile: loaded, groups: [], filtered: true, title: 'empty' });
    expect(iframe.hasAttribute('srcdoc')).toBe(false);
    expect(iframe.onload).toBeNull();
    expect(revoke).toHaveBeenCalledWith('blob:embedded');
    view.dispose();
  });
});

describe('TimelineView integration in mountViewer', () => {
  beforeEach(() => {
    document.documentElement.innerHTML = html;
  });

  it('drives a pluggable TimelineView and applies selection filters back to the table', async () => {
    const loaded = await makeProfile([
      span({ pid: 1, tid: 2, ts: 10, dur: 20, args: { pso: 'aaaa', stage: 'compute' } }),
      span({ pid: 2, tid: 3, ts: 50, dur: 30, args: { pso: 'bbbb', stage: 'pixel' } }),
    ]);

    let capturedCallbacks: TimelineViewCallbacks | undefined;
    const renders: TimelineRenderInput[] = [];
    const clear = vi.fn();
    const customTimeline: TimelineView = {
      capabilities: {
        rendererName: 'Custom Swimlanes',
        multiThreadSwimlanes: true,
        supportsSelectionFilter: true,
      },
      mount: (_container, callbacks) => { capturedCallbacks = callbacks; },
      render: input => { renders.push(input); },
      clear,
      dispose: vi.fn(),
    };

    mountViewer(document, async () => source(loaded), { timelineView: customTimeline });
    chooseFile();
    await vi.waitFor(() => expect(renders.length).toBe(1));

    expect(document.querySelector<HTMLElement>('#timeline-section')!.hidden).toBe(false);
    expect(document.querySelector('#timeline-renderer')!.textContent).toBe('Custom Swimlanes');
    expect(document.querySelectorAll('tbody tr')).toHaveLength(2);

    // Simulate a future custom swimlane renderer clicking a span to filter the table
    capturedCallbacks?.onSelectFilter?.({ thread: '2:3', stage: 'pixel' });
    expect(document.querySelectorAll('tbody tr')).toHaveLength(1);
    expect(renders.length).toBe(2);
    expect(renders[1].filtered).toBe(true);
  });

  it('lays out simultaneous multi-thread swimlanes including crossing overlaps and hit-tests spans', async () => {
    const loaded = await makeProfile([
      span({ pid: 1, tid: 1, name: 'Outer', ts: 100, dur: 100 }),
      span({ pid: 1, tid: 1, name: 'Inner', ts: 120, dur: 40 }),
      span({ pid: 1, tid: 1, name: 'Crossing', ts: 150, dur: 80 }),
      span({ pid: 1, tid: 2, name: 'Worker', ts: 110, dur: 90 }),
    ]);
    const g1 = { pid: 1, tid: 1, name: 'main', spans: loaded.trace.traceEvents.filter(e => e.ph === 'X' && e.tid === 1) };
    const g2 = { pid: 1, tid: 2, name: 'worker-0', spans: loaded.trace.traceEvents.filter(e => e.ph === 'X' && e.tid === 2) };

    const layout = computeSwimlaneLayout([g1, g2]);
    expect(layout.lanes).toHaveLength(2);
    expect(layout.minTs).toBe(100);
    expect(layout.maxEnd).toBe(230);
    // Outer=depth 0, Inner=depth 1, Crossing overlaps both Outer and Inner => depth 2
    expect(layout.lanes[0].spans.map(s => s.depth)).toEqual([0, 1, 2]);
    expect(layout.lanes[0].rowCount).toBe(3);
    expect(layout.lanes[1].rowCount).toBe(1);

    const workerItem = layout.lanes[1].spans[0];
    const hit = hitTestSwimlane(layout, 200, workerItem.y + 4, 650, layout.minTs, layout.maxEnd, 0);
    expect(hit?.span.name).toBe('Worker');
  });

  it('mounts SwimlaneTimelineView in viewer, filters on span click, and resets via toolbar button', async () => {
    const loaded = await makeProfile([
      span({ pid: 1, tid: 1, name: 'Compile', ts: 10, dur: 40, args: { stage: 'compute' } }),
      span({ pid: 2, tid: 2, name: 'Raster', ts: 60, dur: 50, args: { stage: 'pixel' } }),
    ]);

    mountViewer(document, async () => source(loaded), { timelineView: createSwimlaneTimelineView() });
    chooseFile();
    await vi.waitFor(() => expect(document.querySelectorAll('tbody tr')).toHaveLength(2));

    expect(document.querySelector('#timeline-renderer')!.textContent).toBe('Nacelle Swimlanes');
    const canvas = document.querySelector<HTMLCanvasElement>('#timeline-host canvas')!;
    vi.spyOn(canvas, 'getContext').mockReturnValue(null);
    expect(canvas.hidden).toBe(false);

    // Mock bounding rect for deterministic click hit-testing on the first lane's span
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, right: 500, bottom: 400, width: 500, height: 400, x: 0, y: 0, toJSON: () => ({}),
    });
    // Ruler=28, lane0 header=24 => first span y=[52..72]; ts=10..50 in [10..110] => x=[0..200]
    canvas.dispatchEvent(new MouseEvent('pointerdown', { button: 0, clientX: 80, clientY: 60 }));
    canvas.dispatchEvent(new MouseEvent('pointerup', { button: 0, clientX: 80, clientY: 60 }));

    expect(document.querySelectorAll('tbody tr')).toHaveLength(1);
    expect(document.querySelector('#threads')!.textContent).toContain('Compile');
    expect(document.querySelector('#threads')!.textContent).not.toContain('Raster');

    // Click "Show all threads" button in swimlane toolbar to clear filters
    const resetBtn = Array.from(document.querySelectorAll<HTMLButtonElement>('#timeline-host button'))
      .find(b => b.textContent === 'Show all threads')!;
    expect(resetBtn.hidden).toBe(false);
    resetBtn.click();
    expect(document.querySelectorAll('tbody tr')).toHaveLength(2);
  });
});
