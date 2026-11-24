// @vitest-environment jsdom
import html from '../index.html?raw';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { FileProfileDataSource } from '../src/file-source';
import { loadProfile, type Profile, type ProfileDataSource } from '../src/profile';
import { SpeedscopeTimelineView } from '../src/speedscope-timeline';
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
});
