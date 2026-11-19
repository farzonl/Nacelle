// @vitest-environment jsdom
import html from '../index.html?raw';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mountViewer } from '../src/ui';
import { loadProfile, type Profile, type ProfileDataSource } from '../src/profile';
import { FileProfileDataSource } from '../src/file-source';
import { fixture, span, trace } from './fixtures';

function source(profile: Profile): ProfileDataSource {
  return {
    readManifest: async () => profile.manifest,
    readEntry: async path => path === 'trace.json.zst' ? profile.traceBytes : new TextEncoder().encode('[]').buffer,
    close: vi.fn(),
  };
}

function chooseFile(): void {
  const input = document.querySelector<HTMLInputElement>('#profile-file')!;
  Object.defineProperty(input, 'files', { configurable: true, value: [new File([], 'capture.nacelle-profile')] });
  input.dispatchEvent(new Event('change'));
}

async function profile(events = [span()]): Promise<Profile> {
  return loadProfile(FileProfileDataSource.fromBuffer(fixture({}, trace(events))));
}

describe('offline viewer DOM', () => {
  beforeEach(() => {
    document.documentElement.innerHTML = html;
  });

  it('renders incomplete status and untrusted fields only as text', async () => {
    const loaded = await profile([span({
      name: '<img src=x onerror=alert(1)>',
      args: { value: '<script>bad()</script>' },
    })]);
    loaded.manifest.complete = false;
    loaded.manifest.droppedEvents = 5;
    mountViewer(document, async () => source(loaded));
    chooseFile();
    await vi.waitFor(() => expect(document.querySelector('#status')!.textContent).toContain('Incomplete'));
    expect(document.querySelector('#status')!.textContent).toContain('5 dropped');
    expect(document.querySelector('#threads img, #threads script')).toBeNull();
    expect(document.querySelector('#threads')!.textContent).toContain('<img src=x onerror=alert(1)>');
    expect(document.querySelector('#threads')!.textContent).toContain('1250.5');
    expect(document.querySelector<HTMLButtonElement>('#export')!.disabled).toBe(false);
  });

  it('pages timing spans and handles captures with no spans', async () => {
    const loaded = await profile(Array.from({ length: 251 }, (_, ts) => span({ ts })));
    mountViewer(document, async () => source(loaded));
    chooseFile();
    await vi.waitFor(() => expect(document.querySelectorAll('tbody tr')).toHaveLength(250));
    document.querySelector<HTMLButtonElement>('#more')!.click();
    expect(document.querySelectorAll('tbody tr')).toHaveLength(251);
    expect(document.querySelector<HTMLButtonElement>('#more')!.hidden).toBe(true);
    loaded.traceBytes = new TextEncoder().encode(JSON.stringify(trace([]))).buffer;
    chooseFile();
    await vi.waitFor(() => expect(document.querySelector('#threads')!.textContent).toContain('No timing spans'));
    expect(document.querySelector<HTMLButtonElement>('#export-speedscope')!.disabled).toBe(true);
  });

  it('clears stale capture and disables export when a new archive is invalid', async () => {
    const loaded = await profile();
    const open = vi.fn().mockResolvedValueOnce(source(loaded)).mockRejectedValueOnce(new Error('Corrupt ZIP'));
    mountViewer(document, open);
    chooseFile();
    await vi.waitFor(() => expect(document.querySelector<HTMLButtonElement>('#export')!.disabled).toBe(false));
    chooseFile();
    expect(document.querySelector<HTMLButtonElement>('#export')!.disabled).toBe(true);
    expect(document.querySelector<HTMLButtonElement>('#export-speedscope')!.disabled).toBe(true);
    await vi.waitFor(() => expect(document.querySelector('#status')!.textContent).toContain('Invalid profile: Corrupt ZIP'));
    expect(document.querySelector<HTMLElement>('#profile')!.hidden).toBe(true);
    expect(document.querySelectorAll('tbody tr')).toHaveLength(0);
  });

  it('applies filters, resets pagination, rejects invalid windows and leaves the original export available', async () => {
    const loaded = await profile([
      ...Array.from({ length: 251 }, (_, ts) => span({ ts, dur: 1, args: { pso: 'abc', stage: 'compute' } })),
      span({ pid: 2, ts: 20, args: { pso: 'def', stage: 'pixel' } }),
    ]);
    mountViewer(document, async () => source(loaded));
    chooseFile();
    await vi.waitFor(() => expect(document.querySelectorAll('tbody tr')).toHaveLength(250));
    expect(document.querySelectorAll('#stage-filter option')).toHaveLength(3);
    const set = (id: string, value: string) => {
      document.querySelector<HTMLInputElement | HTMLSelectElement>(id)!.value = value;
    };
    const apply = () => document.querySelector('#filters')!.dispatchEvent(new Event('submit', { cancelable: true }));
    set('#thread-filter', '1:2');
    set('#pso-filter', 'ABC');
    set('#stage-filter', 'compute');
    set('#start-filter', '100.5');
    set('#end-filter', '101.5');
    apply();
    expect(document.querySelectorAll('tbody tr')).toHaveLength(2);
    expect(document.querySelector('#filter-status')!.textContent).toContain('2 of 252');
    expect(document.querySelector<HTMLButtonElement>('#more')!.hidden).toBe(true);
    set('#start-filter', '200');
    apply();
    expect(document.querySelector('#filter-status')!.textContent).toContain('Invalid filters');
    expect(document.querySelector<HTMLButtonElement>('#export-speedscope')!.disabled).toBe(true);
    expect(document.querySelector<HTMLButtonElement>('#export')!.disabled).toBe(false);
    document.querySelector<HTMLButtonElement>('#clear-filters')!.click();
    expect(document.querySelectorAll('tbody tr')).toHaveLength(250);
    expect(document.querySelector<HTMLButtonElement>('#export-speedscope')!.disabled).toBe(false);
    set('#pso-filter', 'missing');
    apply();
    expect(document.querySelector<HTMLButtonElement>('#export-speedscope')!.disabled).toBe(true);
    expect(document.querySelector('#filter-status')!.textContent).toContain('0 of 252');
  });

  it('exports applied filters for Speedscope and reports crossing overlaps without breaking Chrome export', async () => {
    const loaded = await profile([
      span({ ts: 0, dur: 10 }), span({ ts: 5, dur: 10 }),
      span({ pid: 2, ts: 20, dur: 1, args: { stage: 'compute' } }),
    ]);
    const create = vi.fn().mockReturnValue('blob:local-profile');
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: create });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    let download = '';
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      download = this.download;
    });
    mountViewer(document, async () => source(loaded));
    chooseFile();
    await vi.waitFor(() => expect(document.querySelector<HTMLButtonElement>('#export-speedscope')!.disabled).toBe(false));
    vi.useFakeTimers();
    try {
      document.querySelector<HTMLButtonElement>('#export-speedscope')!.click();
      expect(document.querySelector('#export-status')!.textContent).toContain('crossing overlaps');
      expect(create).not.toHaveBeenCalled();
      document.querySelector<HTMLButtonElement>('#export')!.click();
      expect(create).toHaveBeenCalledOnce();
      document.querySelector<HTMLSelectElement>('#thread-filter')!.value = '2:2';
      document.querySelector('#filters')!.dispatchEvent(new Event('submit', { cancelable: true }));
      document.querySelector<HTMLButtonElement>('#export-speedscope')!.click();
      expect(create).toHaveBeenCalledTimes(2);
      expect(create.mock.calls[1][0].type).toBe('application/json');
      expect(document.querySelector('#export-status')!.textContent).toContain('nothing was uploaded');
      expect(download).toBe('nacelle-speedscope.json');
      vi.runAllTimers();
    } finally {
      vi.useRealTimers();
      click.mockRestore();
    }
  });

  it('ignores stale async file reads and closes their data source', async () => {
    const loaded = await profile();
    const stale = source(loaded);
    let finish!: (value: ProfileDataSource) => void;
    const first = new Promise<ProfileDataSource>(resolve => { finish = resolve; });
    const open = vi.fn().mockReturnValueOnce(first).mockRejectedValueOnce(new Error('new selection'));
    mountViewer(document, open);
    chooseFile();
    chooseFile();
    await vi.waitFor(() => expect(document.querySelector('#status')!.textContent).toContain('new selection'));
    finish(stale);
    await vi.waitFor(() => expect(stale.close).toHaveBeenCalledOnce());
    expect(document.querySelector('#status')!.textContent).toContain('new selection');
  });

  it('exports the original decoded Chrome trace as a local blob and releases its URL', async () => {
    const loaded = await profile();
    const create = vi.fn().mockReturnValue('blob:local-trace');
    const revoke = vi.fn();
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: create });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    mountViewer(document, async () => source(loaded));
    chooseFile();
    await vi.waitFor(() => expect(document.querySelector<HTMLButtonElement>('#export')!.disabled).toBe(false));
    vi.useFakeTimers();
    try {
      document.querySelector<HTMLButtonElement>('#export')!.click();
      expect(create).toHaveBeenCalledOnce();
      expect(create.mock.calls[0][0].type).toBe('application/json');
      expect(create.mock.calls[0][0].size).toBe(loaded.traceBytes.byteLength);
      expect(click).toHaveBeenCalledOnce();
      vi.runAllTimers();
      expect(revoke).toHaveBeenCalledWith('blob:local-trace');
    } finally {
      vi.useRealTimers();
      click.mockRestore();
    }
  });
});
