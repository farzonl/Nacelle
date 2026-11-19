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
  });

  it('clears stale capture and disables export when a new archive is invalid', async () => {
    const loaded = await profile();
    const open = vi.fn().mockResolvedValueOnce(source(loaded)).mockRejectedValueOnce(new Error('Corrupt ZIP'));
    mountViewer(document, open);
    chooseFile();
    await vi.waitFor(() => expect(document.querySelector<HTMLButtonElement>('#export')!.disabled).toBe(false));
    chooseFile();
    expect(document.querySelector<HTMLButtonElement>('#export')!.disabled).toBe(true);
    await vi.waitFor(() => expect(document.querySelector('#status')!.textContent).toContain('Invalid profile: Corrupt ZIP'));
    expect(document.querySelector<HTMLElement>('#profile')!.hidden).toBe(true);
    expect(document.querySelectorAll('tbody tr')).toHaveLength(0);
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
