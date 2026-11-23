import { readFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import { fixture, trace } from '../test/fixtures';
import { profileCorpus } from '../test/profile-corpus';
import { artifactFixture } from '../test/artifact-fixtures';

const origin = 'http://127.0.0.1:4173';
const dist = resolve('dist');
const browserChecks = new WeakMap<Page, { unexpectedRequests: string[]; pageErrors: string[] }>();

test.beforeEach(async ({ context, page }) => {
  const assets = new Set((await readdir(dist, { recursive: true }))
    .filter(path => path === 'index.html' || /^(assets|speedscope)\/.+\.(html|js|css|woff2|wasm|ico|png)$/.test(path)));
  const unexpectedRequests: string[] = [];
  const pageErrors: string[] = [];
  browserChecks.set(page, { unexpectedRequests, pageErrors });
  page.on('pageerror', error => pageErrors.push(error.message));

  // Only built assets can be served from disk; even localhost networking is offline.
  await context.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.slice('/viewer/'.length);
    if (url.origin !== origin || !url.pathname.startsWith('/viewer/')
      || request.method() !== 'GET' || !assets.has(path)) {
      unexpectedRequests.push(`${request.method()} ${request.url()}`);
      await route.abort();
      return;
    }
    await route.fulfill({ path: resolve(dist, path) });
  });
  await context.setOffline(true);
  await page.goto('index.html');
  await expect(page.locator('#profile-file')).toBeAttached();

});

test.afterEach(async ({ page }) => {
  const checks = browserChecks.get(page);
  expect(checks).toBeDefined();
  expect(checks?.unexpectedRequests, 'No uploads, telemetry, CDNs, or profile fetches').toEqual([]);
  expect(checks?.pageErrors, 'No unhandled browser errors').toEqual([]);
});

async function downloadBytes(page: Page, button: string): Promise<Buffer> {
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.locator(button).click(),
  ]);
  expect(await download.failure()).toBeNull();
  const path = await download.path();
  if (!path) throw new Error('Browser download has no local path.');
  return readFile(path);
}

test('uploads a native capture offline, renders spans, and exports exact native trace bytes', async ({ page }) => {
  const capturePath = process.env.NACELLE_NATIVE_PROFILE;
  const tracePath = process.env.NACELLE_NATIVE_TRACE;
  if (!capturePath || !tracePath) {
    throw new Error('NACELLE_NATIVE_PROFILE and NACELLE_NATIVE_TRACE must name a native capture and CLI trace export.');
  }
  const expectedBytes = await readFile(tracePath);
  const expectedTrace = JSON.parse(expectedBytes.toString('utf8'));
  expect(expectedTrace.traceEvents.some((event: { name: string }) => event.name === 'Compute dispatch')).toBe(true);
  expect(expectedTrace.traceEvents.some((event: { name: string }) => event.name === 'Graphics draw')).toBe(true);

  await page.locator('#profile-file').setInputFiles(capturePath);
  await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
  await expect(page.locator('#profile')).toBeVisible();
  await expect(page.locator('#summary')).toContainText(`Trace events: ${expectedTrace.traceEvents.length}`);
  await expect(page.locator('#threads')).toContainText('Compute dispatch');

  await page.locator('#stage-filter').selectOption('compute');
  await page.locator('#filters').evaluate((form: HTMLFormElement) => form.requestSubmit());
  await expect(page.locator('#threads')).toContainText('Compute dispatch');
  await expect(page.locator('#threads')).not.toContainText('Graphics draw');
  await expect(page.frameLocator('#timeline-host iframe').getByText(/Time Order/)).toBeVisible();
  expect(await downloadBytes(page, '#export')).toEqual(expectedBytes);

  const speedscope = JSON.parse((await downloadBytes(page, '#export-speedscope')).toString('utf8'));
  expect(speedscope.profiles.length).toBeGreaterThan(0);
  expect(speedscope.profiles.every((profile: { unit: string }) => profile.unit === 'microseconds')).toBe(true);
  expect(speedscope.shared.frames.some((frame: { name: string }) => frame.name.includes('Compute dispatch'))).toBe(true);
  expect(speedscope.shared.frames.some((frame: { name: string }) => frame.name.includes('Graphics draw'))).toBe(false);
});

test('keeps incomplete and empty captures viewable offline', async ({ page }) => {
  await page.locator('#profile-file').setInputFiles({
    name: 'incomplete.nacelle-profile', mimeType: 'application/octet-stream',
    buffer: Buffer.from(fixture({ complete: false, droppedEvents: 7 })),
  });
  await expect(page.locator('#status')).toHaveAttribute('data-state', 'incomplete');
  await expect(page.locator('#status')).toContainText('7 dropped events');
  await expect(page.locator('#threads')).toContainText('dispatch');
  expect(JSON.parse((await downloadBytes(page, '#export')).toString('utf8'))).toEqual(trace());

  await page.locator('#profile-file').setInputFiles({
    name: 'empty.nacelle-profile', mimeType: 'application/octet-stream',
    buffer: Buffer.from(fixture({ recordedEvents: 0 }, trace([]))),
  });
  await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
  await expect(page.locator('#profile')).toBeVisible();
  await expect(page.locator('#threads')).toContainText('No timing spans');
  await expect(page.locator('#export-speedscope')).toBeDisabled();
  expect(JSON.parse((await downloadBytes(page, '#export')).toString('utf8'))).toEqual(trace([]));
});

test('rejects malformed uploads and recovers on the next valid local file', async ({ page }) => {
  await page.locator('#profile-file').setInputFiles({
    name: 'truncated.nacelle-profile', mimeType: 'application/octet-stream',
    buffer: Buffer.from(fixture().slice(0, -1)),
  });
  await expect(page.locator('#status')).toHaveAttribute('data-state', 'invalid');
  await expect(page.locator('#status')).toContainText('Invalid profile:');
  await expect(page.locator('#profile')).toBeHidden();
  await expect(page.locator('#export')).toBeDisabled();

  await page.locator('#profile-file').setInputFiles({
    name: 'valid.nacelle-profile', mimeType: 'application/octet-stream',
    buffer: Buffer.from(fixture()),
  });
  await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
  await expect(page.locator('#threads')).toContainText('dispatch');
  await expect(page.locator('#export')).toBeEnabled();
});

test('keeps the UI responsive while a large profile is decoded in a worker', async ({ page }) => {
    const events = Array.from({ length: 80_000 }, (_, ts) => ({
      name: 'Large capture dispatch', cat: 'compute', ph: 'X' as const,
      ts, dur: 1, pid: 1, tid: 1, args: { stage: 'compute' },
    }));
    expect(Buffer.byteLength(JSON.stringify(trace(events)))).toBeGreaterThan(8 * 1024 * 1024);
    let workers = 0;
    page.on('worker', () => workers++);
    await page.evaluate(() => {
      const status = document.querySelector('#status')!;
      status.setAttribute('data-heartbeats', '0');
      const timer = setInterval(() => {
        if (status.getAttribute('data-state') === 'loading')
          status.setAttribute('data-heartbeats', String(Number(status.getAttribute('data-heartbeats')) + 1));
        else if (status.getAttribute('data-state') !== 'idle') clearInterval(timer);
      }, 10);
    });
    await page.locator('#profile-file').setInputFiles({
      name: 'large.nacelle-profile', mimeType: 'application/zip', buffer: Buffer.from(fixture({}, trace(events))),
    });
    await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
    expect(workers).toBe(1);
    expect(Number(await page.locator('#status').getAttribute('data-heartbeats'))).toBeGreaterThanOrEqual(2);
    await expect(page.locator('#summary')).toContainText('Trace events: 80000');
    await expect(page.locator('#cancel-load')).toBeHidden();
  });

test('cancels decoding, discards its result, and loads the next checksummed file offline', async ({ page }) => {
    await page.evaluate(() => {
      document.querySelector('#profile-file')!.addEventListener('change', () => {
        document.querySelector<HTMLButtonElement>('#cancel-load')!.click();
      }, { once: true });
    });

    await page.locator('#profile-file').setInputFiles({
      name: 'cancel.nacelle-profile', mimeType: 'application/zip', buffer: Buffer.from(fixture()),
    });
    await expect(page.locator('#status')).toHaveAttribute('data-state', 'cancelled');
    await expect(page.locator('#export')).toBeDisabled();
    await expect(page.locator('#profile')).toBeHidden();
    const checksum = profileCorpus().find(entry => entry.name === 'zstd-checksum-valid')!;
    await page.locator('#profile-file').setInputFiles({
      name: 'checksum.nacelle-profile', mimeType: 'application/zip', buffer: Buffer.from(checksum.bytes),
    });
    await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
    await expect(page.locator('#summary')).toContainText('Trace events: 1');
  });

test('navigates offline read-only shader panes by exact version and preserved source mappings', async ({ page }) => {
  await page.locator('#profile-file').setInputFiles({
    name: 'shader-artifacts.nacelle-profile', mimeType: 'application/zip', buffer: Buffer.from(await artifactFixture()),
  });
  await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
  await expect(page.locator('#shader-artifacts')).toBeVisible();
  await expect(page.locator('[data-artifact-kind="assembly"]')).toContainText('Unavailable');
  await expect(page.locator('[data-artifact-kind="hlsl"] .cm-content')).toHaveAttribute('contenteditable', 'false');
  expect(await page.locator('[data-artifact-kind="hlsl"] img').count()).toBe(0);
  await page.locator('[data-artifact-kind="hlsl"] .cm-line').nth(1).click();
  await expect(page.locator('#artifact-status')).toContainText('source line 2');
  await expect(page.locator('[data-artifact-kind="dxil"] .cm-activeLine')).toContainText('%b');
  await page.locator('#artifact-version').selectOption('1');
  await expect(page.locator('[data-artifact-kind="assembly"] .cm-content')).toContainText('0: ret');
  await expect(page.locator('#artifact-status')).toContainText('native: available');
  const capture = process.env.NACELLE_NATIVE_ARTIFACT_PROFILE;
  if (capture) {
    await page.locator('#profile-file').setInputFiles(capture);
    await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
    await expect(page.locator('[data-artifact-kind="hlsl"] .cm-content')).toContainText('CSMain');
    await expect(page.locator('[data-artifact-kind="assembly"]')).toContainText('0:');
    await expect(page.locator('#artifact-status')).toContainText('native: available');
  }
});
