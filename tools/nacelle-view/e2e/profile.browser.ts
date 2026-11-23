import { readFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import { fixture, trace } from '../test/fixtures';
import { profileCorpus } from '../test/profile-corpus';
import { artifactFixture } from '../test/artifact-fixtures';
import { spatialArtifactFixture, spatialFixture } from '../test/spatial-fixtures';

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

test('decodes typed spatial observations offline and warns separately about spatial loss', async ({ page }) => {
  await page.locator('#profile-file').setInputFiles({
    name: 'spatial.nacelle-profile', mimeType: 'application/zip', buffer: Buffer.from(await spatialFixture()),
  });
  await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
  await expect(page.locator('#summary')).toContainText('Spatial observations: 2; skipped: 4; dropped: 0');
  await page.locator('#profile-file').setInputFiles({
    name: 'spatial-incomplete.nacelle-profile', mimeType: 'application/zip',
    buffer: Buffer.from(await spatialFixture(metadata => {
      metadata.complete = false;
      metadata.dropped = 1;
      const operation = metadata.operations[0];
      operation.complete = false;
      operation.dropped = 1;
      operation.missingStatus = 'dropped-or-unsampled';
      for (const cell of operation.cells) cell.status = 'partial';
    })),
  });
  await expect(page.locator('#status')).toHaveAttribute('data-state', 'incomplete');
  await expect(page.locator('#status')).toContainText('1 dropped spatial observations');
  const capture = process.env.NACELLE_NATIVE_SPATIAL_PROFILE;
  if (capture) {
    await page.locator('#profile-file').setInputFiles(capture);
    await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
    await expect(page.locator('#summary')).toContainText('Spatial wall time: host entry-call ns per tile/threadgroup');
  }
});

  test('renders true pixel edges, observed zero and unsampled gaps, with exact quad-slot inspection offline', async ({ page }) => {
    await page.locator('#profile-file').setInputFiles({
      name: 'pixel-spatial.nacelle-profile', mimeType: 'application/zip',
      buffer: Buffer.from(await spatialFixture(metadata => { metadata.operations[0].extent = [65, 33, 1]; }, 'pixel')),
    });
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
    await expect(page.locator('#spatial-details')).toContainText('XYZ 0, 0, 0: Observed');
    await expect(page.locator('#spatial-details')).toContainText('0 host entry-call ns');
    const samples = await page.locator('#spatial-canvas').evaluate((canvas: HTMLCanvasElement) => {
      const context = canvas.getContext('2d')!;
      const pixel = (x: number, y: number) => Array.from(context.getImageData(
        Math.floor(x / 65 * canvas.width), Math.floor(y / 33 * canvas.height), 1, 1).data);
      const rect = canvas.getBoundingClientRect();
      return { zero: pixel(1, 1), missing: pixel(40, 1), hot: pixel(64.5, 1),
        ratio: rect.width / rect.height, pixels: canvas.width * canvas.height };
    });
    expect(samples.zero).toEqual([30, 80, 220, 255]);
    expect(samples.missing).toEqual([51, 51, 51, 255]);
    expect(samples.hot).toEqual([255, 180, 30, 255]);
    expect(samples.ratio).toBeCloseTo(65 / 33, 2);
    expect(samples.pixels).toBeLessThanOrEqual(1024 * 1024);
    const box = await page.locator('#spatial-canvas').boundingBox();
    if (!box) throw new Error('Visible pixel heatmap has no bounding box');
    await page.locator('#spatial-canvas').click({ position: { x: box.width * 64.5 / 65, y: box.height / 33 } });
    await expect(page.locator('#spatial-details')).toContainText('XYZ 2, 0, 0: Observed');
    await page.locator('#spatial-x').fill('1');
    await page.locator('#spatial-inspect').click();
    await expect(page.locator('#spatial-details')).toContainText('Unsampled');
    await page.locator('#spatial-x').fill('2');
    await page.locator('#spatial-metric').selectOption('coveredLanes');
    await page.locator('#spatial-lane').selectOption('6');
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
    await expect(page.locator('#spatial-details')).toContainText('1 covered submissions (slot 6)');
    await expect(page.locator('#spatial-details')).toContainText('Selected slot 6: quad 1, local XY 0, 1');
    await page.locator('#spatial-metric').selectOption('wallNs');
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
    await expect(page.locator('#spatial-lane')).toBeDisabled();
    await expect(page.locator('#spatial-details')).toContainText('20 host entry-call ns;');
    await expect(page.locator('#spatial-details')).not.toContainText('ns (slot');
    await page.locator('#spatial-metric').selectOption('coveredLanes');
    await page.locator('#spatial-scale').selectOption('log');
    await page.locator('#spatial-max').fill('2');
    await page.locator('#spatial-max').dispatchEvent('change');
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
    await expect(page.locator('#spatial-legend')).toContainText('0 to 2 covered submissions (log');
    await page.locator('#spatial-min').fill('3');
    await page.locator('#spatial-min').dispatchEvent('change');
    await expect(page.locator('#spatial-status')).toContainText('Invalid range');
    await expect(page.locator('#spatial-canvas')).toBeHidden();
    for (const width of [4, 16]) {
      await page.locator('#profile-file').setInputFiles({
        name: `pixel-simd${width}.nacelle-profile`, mimeType: 'application/zip',
        buffer: Buffer.from(await spatialFixture(undefined, 'pixel', width)),
      });
      await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
      await page.locator('#spatial-metric').selectOption('coveredLanes');
      await page.locator('#spatial-lane').selectOption(String(width - 1));
      await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
      await expect(page.locator('#spatial-details')).toContainText(`1 covered submissions (slot ${width - 1})`);
      await expect(page.locator('#spatial-details')).toContainText(`SIMD ${width}`);
    }
  });

  test('selects 3D group planes and correlates immutable versions with spans and artifacts offline', async ({ page }) => {
    await page.locator('#profile-file').setInputFiles({
      name: 'spatial-artifacts.nacelle-profile', mimeType: 'application/zip',
      buffer: Buffer.from(await spatialArtifactFixture()),
    });
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
    await page.locator('#spatial-correlate').click();
    await expect(page.locator('#artifact-version')).toHaveValue('1');
    await expect(page.locator('[data-artifact-kind="assembly"] .cm-content')).toContainText('0: ret');
    await expect(page.locator('#spatial-details')).toContainText('code ID 2; version 2; Tier 2');
    await expect(page.locator('#spatial-details')).toContainText('Captured span 2:');
    await expect(page.locator('#stage-filter')).toHaveValue('compute');
    await expect(page.locator('#start-filter')).toHaveValue('2');
    await page.locator('#spatial-plane').selectOption('xz');
    await page.locator('#spatial-slice').fill('1');
    await page.locator('#spatial-slice').dispatchEvent('change');
    await expect(page.locator('#spatial-status')).toContainText('0 observed cells on XZ slice 1');
    await page.locator('#spatial-y').fill('1');
    await page.locator('#spatial-inspect').click();
    await expect(page.locator('#spatial-details')).toContainText('Unsampled');
    await page.locator('#spatial-plane').selectOption('yz');
    await page.locator('#spatial-slice').fill('2');
    await page.locator('#spatial-slice').dispatchEvent('change');
    await expect(page.locator('#spatial-status')).toContainText('1 observed cells on YZ slice 2');
    await page.locator('#spatial-x').fill('2');
    await page.locator('#spatial-y').fill('0');
    await page.locator('#spatial-z').fill('2');
    await page.locator('#spatial-inspect').click();
    await expect(page.locator('#spatial-details')).toContainText('XYZ 2, 0, 2: Observed');
    await expect(page.locator('#spatial-details')).toContainText('20 host entry-call ns');
    await page.locator('#spatial-plane').selectOption('xz');
    await page.locator('#spatial-slice').fill('2');
    await page.locator('#spatial-slice').dispatchEvent('change');
    await expect(page.locator('#spatial-status')).toContainText('Invalid slice');
    await page.locator('#spatial-slice').fill('0');
    await page.locator('#spatial-slice').dispatchEvent('change');
    await page.locator('#spatial-metric').selectOption('coveredLanes');
    await expect(page.locator('#spatial-status')).toContainText('compute lane submission layout is not captured');
    await expect(page.locator('#spatial-details')).toContainText('Value: unavailable, never inferred as zero.');
    await expect(page.locator('#spatial-details')).not.toContainText('0 covered submissions');
    await expect(page.locator('#spatial-details')).toContainText('XYZ 2, 0, 2: Observed');
  });

  test('warns about partial and unsupported spatial precision without fabricating values', async ({ page }) => {
    await page.locator('#profile-file').setInputFiles({
      name: 'spatial-loss.nacelle-profile', mimeType: 'application/zip',
      buffer: Buffer.from(await spatialFixture(metadata => {
        metadata.complete = false;
        metadata.dropped = 1;
        const operation = metadata.operations[0];
        operation.complete = false;
        operation.dropped = 1;
        operation.missingStatus = 'dropped-or-unsampled';
        for (const cell of operation.cells) cell.status = 'partial';
        operation.cells[1].xyz = [2, 2, 0];
        operation.dimensions = operation.extent = [3, 3, 1];
      })),
    });
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
    await expect(page.locator('#spatial-status')).toContainText('Warning: capture incomplete');
    await expect(page.locator('#spatial-details')).toContainText('Partial');
    await page.locator('#spatial-x').fill('2');
    await page.locator('#spatial-inspect').click();
    await expect(page.locator('#spatial-details')).toContainText('Dropped-or-missing');
    await page.locator('#profile-file').setInputFiles({
      name: 'spatial-unavailable.nacelle-profile', mimeType: 'application/zip',
      buffer: Buffer.from(await spatialFixture(metadata => {
        metadata.recorded = metadata.skipped = 0;
        const operation = metadata.operations[0];
        operation.recorded = operation.skipped = 0;
        operation.cells = [];
        operation.availability = 'dispatch-wide-entry-unavailable';
        operation.missingStatus = 'unavailable';
      })),
    });
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'ready');
    await expect(page.locator('#spatial-details')).toContainText('Unavailable: dispatch-wide');
    await expect(page.locator('#spatial-details')).toContainText('Value: unavailable, never inferred as zero');
    await page.locator('#profile-file').setInputFiles({
      name: 'spatial-empty.nacelle-profile', mimeType: 'application/zip',
      buffer: Buffer.from(await spatialFixture(metadata => {
        metadata.recorded = metadata.skipped = 0;
        metadata.operations = [];
      })),
    });
    await expect(page.locator('#spatial-status')).toContainText('No eligible spatial operations');
    await expect(page.locator('#spatial-canvas')).toBeHidden();
  });

  test('renders the full 39568-cell pool with bounded canvas and heartbeats, then replaces all spatial state', async ({ page }) => {
    await page.locator('#profile-file').setInputFiles({
      name: 'large-spatial.nacelle-profile', mimeType: 'application/zip',
      buffer: Buffer.from(await spatialFixture(metadata => {
        metadata.recordBytes = 16 * 1024 * 1024;
        metadata.sampleStride = 1;
        metadata.recorded = 39568;
        metadata.skipped = 0;
        const operation = metadata.operations[0];
        operation.dimensions = operation.extent = [200, 198, 1];
        operation.recorded = 39568;
        operation.skipped = 0;
        operation.cells = Array.from({ length: 39568 }, (_, index) => ({
          xyz: [index % 200, Math.floor(index / 200), 0], status: 'observed', calls: 1,
          wallNs: index, invocations: 8, coveredLanes: [], companionLanes: [], paddingLanes: [],
        }));
      })),
    });
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'rendering');
    const heartbeats = await page.evaluate(async () => {
      let beats = 0;
      const timer = setInterval(() => beats++, 10);
      while (document.querySelector<HTMLElement>('#spatial-status')!.dataset.state === 'rendering')
        await new Promise(resolve => setTimeout(resolve, 10));
      clearInterval(timer);
      return beats;
    });
    expect(heartbeats).toBeGreaterThanOrEqual(2);
    await expect(page.locator('#spatial-status')).toContainText('39568 observed cells');
    await page.locator('#spatial-x').fill('167');
    await page.locator('#spatial-y').fill('197');
    await page.locator('#spatial-inspect').click();
    await expect(page.locator('#spatial-details')).toContainText('39567 host entry-call ns');
    const area = await page.locator('#spatial-canvas').evaluate((canvas: HTMLCanvasElement) => canvas.width * canvas.height);
    expect(area).toBeLessThanOrEqual(1024 * 1024);
    await page.locator('#spatial-scale').selectOption('log');
    await expect(page.locator('#spatial-status')).toHaveAttribute('data-state', 'rendering');
    await page.locator('#profile-file').setInputFiles({
      name: 'no-spatial.nacelle-profile', mimeType: 'application/zip', buffer: Buffer.from(fixture()),
    });
    await expect(page.locator('#status')).toHaveAttribute('data-state', 'complete');
    await expect(page.locator('#spatial-view')).toBeHidden();
    await expect(page.locator('#spatial-details')).toBeEmpty();
  });
