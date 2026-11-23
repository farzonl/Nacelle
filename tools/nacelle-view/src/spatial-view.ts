import type { Profile } from './profile';
import type { SpatialCell, SpatialOperation, XYZ } from './spatial';
import { cellRectangle, coordinateStatus, heatColor, metricUnits, planeAxes, spatialValue,
  type SpatialMetric, type SpatialPlane } from './heatmap';

export function createSpatialView(root: Document, correlate: (operation: SpatialOperation) => void): {
  show(profile?: Profile): void;
} {
  const section = root.querySelector<HTMLElement>('#spatial-view')!;
  const operationSelect = root.querySelector<HTMLSelectElement>('#spatial-operation')!;
  const metricSelect = root.querySelector<HTMLSelectElement>('#spatial-metric')!;
  const laneSelect = root.querySelector<HTMLSelectElement>('#spatial-lane')!;
  const planeSelect = root.querySelector<HTMLSelectElement>('#spatial-plane')!;
  const slice = root.querySelector<HTMLInputElement>('#spatial-slice')!;
  const scale = root.querySelector<HTMLSelectElement>('#spatial-scale')!;
  const minimum = root.querySelector<HTMLInputElement>('#spatial-min')!;
  const maximum = root.querySelector<HTMLInputElement>('#spatial-max')!;
  const coordinates = ['x', 'y', 'z'].map(axis => root.querySelector<HTMLInputElement>(`#spatial-${axis}`)!);
  const canvas = root.querySelector<HTMLCanvasElement>('#spatial-canvas')!;
  const status = root.querySelector<HTMLElement>('#spatial-status')!;
  const legend = root.querySelector<HTMLElement>('#spatial-legend')!;
  const details = root.querySelector<HTMLElement>('#spatial-details')!;
  const inspectButton = root.querySelector<HTMLButtonElement>('#spatial-inspect')!;
  const link = root.querySelector<HTMLButtonElement>('#spatial-correlate')!;
  let profile: Profile | undefined;
  let lookup = new Map<string, SpatialCell>();
  let generation = 0;
  let frame = 0;
  const win = root.defaultView!;
  const operation = () => profile?.spatial?.operations[Number(operationSelect.value)];
  const metric = () => metricSelect.value as SpatialMetric;
  const plane = () => planeSelect.value as SpatialPlane;
  const option = (select: HTMLSelectElement, value: string, text: string) => {
    const item = root.createElement('option');
    item.value = value;
    item.textContent = text;
    select.append(item);
  };
  function inspect(): void {
    const selected = operation();
    const metadata = profile?.spatial;
    if (!selected || !metadata) return;
    const xyz: XYZ = [coordinates[0].valueAsNumber, coordinates[1].valueAsNumber, coordinates[2].valueAsNumber];
    if (xyz.some((value, axis) => !Number.isSafeInteger(value) || value < 0 || value >= selected.dimensions[axis])) {
      details.textContent = 'Invalid coordinate: enter integer XYZ within the operation dimensions.';
      return;
    }
    const cell = lookup.get(xyz.join(','));
    const code = profile?.shaders.codeVersions[selected.codeVersionId - 1];
    const span = profile?.trace.traceEvents.find(event => event.args.id === selected.eventId);
    const lane = metric().endsWith('Lanes') ? Number(laneSelect.value) : -1;
    details.textContent = [
      `XYZ ${xyz.join(', ')}: ${coordinateStatus(metadata, selected, xyz, cell)}`,
      cell ? `${spatialValue(cell, metric(), lane)} ${metricUnits[metric()]}${lane >= 0 ? ` (slot ${lane})` : ''}; ${cell.calls} calls; ${cell.invocations} invocations; ${cell.wallNs} host ns` : 'Value: unavailable, never inferred as zero.',
      `Pipeline ${selected.pipelineId}; ${selected.stage}; PSO ${selected.pso}; code ID ${selected.codeVersionId}; version ${code?.version}; Tier ${code?.tier}; ${code?.cache}; SIMD ${selected.simdWidth}`,
      `Extent ${selected.extent.join(' x ')}; cells ${selected.dimensions.join(' x ')}; cell extent ${selected.cellExtent.join(' x ')}; ${selected.orientation}`,
      span ? `Captured span ${selected.eventId}: ${span.name}; ${span.dur} us on ${span.pid}:${span.tid}` : `Captured span ${selected.eventId}: unavailable (event recording may have dropped it).`,
      profile?.artifacts ? 'Artifact version available through the correlation button; individual panes/mappings may be unavailable.' : 'Artifacts: not retained in this capture.',
      selected.stage === 'pixel' ? `Quad-major slots: quad = floor(slot / 4), local XY = (slot % 2, floor(slot / 2) % 2). Counts are pre-depth/pre-discard submissions, not unique pixels or instruction activity.${lane >= 0 ? ` Selected slot ${lane}: quad ${Math.floor(lane / 4)}, local XY ${lane % 2}, ${Math.floor(lane / 2) % 2}.` : ''}` :
        `Threads per group ${selected.threadsPerGroup.join(' x ')}. Compute SIMD lane layout/values unavailable.`,
      'Source-lane values and dynamic divergence unavailable. Artifact correlation never infers source mapping.',
    ].join('\n');
  }
  function render(): void {
    const selected = operation();
    const metadata = profile?.spatial;
    const token = ++generation;
    win.cancelAnimationFrame(frame);
    lookup = new Map();
    canvas.hidden = !selected;
    link.disabled = !selected;
    details.textContent = '';
    inspectButton.disabled = true;
    status.dataset.state = 'idle';
    legend.textContent = '';
    if (!selected || !metadata) {
      status.textContent = 'No eligible spatial operations in this capture.';
      return;
    }
    const [horizontal, vertical, sliced] = planeAxes[plane()];
    const layer = slice.valueAsNumber;
    slice.max = String(selected.dimensions[sliced] - 1);
    if (!Number.isSafeInteger(layer) || layer < 0 || layer >= selected.dimensions[sliced]) {
      status.textContent = `Invalid slice: integer 0 through ${slice.max} required.`;
      canvas.hidden = true;
      return;
    }
    coordinates[sliced].value = String(layer);
    const low = minimum.valueAsNumber;
    const explicitHigh = maximum.value === '' ? undefined : maximum.valueAsNumber;
    if (!Number.isFinite(low) || low < 0 || low >= Number.MAX_SAFE_INTEGER ||
      (explicitHigh !== undefined && (!Number.isFinite(explicitHigh) || explicitHigh <= low || explicitHigh > Number.MAX_SAFE_INTEGER))) {
      status.textContent = 'Invalid range: require 0 <= minimum < maximum <= 2^53 - 1.';
      canvas.hidden = true;
      return;
    }
    const ratio = selected.extent[horizontal] / selected.extent[vertical];
    canvas.width = Math.max(1, Math.round(1024 * Math.min(1, ratio)));
    canvas.height = Math.max(1, Math.round(1024 * Math.min(1, 1 / ratio)));
    canvas.style.aspectRatio = `${selected.extent[horizontal]} / ${selected.extent[vertical]}`;
    canvas.style.width = `${canvas.width}px`;
    canvas.style.height = 'auto';
    const context = canvas.getContext('2d');
    if (!context) {
      status.textContent = 'Heatmap unavailable: this browser has no 2D canvas context.';
      return;
    }
    context.fillStyle = '#333333';
    context.fillRect(0, 0, canvas.width, canvas.height);
    status.dataset.state = 'rendering';
    status.textContent = 'Preparing bounded sparse heatmap...';
    let index = 0;
    let high = explicitHigh ?? low + 1;
    let visible = 0;
    laneSelect.disabled = selected.stage !== 'pixel' || !metric().endsWith('Lanes');
    const lane = metric().endsWith('Lanes') ? Number(laneSelect.value) : -1;
    const requestedMetric = metric();
    if (selected.stage !== 'pixel' && requestedMetric.endsWith('Lanes')) {
      status.textContent = 'Unavailable: compute lane submission layout is not captured.';
      status.dataset.state = 'ready';
      legend.textContent = 'Gray: unavailable, no numeric value.';
      inspectButton.disabled = false;
      inspect();
      return;
    }
    function prepare(): void {
      if (token !== generation) return;
      const end = Math.min(index + 1000, selected!.cells.length);
      for (; index < end; index++) {
        const cell = selected!.cells[index];
        lookup.set(cell.xyz.join(','), cell);
        if (cell.xyz[sliced] !== layer) continue;
        visible++;
        if (explicitHigh === undefined) high = Math.max(high, spatialValue(cell, requestedMetric, lane));
      }
      if (index < selected!.cells.length) frame = win.requestAnimationFrame(prepare);
      else {
        index = 0;
        frame = win.requestAnimationFrame(draw);
      }
    }
    function draw(): void {
      if (token !== generation) return;
      const end = Math.min(index + 1000, selected!.cells.length);
      for (; index < end; index++) {
        const cell = selected!.cells[index];
        if (cell.xyz[sliced] !== layer) continue;
        context!.fillStyle = heatColor(spatialValue(cell, requestedMetric, lane), low, high, scale.value === 'log');
        const [x, y, width, height] = cellRectangle(selected!, cell.xyz, plane(), canvas.width, canvas.height);
        context!.fillRect(x, y, width, height);
        if (cell.status === 'partial') {
          context!.strokeStyle = '#ff00ff';
          context!.strokeRect(x, y, width, height);
        }
      }
      if (index < selected!.cells.length) frame = win.requestAnimationFrame(draw);
      else {
        status.dataset.state = 'ready';
        inspectButton.disabled = false;
        status.textContent = `${visible} observed cells on ${plane().toUpperCase()} slice ${layer}. ${selected!.orientation}. ` +
          (!profile!.manifest.complete ? 'Warning: capture incomplete; missing coordinates or values may be unreliable. ' : '') +
          `${selected!.recorded} recorded; ${selected!.skipped} skipped; ${selected!.dropped} dropped. ` +
          (selected!.availability !== 'available' ? 'Per-group timing unavailable. ' : '') +
          'Sparse projection only; subpixel cells can overlap. Use exact XYZ inspection, not color, for exact values.';
        legend.textContent = `Blue to orange: ${low} to ${high} ${metricUnits[requestedMetric]} (${scale.value}; values clipped to range). ` +
          'Observed zero is numeric blue at range 0; gray = missing/unsampled/unavailable, not zero; magenta outline = partial. ' +
          `Horizontal ${'XYZ'[horizontal]}, vertical ${'XYZ'[vertical]} increasing downward; fixed ${'XYZ'[sliced]} = ${layer}.`;
        inspect();
      }
    }
    frame = win.requestAnimationFrame(prepare);
  }
  function selectOperation(): void {
    const selected = operation();
    laneSelect.replaceChildren();
    option(laneSelect, '-1', 'All submitted slots');
    if (selected?.stage === 'pixel')
      for (let lane = 0; lane < selected.simdWidth; lane++) option(laneSelect, String(lane), `Slot ${lane} (quad ${Math.floor(lane / 4)})`);
    laneSelect.disabled = selected?.stage !== 'pixel';
    slice.value = '0';
    coordinates.forEach((input, axis) => {
      input.value = '0';
      input.max = String((selected?.dimensions[axis] ?? 1) - 1);
    });
    render();
  }
  operationSelect.addEventListener('change', selectOperation);
  for (const input of [metricSelect, laneSelect, planeSelect, slice, scale, minimum, maximum])
    input.addEventListener('change', render);
  inspectButton.addEventListener('click', inspect);
  link.addEventListener('click', () => {
    const selected = operation();
    if (selected) correlate(selected);
  });
  canvas.addEventListener('click', event => {
    const selected = operation();
    if (!selected || status.dataset.state !== 'ready') return;
    const [horizontal, vertical, sliced] = planeAxes[plane()];
    const rect = canvas.getBoundingClientRect();
    const xyz: XYZ = [0, 0, 0];
    xyz[horizontal] = Math.min(selected.dimensions[horizontal] - 1,
      Math.floor((event.clientX - rect.left) / rect.width * selected.extent[horizontal] / selected.cellExtent[horizontal]));
    xyz[vertical] = Math.min(selected.dimensions[vertical] - 1,
      Math.floor((event.clientY - rect.top) / rect.height * selected.extent[vertical] / selected.cellExtent[vertical]));
    xyz[sliced] = slice.valueAsNumber;
    coordinates.forEach((input, axis) => { input.value = String(xyz[axis]); });
    inspect();
  });
  return {
    show(value) {
      profile = value;
      generation++;
      win.cancelAnimationFrame(frame);
      lookup.clear();
      section.hidden = !value?.spatial;
      operationSelect.replaceChildren();
      metricSelect.value = 'wallNs';
      planeSelect.value = 'xy';
      scale.value = 'linear';
      minimum.value = '0';
      maximum.value = '';
      for (const selected of value?.spatial?.operations ?? [])
        option(operationSelect, String(selected.id - 1),
          `Operation ${selected.id}: ${selected.stage}, pipeline ${selected.pipelineId}, code ${selected.codeVersionId}, SIMD ${selected.simdWidth}`);
      selectOperation();
    },
  };
}
