import { artifactDigest } from './artifacts';
import { object, parseJson, requireValid, type ProfileManifest, type ShaderMetadata, type Trace } from './profile';

export interface CounterSample {
  eventId: number;
  codeVersionId: number;
  region: 'compute-worker' | 'pixel-worker' | 'compute-direct';
  status: string;
  nativeError: number;
  cycles: string;
  instructions: string;
  timeEnabledNs: string;
  timeRunningNs: string;
}

export interface CounterMetadata {
  schemaVersion: 1;
  provider: 'linux-perf-event' | 'unsupported';
  providerImplemented: boolean;
  metricMask: number;
  cycleUnit: 'cycles';
  instructionUnit: 'instructions';
  timeUnit: 'ns';
  scope: 'thread-user-region';
  sampling: 'none';
  scaling: 'raw-with-enabled-running';
  availability: 'available' | 'partial' | 'unavailable' | 'unsupported-scope';
  kernelExcluded: true;
  hypervisorExcluded: true;
  inherited: false;
  complete: boolean;
  recordBytes: number;
  attempted: number;
  recorded: number;
  unavailable: number;
  dropped: number;
  records: CounterSample[];
}

const statuses = new Set(['available', 'permission-denied', 'unsupported-provider', 'unsupported-event',
  'not-running', 'io-failure', 'invalid-reading', 'resource-unavailable', 'unsupported-scope']);
const rootKeys = ['schemaVersion', 'provider', 'providerImplemented', 'metricMask', 'cycleUnit', 'instructionUnit', 'timeUnit', 'scope',
  'sampling', 'scaling', 'availability', 'kernelExcluded', 'hypervisorExcluded', 'inherited', 'complete',
  'recordBytes', 'attempted', 'recorded', 'unavailable', 'dropped', 'records'];
const recordKeys = ['eventId', 'codeVersionId', 'region', 'status', 'nativeError', 'cycles',
  'instructions', 'timeEnabledNs', 'timeRunningNs'];

function unsignedValue(value: unknown): bigint {
  requireValid(typeof value === 'string' && /^(0|[1-9][0-9]{0,19})$/.test(value),
    'Invalid lossless counter integer.');
  const number = BigInt(value);
  requireValid(number <= 18446744073709551615n, 'Counter integer exceeds uint64.');
  return number;
}

export async function validateCounters(
  bytes: ArrayBuffer, manifest: ProfileManifest, shaders: ShaderMetadata, trace: Trace,
): Promise<CounterMetadata> {
  const entry = manifest.entries.find(item => item.path === 'counters.json');
  requireValid(entry && bytes.byteLength === entry.uncompressedSize &&
    await artifactDigest(new Uint8Array(bytes)) === entry.digest, 'Counter size or digest mismatch.');
  const value = parseJson(bytes);
  requireValid(object(value) && Object.keys(value).every(key => rootKeys.includes(key)) &&
    value.schemaVersion === 1 && (value.provider === 'linux-perf-event' || value.provider === 'unsupported') &&
    value.providerImplemented === (value.provider === 'linux-perf-event') &&
    value.metricMask === (value.providerImplemented ? 3 : 0) &&
    value.cycleUnit === 'cycles' && value.instructionUnit === 'instructions' && value.timeUnit === 'ns' &&
    value.scope === 'thread-user-region' && value.sampling === 'none' && value.scaling === 'raw-with-enabled-running' &&
    value.kernelExcluded === true && value.hypervisorExcluded === true && value.inherited === false &&
    typeof value.complete === 'boolean' && Array.isArray(value.records), 'Invalid counter metadata.');
  for (const field of ['recordBytes', 'attempted', 'recorded', 'unavailable', 'dropped']) {
    requireValid(Number.isSafeInteger(value[field]) && (value[field] as number) >= 0, 'Invalid counter count/budget.');
  }
  const counters = value as unknown as CounterMetadata;
  const { attempted, recorded, unavailable, dropped, recordBytes } = counters;
  requireValid(recordBytes === manifest.captureOptions.counterBytes && recordBytes >= 64 &&
    recordBytes <= 16 * 1024 * 1024 && recorded === counters.records.length &&
    recorded <= Math.floor(recordBytes / 64) && attempted >= recorded && dropped === attempted - recorded &&
    unavailable <= recorded && counters.complete === (attempted > 0 && dropped === 0 && unavailable === 0) &&
    counters.availability === (counters.complete ? 'available' : !attempted ? 'unsupported-scope' :
      recorded === unavailable ? 'unavailable' : 'partial'), 'Inconsistent counter completeness/availability.');
  const references = new Map<number, CounterSample>();
  let failures = 0;
  for (const record of counters.records) {
    requireValid(object(record) && Object.keys(record).every(key => recordKeys.includes(key)) &&
      Number.isSafeInteger(record.eventId) && record.eventId > 0 && !references.has(record.eventId) &&
      Number.isSafeInteger(record.codeVersionId) && record.codeVersionId > 0 &&
      record.codeVersionId <= shaders.codeVersions.length && statuses.has(record.status) &&
      Number.isSafeInteger(record.nativeError) && record.nativeError >= 0 && record.nativeError <= 0x7fffffff,
    'Invalid counter record/status/reference.');
    const stage = shaders.codeVersions[record.codeVersionId - 1].stage;
    requireValid(record.region === 'pixel-worker' ? stage === 'pixel' :
      (record.region === 'compute-worker' || record.region === 'compute-direct') && stage === 'compute',
    'Counter region does not match code version.');
    const cycles = unsignedValue(record.cycles);
    const instructions = unsignedValue(record.instructions);
    const enabled = unsignedValue(record.timeEnabledNs);
    const running = unsignedValue(record.timeRunningNs);
    if (record.status === 'available') {
      requireValid(counters.provider === 'linux-perf-event' && record.nativeError === 0 &&
        enabled > 0n && running > 0n && running <= enabled, 'Invalid available counter timing/provider.');
    } else {
      ++failures;
      requireValid(cycles === 0n && instructions === 0n && enabled === 0n && running === 0n,
        'Unavailable counters cannot expose successful numeric measurements.');
    }
    references.set(record.eventId, record);
  }
  requireValid(failures === unavailable, 'Counter unavailable count disagrees with records.');
  for (const event of trace.traceEvents) {
    const record = references.get(Number(event.args.id));
    if (!record) continue;
    const name = record.region === 'compute-worker' ? 'Compute worker' :
      record.region === 'pixel-worker' ? 'Raster worker' : 'Compute direct';
    requireValid(event.ph === 'X' && event.name === name && event.args.codeVersionId === record.codeVersionId,
      'Counter record disagrees with exact recorded region/code version.');
    references.delete(record.eventId);
  }
  requireValid(manifest.droppedEvents > 0 || !references.size, 'Counter record references a missing span.');
  return counters;
}
