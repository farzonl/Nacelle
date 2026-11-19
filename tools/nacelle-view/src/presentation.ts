import type { Profile, Trace, TraceEvent } from './profile';

export interface ThreadGroup {
  pid: number;
  tid: number;
  name: string;
  spans: TraceEvent[];
}

export function groupThreads(trace: Trace): ThreadGroup[] {
  const groups = new Map<string, ThreadGroup>();
  const names = new Map<string, string>();
  for (const event of trace.traceEvents) {
    const key = `${event.pid}:${event.tid}`;
    if (event.ph === 'M' && event.name === 'thread_name' && typeof event.args.name === 'string') {
      names.set(key, event.args.name);
    }
    if (event.ph !== 'X') continue;
    let group = groups.get(key);
    if (!group) {
      group = { pid: event.pid, tid: event.tid, name: '', spans: [] };
      groups.set(key, group);
    }
    group.spans.push(event);
  }
  for (const [key, group] of groups) {
    group.name = names.get(key) ?? '';
    group.spans.sort((a, b) => a.ts - b.ts);
  }
  return [...groups.values()].sort((a, b) => a.pid - b.pid || a.tid - b.tid);
}

export function captureStatus(profile: Profile): string {
  const { manifest } = profile;
  if (!manifest.complete) {
    return `Incomplete capture: partial data only. ${manifest.droppedEvents} dropped events.`;
  }
  if (manifest.droppedEvents) {
    return `Capture completed with data loss: ${manifest.droppedEvents} dropped events.`;
  }
  return 'Complete capture loaded.';
}

export function summaryText(profile: Profile): string {
  const { manifest, trace } = profile;
  return [
    `Recorded events: ${manifest.recordedEvents}`,
    `Dropped events: ${manifest.droppedEvents}`,
    `Trace events: ${trace.traceEvents.length}`,
    `Timing spans: ${trace.traceEvents.filter(event => event.ph === 'X').length}`,
    `Clock: ${manifest.clock}; manifest timestamp unit: ${manifest.timestampUnit}`,
    'Trace timestamp/duration unit: µs; display unit: ms',
    'Replayable: no',
    `Build: ${JSON.stringify(manifest.build, null, 2)}`,
    `Capture options: ${JSON.stringify(manifest.captureOptions, null, 2)}`,
  ].join('\n');
}
