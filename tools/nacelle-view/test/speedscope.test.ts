import { describe, expect, it } from 'vitest';
import { FileProfileDataSource } from '../src/file-source';
import { loadProfile, type TraceEvent } from '../src/profile';
import { filterThreads, groupThreads } from '../src/presentation';
import { speedscopeProfile } from '../src/speedscope';
import { fixture, span, trace } from './fixtures';

async function capture(events: TraceEvent[], incomplete = false) {
  return loadProfile(FileProfileDataSource.fromBuffer(fixture({
    complete: !incomplete, droppedEvents: incomplete ? 7 : 0,
  }, trace(events))));
}

describe('offline timeline filtering', () => {
  it('combines process/thread, case-insensitive PSO, stage and inclusive microsecond bounds without clipping', () => {
    const kept = span({ ts: 10.25, dur: 20.5, args: { pso: 'ABC123', stage: 'compute' } });
    const groups = groupThreads(trace([
      kept, span({ ...kept, pid: 2 }), span({ ...kept, args: { pso: 'abc123', stage: 'pixel' } }),
      span({ ts: 31, args: kept.args }), span({ ts: 0, dur: 1, args: kept.args }),
    ]));
    const result = filterThreads(groups, {
      thread: '1:2', pso: ' abc ', stage: 'compute', start: 30.75, end: 30.75,
    });
    expect(result).toHaveLength(1);
    expect(result[0].spans).toEqual([kept]);
    expect(result[0].spans[0].dur).toBe(20.5);
    expect(groups[0].spans).toHaveLength(4);
  });

  it('rejects invalid time windows and handles missing PSOs and empty results explicitly', () => {
    const groups = groupThreads(trace([span()]));
    expect(filterThreads(groups, { pso: 'absent' })).toEqual([]);
    expect(filterThreads(groups, { stage: 'compute' })).toEqual([]);
    expect(() => filterThreads(groups, { start: 2, end: 1 })).toThrow('Start time');
    expect(() => filterThreads(groups, { end: Infinity })).toThrow('finite');
  });
});

describe('Speedscope evented export', () => {
  it('preserves nesting, gaps, zero-duration events, equal-start containment and fractional microseconds', async () => {
    const profile = await capture([
      span({ name: 'child', ts: 10.5, dur: 1.25 }),
      span({ name: 'parent', ts: 10.5, dur: 5.5 }),
      span({ name: 'point', ts: 12, dur: 0 }),
      span({ name: 'later', ts: 20.25, dur: 2 }),
      span({ name: 'other thread', tid: 3, ts: 11, dur: 10 }),
      span({ ph: 'M', name: 'thread_name', args: { name: '<worker>' } }),
    ]);
    const before = JSON.stringify(profile.trace);
    const result = speedscopeProfile(profile, groupThreads(profile.trace), false);
    expect(JSON.stringify(profile.trace)).toBe(before);
    expect(result.profiles).toHaveLength(2);
    expect(result.profiles[0].name).toContain('<worker>');
    expect(result.profiles.every(item => item.unit === 'microseconds'
      && item.startValue === 10.5 && item.endValue === 22.25)).toBe(true);
    expect(result.profiles[0].events.map(event => [event.type, event.at])).toEqual([
      ['O', 10.5], ['O', 10.5], ['C', 11.75], ['O', 12], ['C', 12],
      ['C', 16], ['O', 20.25], ['C', 22.25],
    ]);
  });

  it('marks incomplete and filtered data and keeps metadata-only/empty captures out of Speedscope', async () => {
    const profile = await capture([span({ args: { pso: 'abc', stage: 'compute' } })], true);
    const result = speedscopeProfile(profile, groupThreads(profile.trace), true);
    expect(result.name).toMatch(/INCOMPLETE.*7 dropped.*filtered.*not CPU samples/);
    expect(result.profiles[0].name).toContain('INCOMPLETE');
    expect(result.shared.frames[0].name).toContain('stage=compute; pso=abc');
    expect(() => speedscopeProfile(profile, [], true)).toThrow('No timing spans');
  });

  it('rejects crossing spans within a thread rather than rewriting durations or dropping spans', async () => {
    const profile = await capture([
      span({ name: 'first', ts: 0, dur: 10 }), span({ name: 'crossing', ts: 5, dur: 10 }),
    ]);
    const before = JSON.stringify(profile.trace);
    expect(() => speedscopeProfile(profile, groupThreads(profile.trace), false)).toThrow(/crossing overlaps.*Chrome/);
    expect(JSON.stringify(profile.trace)).toBe(before);
  });

  it('handles a large bounded trace without recursive stack traversal or one frame per event ID', async () => {
    const profile = await capture(Array.from({ length: 100_000 }, (_, i) =>
      span({ ts: i * 2, dur: 1, args: { id: i + 1, parent: 0, stage: 'compute', pso: 'abc' } })));
    const result = speedscopeProfile(profile, groupThreads(profile.trace), false);
    expect(result.shared.frames).toHaveLength(1);
    expect(result.profiles[0].events).toHaveLength(200_000);
    expect(result.profiles[0].events.at(-1)?.at).toBe(199_999);
  });
});
