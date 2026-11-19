import { requireValid, type Profile } from './profile';
import type { ThreadGroup } from './presentation';

interface FrameEvent {
  type: 'O' | 'C';
  at: number;
  frame: number;
}

interface EventedProfile {
  type: 'evented';
  name: string;
  unit: 'microseconds';
  startValue: number;
  endValue: number;
  events: FrameEvent[];
}

export interface SpeedscopeFile {
  $schema: 'https://www.speedscope.app/file-format-schema.json';
  name: string;
  exporter: string;
  shared: { frames: { name: string }[] };
  profiles: EventedProfile[];
}

export function speedscopeProfile(profile: Profile, groups: ThreadGroup[], filtered: boolean): SpeedscopeFile {
  const capture = profile.manifest.complete && !profile.manifest.droppedEvents
    ? 'Complete capture' : `INCOMPLETE capture (${profile.manifest.droppedEvents} dropped events)`;
  const result: SpeedscopeFile = {
    $schema: 'https://www.speedscope.app/file-format-schema.json',
    name: `${capture}${filtered ? ' - filtered' : ''}; inclusive wall time, not CPU samples`,
    exporter: 'nacelle-view',
    shared: { frames: [] },
    profiles: [],
  };
  const frames = new Map<string, number>();
  let startValue = Infinity;
  let endValue = -Infinity;
  for (const group of groups) {
    for (const span of group.spans) {
      startValue = Math.min(startValue, span.ts);
      endValue = Math.max(endValue, span.ts + span.dur);
    }
  }
  requireValid(Number.isFinite(startValue), 'No timing spans match the current filters.');
  for (const group of groups) {
    if (!group.spans.length) continue;
    const name = `Process ${group.pid} / Thread ${group.tid}${group.name ? ` - ${group.name}` : ''}`;
    const events: FrameEvent[] = [];
    const stack: { end: number; frame: number }[] = [];
    const close = () => {
      const top = stack.pop()!;
      events.push({ type: 'C', at: top.end, frame: top.frame });
    };
    const spans = group.spans.slice().sort((a, b) => a.ts - b.ts || b.dur - a.dur);
    for (const span of spans) {
      while (stack.length && stack[stack.length - 1].end <= span.ts) close();
      const end = span.ts + span.dur;
      requireValid(!stack.length || end <= stack[stack.length - 1].end,
        `Speedscope cannot represent crossing overlaps in ${name}. Use the original Chrome trace export instead.`);
      const details = [span.cat];
      for (const key of ['stage', 'pso', 'simdWidth', 'tier', 'codeVersionId']) {
        const value = span.args[key];
        if (typeof value === 'string' || typeof value === 'number') details.push(`${key}=${value}`);
      }
      const key = JSON.stringify([span.name, ...details]);
      let frame = frames.get(key);
      if (frame === undefined) {
        frame = result.shared.frames.length;
        result.shared.frames.push({ name: `${span.name} [${details.join('; ')}]` });
        frames.set(key, frame);
      }
      events.push({ type: 'O', at: span.ts, frame });
      stack.push({ end, frame });
      if (!span.dur) close();
    }
    while (stack.length) close();
    result.profiles.push({
      type: 'evented', name: `${capture}${filtered ? ' - filtered' : ''} / ${name}`,
      unit: 'microseconds', startValue, endValue, events,
    });
  }
  return result;
}
