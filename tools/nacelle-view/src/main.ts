import './style.css';
import { createSpeedscopeTimelineView } from './speedscope-timeline';
import { mountViewer } from './ui';

mountViewer(document, undefined, {
  preloadedEnabled: import.meta.env.VITE_ENABLE_PRELOADED_PROFILES === '1',
  timelineView: createSpeedscopeTimelineView(),
});
