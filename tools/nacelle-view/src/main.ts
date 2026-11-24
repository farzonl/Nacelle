import './style.css';
import { createSpeedscopeTimelineView } from './speedscope-timeline';
import { mountViewer } from './ui';

if (typeof window.acquireVsCodeApi === 'function') {
  void import('./vscode-host').then(host => host.mountVSCodeViewer()).catch(error => {
    const status = document.querySelector<HTMLElement>('#status')!;
    status.dataset.state = 'invalid';
    status.textContent = `VS Code viewer failed: ${error instanceof Error ? error.message : String(error)}`;
  });
} else if (window.__NACELLE_DESKTOP_HOST__) {
  void import('./desktop-host').then(host => host.mountDesktopViewer()).catch(error => {
    const status = document.querySelector<HTMLElement>('#status')!;
    status.dataset.state = 'invalid';
    status.textContent = `Desktop viewer failed: ${error instanceof Error ? error.message : String(error)}`;
  });
} else {
  mountViewer(document, undefined, {
    preloadedEnabled: import.meta.env.VITE_ENABLE_PRELOADED_PROFILES === '1',
    timelineView: createSpeedscopeTimelineView(),
  });
}
