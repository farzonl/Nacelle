import './style.css';
import { mountViewer } from './ui';

mountViewer(document, undefined, {
  preloadedEnabled: import.meta.env.VITE_ENABLE_PRELOADED_PROFILES === '1',
});
