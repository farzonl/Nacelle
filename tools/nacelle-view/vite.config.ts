import tailwindcss from '@tailwindcss/vite';
import { defineConfig, type Plugin } from 'vite';

function preloadedProfilesCspPlugin(enabled: boolean): Plugin {
  return {
    name: 'nacelle-preloaded-profiles-csp',
    transformIndexHtml(html) {
      if (!enabled) return html;
      return html.replace("connect-src 'none'", "connect-src 'self'");
    },
  };
}

export default defineConfig(() => {
  const preloadedEnabled = process.env.VITE_ENABLE_PRELOADED_PROFILES === '1';
  return {
    base: './',
    plugins: [tailwindcss(), preloadedProfilesCspPlugin(preloadedEnabled)],
    build: { target: 'es2022' },
  };
});
