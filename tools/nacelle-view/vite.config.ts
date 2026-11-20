import { cp, mkdir, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, type Plugin } from 'vite';

const SPEEDSCOPE_ALLOWED_FILE = /^(index\.html|release\.txt|source-code-pro\.LICENSE\.md|speedscope-[A-Za-z0-9_-]+\.(js|css)|SourceCodePro-[A-Za-z0-9_.-]+\.woff2|jfrview_bg-[A-Za-z0-9_-]+\.wasm|favicon-[A-Za-z0-9_-]+\.(ico|png|js))$/;

function preloadedProfilesCspPlugin(enabled: boolean): Plugin {
  return {
    name: 'nacelle-preloaded-profiles-csp',
    transformIndexHtml(html) {
      if (!enabled) return html;
      return html.replace("connect-src 'none'", "connect-src 'self'");
    },
  };
}

function bundleSpeedscopePlugin(): Plugin {
  const rootDir = fileURLToPath(new URL('.', import.meta.url));
  const releaseDir = resolve(rootDir, 'node_modules/speedscope/dist/release');
  let outDir = resolve(rootDir, 'dist');

  return {
    name: 'nacelle-bundle-speedscope',
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    async closeBundle() {
      const target = join(outDir, 'speedscope');
      await mkdir(target, { recursive: true });
      const entries = await readdir(releaseDir);
      for (const name of entries) {
        if (!SPEEDSCOPE_ALLOWED_FILE.test(name)) continue;
        await cp(join(releaseDir, name), join(target, name));
      }
    },
  };
}

export default defineConfig(() => {
  const preloadedEnabled = process.env.VITE_ENABLE_PRELOADED_PROFILES === '1';
  return {
    base: './',
    plugins: [tailwindcss(), preloadedProfilesCspPlugin(preloadedEnabled), bundleSpeedscopePlugin()],
    build: { target: 'es2022' },
  };
});
