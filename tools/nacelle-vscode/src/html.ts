export function escapeAttribute(value: string): string {
  return value.replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;',
    '"': '&quot;', "'": '&#39;' })[char]!);
}

export function resourcePath(value: string): string {
  const path = value.replace(/^\.\//, '');
  if (!path || path.startsWith('/') || path.includes('\\') ||
      path.split('/').some(part => part === '..' || part === '.') ||
      /[:?#]/.test(path)) throw new Error('Unsafe packaged asset path.');
  return path;
}

export function rewriteAssets(html: string, uri: (path: string) => string, nonce: string): string {
  return html.replace(/<(script|link)\b[^>]*>/g, tag => {
    const rewritten = tag.replace(/\b(src|href)="([^"]+)"/g,
      (_match, attribute: string, path: string) => `${attribute}="${escapeAttribute(uri(resourcePath(path)))}"`);
    return rewritten.startsWith('<script') ? rewritten.replace('<script', `<script nonce="${nonce}"`) : rewritten;
  }).replace(/<style>/g, `<style nonce="${nonce}">`);
}

export function configurationJson(value: unknown): string {
  return JSON.stringify(value).replace(/</g, '\\u003c').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
}
