import { isRecord } from '../../nacelle-view/src/host-protocol.ts';

export function nativeDebugSupported(platform: string, arch: string): boolean {
  return platform === 'linux' && arch === 'x64';
}

export function checkDebugResult(value: unknown): void {
  if (!isRecord(value) || typeof value.result !== 'string') throw new Error('Debugger returned no inspection result.');
  if (/^(?:fatal )?error:/m.test(value.result)) throw new Error(`LLDB command failed: ${value.result}`);
}

export function debugEnvironment(value: unknown): string[] {
  if (value === undefined) return [];
  if (!isRecord(value) || Object.keys(value).length > 128) throw new Error('env must be a bounded dictionary of strings.');
  return Object.entries(value).map(([name, text]) => {
    if (!name || /[=\0]/.test(name) || name.length > 1024 || typeof text !== 'string' ||
        text.includes('\0') || text.length > 16384) throw new Error('Invalid debugger environment entry.');
    return `${name}=${text}`;
  });
}

export function debugInteger(value: unknown, maximum = 0xffffffff): number {
  if (!isRecord(value) || typeof value.result !== 'string' ||
      !/^(?:[0-9]+|0x[0-9a-f]+)$/i.test(value.result.trim())) throw new Error('Debugger value is unavailable or not an integer.');
  const number = Number(value.result.trim());
  if (!Number.isSafeInteger(number) || number < 0 || number > maximum) throw new Error('Debugger value is outside supported bounds.');
  return number;
}

export function commandList(value: unknown, name: string): string[] {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > 128 ||
      !value.every(item => typeof item === 'string' && item.length <= 16384)) {
    throw new Error(`${name} must be a bounded array of command strings.`);
  }
  return value;
}
