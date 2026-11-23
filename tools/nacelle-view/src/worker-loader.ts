import { LIMITS, requireValid, type Profile } from './profile';
import type { DecodeReply } from './profile-worker';

export function decodeProfileFile(file: Blob, signal: AbortSignal): Promise<Profile> {
  requireValid(file.size <= LIMITS.archive, 'Archive exceeds 256 MiB limit.');
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./profile-worker.ts', import.meta.url), { type: 'module' });
    const finish = () => {
      worker.terminate();
      signal.removeEventListener('abort', abort);
    };
    const abort = () => {
      finish();
      reject(signal.reason);
    };
    signal.addEventListener('abort', abort, { once: true });
    worker.onmessage = (event: MessageEvent<DecodeReply>) => {
      finish();
      if ('error' in event.data) reject(new Error(event.data.error));
      else resolve(event.data.profile);
    };
    worker.onerror = event => {
      event.preventDefault();
      finish();
      reject(new Error(event.message || 'Profile decoder worker failed.'));
    };
    worker.onmessageerror = () => {
      finish();
      reject(new Error('Unable to receive decoded profile.'));
    };
    try {
      worker.postMessage(file);
    } catch (error) {
      finish();
      reject(error);
    }
  });
}
