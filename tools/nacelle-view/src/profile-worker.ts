import { FileProfileDataSource } from './file-source';
import { LIMITS, loadProfile, requireValid, type Profile } from './profile';

export type DecodeReply = { profile: Profile } | { error: string };

globalThis.onmessage = async (event: MessageEvent<Blob>) => {
  try {
    requireValid(event.data instanceof Blob && event.data.size <= LIMITS.archive,
      'Archive exceeds 256 MiB limit or is not a file.');
    const source = FileProfileDataSource.fromBuffer(await event.data.arrayBuffer());
    const profile = await loadProfile(source);
    globalThis.postMessage({ profile } satisfies DecodeReply, { transfer: [profile.traceBytes] });
  } catch (error) {
    globalThis.postMessage({
      error: error instanceof Error ? error.message : 'Unable to decode profile.',
    } satisfies DecodeReply);
  } finally {
    globalThis.close();
  }
};
