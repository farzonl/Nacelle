# Nacelle profile viewer

Static, local-only viewer for `.nacelle-profile` archives. Select a file to see
capture completeness, build/options metadata and timing spans grouped by process
and thread, with 250 rows added at a time. Export preserves the decoded Chrome
trace JSON (timestamps/durations in **microseconds**, despite manifest clock
timestamps being specified in nanoseconds). No replay or custom flamegraph.
Empty shader metadata and empty traces are supported.

## Build and test

Requires Node.js 22 and npm:

```sh
cd tools/nacelle-view
npm ci
npm test
npm run build
npm run stage -- ../../_site
```

Serve `dist/` or the staged `viewer/` directory with a local static HTTP server;
ES modules are not reliably supported via `file://`. Once served, no network
access is required. Assets are bundled locally, with no CDN, uploads, telemetry,
or persistent browser storage. The production Pages link is `viewer/`.
Exported traces may contain sensitive application data; keep them private.

All three existing Pages publishers build/test and stage this viewer before
their existing Pages artifact upload. Staging copies only `dist/index.html` and
bundled JS/CSS assets, not captures, examples, source, or arbitrary build output.
Do not put production profiles in `pages/` or any published artifact directory.
There is no additional deployment workflow.

## Data-source contract and format

`src/profile.ts` exports the transport-independent `ProfileDataSource` interface.
`readManifest()` returns a validated schema-v1 manifest; `readEntry(path)` returns
a fresh `ArrayBuffer` of **decoded** bytes (including zstd entries). `close()` is
idempotent and subsequent reads fail. `loadProfile()` always closes its source,
including on failure. `FileProfileDataSource.open(file)` adapts browser files;
no server endpoint or URL input is used.

Accepted native format: a single-disk, non-ZIP64 **stored ZIP**, with
`manifest.json`, `trace.json.zst`, and `shaders.json`. The manifest must declare
all payloads with `json` or `zstd` encoding and decoded `uncompressedSize`.
The trace is a single dictionary-free zstd frame containing Chrome trace JSON
(`X` timing spans and `M` metadata, `displayTimeUnit: "ms"`). ZIP directory/local
headers, paths, duplicates, overlaps, sizes, CRC32, manifest schema, zstd frame
bounds and decoded sizes, and trace fields are validated. Unsupported archives
and invalid/truncated captures show an explicit error; `complete: false` remains
viewable but is prominently marked incomplete. Dropped events warn of data loss.
Counters are reported as recorded, not assumed equal to Chrome event counts.
Shader JSON must be an object or array; its evolving internal schema is not
interpreted yet.

Limits (reject rather than silently truncate): 256 MiB archive, 16 ZIP entries,
1 MiB manifest, 256 MiB decoded per entry / zstd window, 256 MiB decoded total,
and 1,000,000 trace events. Paths must be relative ASCII components without `.` or
`..`, backslashes, URLs, or empty components. No files are extracted to disk.
Large profiles are decoded synchronously and may briefly occupy the browser;
these conservative limits are intended for initial offline inspection.
