# Nacelle profile viewer

Static, local-only viewer for `.nacelle-profile` archives. Select a file to see
capture completeness, build/options metadata and timing spans grouped by process
and thread, with 250 rows added at a time. Export preserves the decoded Chrome
trace JSON (timestamps/durations in **microseconds**, despite manifest clock
timestamps being specified in nanoseconds). No replay or custom flamegraph.
Empty shader metadata and empty traces are supported.

## Local filters and Speedscope export

Apply process/thread, case-insensitive PSO-substring, stage and optional
microsecond time-window filters to the paginated tables. Bounds are inclusive:
any intersecting span is selected at its full original duration, not clipped.
Filtering can hide parents. The separate filter count does not replace capture
completeness or dropped-event warnings. **Export Chrome trace JSON** always
preserves the entire original decoded trace, irrespective of active filters.

**Export filtered Speedscope profile** produces a local download using the
[Speedscope evented format](https://github.com/jlfwong/speedscope/blob/v1.25.0/src/lib/file-format-spec.ts).
For offline interactive navigation, install a fixed release while online, then
open the downloaded profile locally:

```sh
npm install --global speedscope@1.25.0
speedscope ./nacelle-speedscope.json
```

The CLI starts its own local viewer. Speedscope's Time Order, Left Heavy and
Sandwich views provide established zoom/search/thread-navigation/flamegraph
tooling; no renderer is implemented here. No viewer dependency or remote handoff
is added to Nacelle, and its CSP and asset-only deployment remain unchanged.
Do not upload private captures to a hosted viewer.

Each recorded process/thread becomes an evented profile in **microseconds**,
with shared selected-trace time bounds, original timestamps, gaps and durations.
Frame labels include available category/PSO/stage/SIMD/tier/code-version data.
Nesting represents temporal containment on that recorded thread, **not HLSL call
stacks or CPU samples**. Asynchronous parent IDs do not create cross-thread call
stacks. Inclusive nested durations and different thread profiles must not be
summed as CPU time. Incomplete/filtered status is included in exported names.
Zero-duration spans are preserved. Empty selections disable this export.

Speedscope requires properly nested per-thread intervals: crossing overlaps
produce an explicit export error, never dropped spans, clipped durations or
fabricated nesting. Tables and unchanged Chrome export remain available.
Exports above 256 MiB are rejected. Conversion/serialization is synchronous;
large bounded captures may briefly occupy the browser.
Validation covers nested/equal-start/zero-duration spans, crossing rejection,
incomplete captures, filter/export isolation, and a 100,000-span conversion.
An actual native capture was exported and opened in locally served Speedscope
1.25.0 with the browser offline, without external requests or viewer storage.

This is the export-first increment for #216, not an embedded timeline UI.
[Perfetto's embedding API](https://perfetto.dev/docs/visualization/deep-linking-to-perfetto-ui)
was also evaluated: it supports richer Chrome-trace exploration but requires a
separately self-hosted application or an explicitly consented external handoff.
Speedscope's documented offline CLI and evented format were selected for this
bounded increment. Bundled offline UI integration remains future work; neither
hosted Perfetto nor hosted Speedscope is contacted by this viewer.

## Build and test

Requires Node.js 22 and npm:

```sh
cd tools/nacelle-view
npm ci
npm test
npm run build
npm run stage -- ../../_site
```

The dedicated Linux profiling CI job also runs production assets in offline
Chromium with Playwright. It uploads the real native static runtime-test capture,
checks span rendering and filtering, and compares downloaded Chrome trace bytes
against the native CLI export. It exercises filtered Speedscope downloads,
incomplete/empty captures, malformed uploads, and recovery. The browser network
is disabled; the harness serves only built viewer assets from local disk and
rejects any other request, including profile uploads or external services.
Native captures and downloads are not published.

To reproduce with an existing native capture (the trace output must not exist):

```sh
../../build/profile/tools/nacelle-dbg/nacelle-dbg \
  ../../build/profile/tests/static.nacelle-profile \
  --trace ../../build/profile/tests/static-trace.json
npx playwright install chromium
npm run build
NACELLE_NATIVE_PROFILE="$PWD/../../build/profile/tests/static.nacelle-profile" \
NACELLE_NATIVE_TRACE="$PWD/../../build/profile/tests/static-trace.json" \
  npm run test:browser
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
Shader JSON accepts legacy `[]`, or an independently versioned
`{ "schemaVersion": 1, "codeVersions": [...] }` object, or schema v2 with graphics
entry roles, optional shader/native-object content digests and composite
`codeSelections`. The loader exposes typed metadata as `Profile.shaders` and
validates pipeline/code IDs, PSO/stage/SIMD/tier/cache fields, unique
pipeline/entry-role/version identities, and exact/composite trace references.
Draw/extended-geometry selections do not imply one latest tier or isolated
per-stage CPU time. Unsupported shader versions and inconsistent references are explicit
errors. Unknown object fields remain forward-compatible. Metadata is available
through the data-source boundary; shader/code-view panes are not implemented.
See the [native format guide](../nacelle-dbg/README.md#compute-entry-metadata) for
identity, promotion, privacy and old-reader compatibility rules.

Limits (reject rather than silently truncate): 256 MiB archive, 16 ZIP entries,
1 MiB manifest, 256 MiB decoded per entry / zstd window, 256 MiB decoded total,
1,000,000 trace events, 65,536 code-version/selection records combined, and eight
entry references per composite selection. Paths must be relative ASCII components without `.` or
`..`, backslashes, URLs, or empty components. No files are extracted to disk.
Large profiles are decoded synchronously and may briefly occupy the browser;
these conservative limits are intended for initial offline inspection.
