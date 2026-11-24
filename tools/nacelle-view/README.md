# Nacelle profile viewer

Static, local-only viewer for `.nacelle-profile` archives. Select a file to see
capture completeness, build/options metadata and timing spans grouped by process
and thread, with 250 rows added at a time. Export preserves the decoded Chrome
trace JSON (timestamps/durations in **microseconds**, despite manifest clock
timestamps being specified in nanoseconds). No replay or custom flamegraph.
Empty shader metadata and empty traces are supported.

The optional [VS Code integration](../nacelle-vscode/README.md) embeds this same
frontend in a read-only profile editor with explicit export permissions.

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

## Shader artifact panes

Opt-in version-2 captures expose five pinned, locally bundled CodeMirror
read-only panes: HLSL, DXIL, MLIR, optimized LLVM and native assembly. The
pipeline/stage/version/Tier/cache/SIMD selector resolves the **captured** version,
not the latest promoted entry. Missing panes explicitly say unavailable.
Clicking an HLSL/DXIL line synchronizes only genuinely retained source mappings;
optimized/unmapped panes stay unchanged. Lane data is unavailable. Source text
is rendered as text, never HTML or executable code; editor styles use a fixed
CSP nonce. No source lookup, uploads, external assets or telemetry are added.

See [collection and privacy details](../nacelle-dbg/README.md#opt-in-shader-text-and-source-retention)
for the separate default-OFF text/source flags, supported DXC debug options,
bounded retention and exact artifact identities. Legacy captures remain
supported and do not show the artifact section. Worker decoding validates
declared checksums, source consent, owner references and mapping/text budgets
before presenting any panes. Editors are destroyed on profile replacement.

## Typed spatial observations

Version-3 captures expose validated `Profile.spatial` through the same
`ProfileDataSource` and offline decoding worker. The summary reports recorded,
skipped and dropped spatial observations separately from event drops. Either
kind of loss makes capture status incomplete. Shader artifacts can coexist
with spatial data; versions 1/2 remain supported.

The payload describes host entry-call nanoseconds per raster tile
or threadgroup and submitted pixel coverage/quad-companion/padding counts,
with exact operation/code-version identities. It does not contain per-pixel
timings, instruction divergence, source-lane values or replay resources.
Compute lane layout and dispatch-wide per-group timing can be explicitly
unavailable. Sparse absent cells must never become numeric zero; sampling,
no-submission and lost-coordinate ambiguity is preserved. In incomplete
operations all retained cells are conservatively partial.

See [spatial configuration, semantics, bounds and measured costs](../nacelle-dbg/README.md#opt-in-spatial-and-submitted-lane-capture).
The default-OFF spatial flag grants no source/resource retention consent.
Coordinates and workload counts remain private capture data. Decoding validates
the declared checksum/size, exact integer ranges, dimensions, selection, owners,
event references, per-slot counts and separate completeness without allocating
a dense image from dimensions. Parsed/worker/UI objects consume memory beyond
the record pool and archive limits.

### Heatmaps and exact coordinate/lane inspection

Choose an **operation** to keep stage/PSO/code-version attribution separate.
Choose host wall-time **ns**, entry calls, submitted invocations, or pixel
covered/quad-companion/padding submissions. Pixel lane selection chooses all
slots or one captured SIMD slot (4/8/16); slot packing is quad-major with local
XY `(slot % 2, floor(slot / 2) % 2)`. These are repeated entry submissions,
not source-lane values, unique pixels or dynamic branch activity. Compute lane
metrics are explicitly unavailable rather than empty numerical heatmaps.

Choose **XY/XZ/YZ** and an integer slice for 3D groups. Displayed axes increase
right/down; pixel XY preserves global top-left-Y-down orientation and actual
render-target extent, including clipped edge tiles. The bounded canvas has
at most 1024-by-1024 backing pixels, with CSS preserving the true extent aspect
ratio. It draws sparse captured cells only, never allocates an image from
untrusted dimensions. Subpixel cells can overlap: color is a projection, not
an averaged or interpolated value. Use **exact XYZ inspection** for precise data
in highly downscaled/extreme-aspect views. Click a visible cell or enter integer
cell coordinates directly; pixel coordinates here identify 32-by-32 tiles,
not individual framebuffer pixels.

Linear and **log1p** scales share a documented blue-to-orange legend.
Minimum/maximum controls clip colors only, not data; blank maximum uses the
visible slice's maximum (at least minimum + 1). Invalid ranges/slices/coordinates
produce explicit errors. Numeric zero is observed data and has a real metric
value; gray means missing/unsampled/unavailable, never zero. Exact inspection
distinguishes outside-ROI/stride **unsampled** from selected-but-**missing**
submissions. A lost coordinate cannot be identified: **dropped-or-missing**
stays ambiguous, all retained lossy-operation cells are **partial** with magenta
outlines, and incomplete captures warn against reliable interpretation.

Inspection includes exact pipeline/stage/PSO/code ID/version/Tier/cache/SIMD,
extent and captured span ID/timing/thread when available. **Show captured span /
artifact version** filters the span table/timeline to its PSO/stage/thread/time
window and selects the exact immutable artifact version, not the newest tier.
Intersecting spans may include parents or overlaps; the inspection's span ID
is authoritative. Missing events/artifacts stay unavailable; source mappings
and instruction-lane values are not invented.

Preparation, indexing and painting yield to animation frames every 1,000 cells;
replacement cancels stale scheduled work and releases the previous coordinate
index. This adds one bounded sparse index for the selected operation and a
canvas (up to 4 MiB pixel storage), beyond decoded objects and existing limits.
Operation selectors are bounded by the 4,096-operation schema limit. No new
libraries, runtime hooks, schema changes, uploads, telemetry or external assets
are introduced. CSP and offline worker decoding are unchanged.

The shared spatial corpus checks observed zeros, sampling, partial loss,
empty/unsupported operations, malformed attribution/geometry/counts and native
parser parity. Real static/shared captures exercise SIMD 4/8/16, odd edges,
derivatives/discard, 3D groups, overflow, cache hits and promotion. The offline
browser tests check native spatial decoding, actual canvas colors/orientation,
edge aspect ratio, lane selection, range errors, immutable artifact correlation,
3D slices and separate loss warnings without uploads or external requests.
A full-capacity 39,568-cell capture must render with main-thread heartbeats, bounded canvas
storage, exact final-coordinate values and stale-render cancellation.

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
incomplete/empty captures, malformed uploads, cancellation/recovery, checksummed
zstd input, and an 80,000-span capture larger than 8 MiB decoded. That test
requires a real worker and at least two main-thread heartbeats while loading.
The browser network
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
NACELLE_NATIVE_SPATIAL_PROFILE="$PWD/../../build/profile/tests/static-spatial-dense-simd8.nacelle-profile" \
  npm run test:browser
```

Serve `dist/` or the staged `viewer/` directory with a local static HTTP server;
ES modules are not reliably supported via `file://`. Once served, no network
access is required. Assets are bundled locally, with no CDN, uploads, telemetry,
or persistent browser storage. The production Pages link is `viewer/`.
Exported traces may contain sensitive application data; keep them private.

All three existing Pages publishers build/test and stage this viewer before
their existing Pages artifact upload. Staging copies only `dist/index.html` and
allowlisted hashed viewer, worker and optional editor-host JS/CSS assets, plus
bundled Speedscope files; not captures, examples, source, or arbitrary build output.
Do not put production profiles in `pages/` or any published artifact directory.
There is no additional deployment workflow.

## Data-source contract and format

`src/profile.ts` exports the transport-independent `ProfileDataSource` interface.
`readManifest()` returns a validated supported-version manifest; `readEntry(path)` returns
a fresh `ArrayBuffer` of **decoded** bytes (including zstd entries). `close()` is
idempotent and subsequent reads fail. `loadProfile()` always closes its source,
including on failure. `FileProfileDataSource.open(file)` adapts browser files;
no server endpoint or URL input is used. Production local-file and explicitly
enabled preloaded-benchmark loads perform ZIP/zstd/JSON decoding and metadata
validation in a dedicated module worker using that same contract. Only one load
is active: selecting another file/benchmark or **Cancel loading** aborts the
previous fetch/worker and discards its result. Worker errors are explicit; there
is no synchronous decode fallback. The decoded trace buffer is transferred,
not cloned, to preserve exact export bytes.

Stored ZIP payloads are bounded views of the archive, not full extraction
copies. The zstd decoder allocates one declared-size output buffer without a
second decoded-buffer copy. Archive and aggregate decoded payload limits remain
256 MiB, the manifest limit is 1 MiB, timing spans are capped at 1,000,000,
and shader versions/selections share a 65,536-record cap. These are **payload
bounds, not a 256 MiB browser RSS guarantee**: JSON strings/objects, the decoder
window, structured-cloned metadata/event objects during worker handoff, grouping,
and timeline/export structures use additional memory. A maximum-size archive and
maximum-size decoded trace can coexist. Completed worker state is released;
the UI retains the parsed profile and original trace bytes until replacement.
Preloaded downloads are bounded while streaming, not only after full download.
Rendering and Speedscope conversion/export remain main-thread work.

### Compatibility, parser parity and fuzzing

Manifest v1 and shader metadata `[]`/v1/v2 remain supported. Unsupported schema
versions fail explicitly; required fields and reference/completeness/size rules
are not relaxed. Native v1 readers require exactly the three defined files and
reject unknown JSON fields. Browser readers tolerate unknown object fields and
bounded, manifest-declared extra payloads as opaque extensions; they do not
interpret them as sources, resources, or replay data. A new required capability
must use an approved versioned schema, not silently change v1 semantics.

The shared synthetic corpus in `test/profile-corpus.ts` covers local/directory
flags, names, methods, CRC and size conflicts, overlaps, unsafe paths, corrupt
payloads, dictionary/concatenated/truncated/oversized-window zstd frames,
checksummed frames, and manifest/trace/shader inconsistencies. Native and browser
accept/reject decisions on the same bytes are checked in all profiling CI jobs:

```sh
NACELLE_NATIVE_READER="$PWD/../../build/profile/tools/nacelle-dbg/nacelle-dbg" \
  npm test -- test/parser-parity.test.ts
NACELLE_FUZZ_ROUNDS=10000 npm test -- test/parser-parity.test.ts
```

`fuzzProfileBytes()` is the reusable browser parser entry; mutation tests reject
unexpected bounds exceptions and report rejected inputs explicitly. Export
only these synthetic seeds for the native libFuzzer entry:

```sh
mkdir -p ../../build/profile-fuzz-corpus
NACELLE_FUZZ_CORPUS="$PWD/../../build/profile-fuzz-corpus" \
  npm test -- test/parser-parity.test.ts
```

See the [native fuzz build instructions](../nacelle-dbg/README.md#profile-reader-fuzzing).
Finite mutation runs are regression evidence, not proof that all malformed
inputs are safe. CRC32/zstd checksums detect corruption, not authenticity.
The pinned `xxhash-wasm` dependency verifies zstd's optional checksum locally;
CSP permits local module workers and WebAssembly compilation, not JavaScript
evaluation, remote scripts, or uploads.

### Privacy boundaries and retention

Treat archives and all contained labels as untrusted. Readers never extract
archive paths to disk, and presentation uses text rather than untrusted HTML.
Safe entry names do not sanitize arbitrary text inside a trace.

| Boundary | Current policy |
| --- | --- |
| Sources/resources | Neither is embedded by current producers. Any future embedding needs separate explicit opt-ins, size budgets, sanitized relative references, and schema/design approval. Debug metadata must never authorize arbitrary host-file reads. Hashes do not anonymize content. |
| Imports/exports | File selection and downloads are explicit and local. Chrome export preserves original trace bytes, including potentially sensitive application strings. Filtering is not redaction. Keep exports private. |
| Network/remote access | No upload, telemetry, CDN, remote-capture endpoint, or credential handoff. Default CSP disallows connections. Explicit benchmark builds may fetch only same-origin synthetic profiles; this is not permission to upload user captures. Future remote transport needs separate authentication/authorization and threat review. |
| Retention | Capture bytes and decoded objects remain in memory until replacement/cancellation or page close; no capture is put in localStorage, IndexedDB, or a service worker. Theme preference alone uses localStorage. Downloads and original files remain the user's responsibility. Worker termination/garbage collection is not a secure-memory-wipe guarantee. |
| Publication | Stage only viewer assets and explicitly generated benchmark fixtures. Never stage uploaded files, private sources/resources, production captures, fuzz inputs from applications, or downloaded traces. |

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
per-stage CPU time. Capture completeness must agree with drops; the timing-span
count must agree with recorded events (`M` metadata is not a recorded timing
event). Unsupported shader versions and inconsistent references are explicit
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
