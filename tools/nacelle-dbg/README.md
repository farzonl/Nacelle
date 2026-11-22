# Offline profiling

Instrumentation is opt-in. The normal runtime remains C++20; the private Glaze
serialization target uses C++23. Dependencies are pinned submodules:

```sh
git submodule update --init third_party/glaze third_party/zstd third_party/miniz third_party/CLI11
cmake -S . -B build/profile -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNACELLE_ENABLE_INSTRUMENTATION=ON \
  -DNACELLE_BUILD_PROFILE_TOOLS=ON -DNACELLE_BUILD_PROFILE_TESTS=ON
cmake --build build/profile --target nacelle_d3d12 bench-worker-nacelle nacelle-dbg \
  instrumentation_test profile_runtime_static_test profile_runtime_shared_test
ctest --test-dir build/profile -R '^(instrumentation_test|profile_runtime_.*_test)$' --output-on-failure
```

Use the repository-pinned LLVM/MLIR/Clang toolchain described in the main
[build instructions](../../README.md#building-from-source). Keep profiling and
benchmark builds separate. Instrumented runtime caches use `profile-v1/` beneath
the normal cache directory (including `NACELLE_CACHE_DIR`); OFF cache selection is
unchanged. Normal compiler-source build-hash invalidation still applies.

## Capture and inspect

The benchmark worker offers an explicit capture path in instrumented builds:

```sh
NACELLE_PROFILE_OUTPUT="$PWD/capture.nacelle-profile" \
  build/profile/bench-worker-nacelle \
  --single-bench bench_raymarch --single-backend nacelle-w8 \
  --shader-dir build/profile/tests/shaders --iters 20
build/profile/tools/nacelle-dbg/nacelle-dbg capture.nacelle-profile \
  --trace capture-trace.json
```

Output paths must not exist. The worker's ordinary JSON benchmark output is
unchanged; its measurements while capturing are **not** release benchmark data.
The CLI exits 0 for a complete capture, 2 for an incomplete capture, and 1 for an
inspection/export failure. Open the bundle in the [static viewer](../nacelle-view/README.md),
or load exported Chrome Trace JSON into Perfetto. Nothing uploads automatically.

Tools and recorder tests can also build without LLVM/MLIR or DXC:

```sh
cmake -S tools/nacelle-dbg -B build/profile-tools -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNACELLE_BUILD_PROFILE_TESTS=ON
cmake --build build/profile-tools
ctest --test-dir build/profile-tools --output-on-failure
```

This standalone build also needs the initialized DirectX-Headers submodule.

## CI coverage

The Linux, macOS, and Windows CI jobs retain their normal OFF build and test
run, checking OFF compile commands, build targets, and static-library symbols
for instrumentation leakage using the repository-built `llvm-nm` without
fetching profiling dependencies. A dedicated Linux (`ubuntu-latest`) profiling
job runs concurrently to build `build-profile/` with instrumentation enabled,
run the recorder and real static/shared capture tests, and run native-to-viewer
interoperability tests and the viewer build.

The integration tests use fresh caches and exercise compute/graphics output
checks and asynchronous Tier-2 capture. Captures remain temporary CI workspace
data, not Pages assets. The Linux profiling job also runs offline Chromium upload/render/download tests,
comparing Chrome trace downloads with the native CLI export. Only production
viewer assets are served from local disk; all other requests are rejected.
These jobs do not certify performance budgets or guarantee a separate ARM64
runner. Platform coverage is only verified
when the corresponding instrumented job actually passes.

## Application API

Include [instrumentation.h](../../include/nacelle/instrumentation.h) and link
one instrumented runtime, static or shared:

```cpp
NacelleProfileOptions options;
NacelleProfileCapture* capture = nullptr;
HRESULT status = NacelleBeginProfileCapture(&options, &capture);
if (SUCCEEDED(status)) {
    // Create pipelines and submit the application work to capture.
    // Finish submissions before ending the session.
    NacelleProfileSummary summary;
    status = NacelleEndProfileCapture(capture, "capture.nacelle-profile", &summary);
    NacelleDestroyProfileCapture(capture);
}
```

Check `status == S_OK` for a complete saved capture: `S_FALSE` explicitly reports
event loss. Errors are returned and logged to stderr. After a write failure the
stopped handle can retry a different output path. Destroying a session without
ending cancels it, waits for captured work, and writes nothing.

One session may record per loaded runtime. Begin/End/Destroy must be called
outside captured work; End/Destroy calls on a given handle must be serialized.
End stops admission of new roots, waits for already captured scopes and
asynchronous JIT descendants, and only then serializes. It is not a GPU fence
or an application submission barrier. Do not mix statically and dynamically
linked Nacelle instances in one capture.

The default event budget is 64 MiB, configurable downward. Recording uses
per-thread blocks from that bounded pool; stranded thread-local block capacity
can cause explicit drops before every slot is filled. No file I/O, JSON work,
or consumer callback executes in recording scopes. Serialization, compression,
thread/context bookkeeping, and decoded viewer objects use additional memory;
the event budget is not a total-process RSS limit.

Event IDs are reserved in per-thread blocks and need not be contiguous or
chronologically ordered. Exact recorded-event counts are finalized from completed
buffer slots after captured work drains, avoiding shared per-event count updates.

### Memory validation

The first [#212](https://github.com/farzonl/Nacelle/issues/212) diagnostic used
three fresh Linux x86_64 processes per synthetic workload (Clang 21 Release).
RSS was sampled after allocation, metadata/context creation, recording, drain,
serialization/write, and destruction. Peak figures below are process-lifetime
high-water marks sampled after serialization, not allocation attribution:

| Workload | Event budget | Recorded / dropped | Recording RSS | Save peak RSS | Serialize/write |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nearly full | 1 MiB | 9,349 / 0 | 4.89 MiB | 11.44 MiB | 7.92 ms |
| Nearly full | 16 MiB | 149,783 / 0 | 19.95 MiB | 122.51 MiB | 125.94 ms |
| Nearly full | 64 MiB | 599,173 / 0 | 67.95 MiB | 479.44 MiB | 503.92 ms |
| Overflow | 64 MiB | 599,186 / 4,096 | 67.95 MiB | 479.45 MiB | 499.19 ms |
| 512 short-lived threads | 1 MiB | 147 / 365 | 4.96 MiB | 5.59 MiB | 0.38 ms |
| Oversized metadata, rejected | 64 MiB | 599,173 / 0 | 68.92 MiB | 71.50 MiB | 1.79 ms |

These are medians, not memory or latency guarantees. Nearly-full workloads leave
13 event slots unused; the short-lived-thread case records one span per thread,
showing explicit loss from stranded blocks. Every saved bundle was checked for
format limits and event accounting, and a sample of each workload was read by
the native CLI. Rejected metadata created no bundle. These measurements predate
the additional per-event compute identity fields and compute metadata registry;
they are not memory measurements of version-tracked captures.

At 64 MiB, event allocation alone raised RSS from about 3.54 to 67.80 MiB.
Synthetic metadata and one retained context added approximately 12 KiB, below
reliable RSS attribution granularity. Metadata and 512 retained contexts used
about 56 KiB in the short-lived-thread case, excluding thread execution effects.
Workers were joined before drain, so drain latency does not characterize waiting
for live application work. Serialization remains non-streaming and its peak
memory is not separately bounded; metadata admission, allocation-failure tests,
and per-serialization-stage attribution remain open under #212.

## Format and current scope

A `.nacelle-profile` is a stored ZIP containing:

* `manifest.json`: schema version 1, capture options, completeness/event counts,
  producer revision/compiler/OS/architecture, and captured compiler hash,
  target triple, CPU/features when a pipeline is created or an existing compute,
  raster draw, or mesh pipeline executes during the capture. Execution-only
  captures do not require recompilation; empty captures leave these fields
  unavailable. Configuration is recorded once per session, including concurrent
  compilation/execution, rather than collected for every dispatch.
* `trace.json.zst`: dictionary-free zstd-compressed Chrome Trace JSON.
* `shaders.json`: `[]` when no compute entry was observed, otherwise the versioned
  compute metadata object described below. PSO hashes, stage, SIMD width,
  compilation tier, cache outcome, and parent IDs also live in trace arguments.

Manifest clock offsets are nanoseconds; Chrome Trace `ts`/`dur` are microseconds.
Trace IDs and thread IDs are capture-local, not addresses or OS thread IDs.
Overlapping spans are not additive CPU time. Tier 0 and cache `unknown` mean
unavailable, not Tier 1 or a miss. Compute dispatch/worker spans identify the
selected native entry version, including pre-existing and cached pipelines.
Graphics/mesh stage execution versions remain unavailable.

### Compute entry metadata

The manifest remains schema v1 with the same three required entries. Nonempty
`shaders.json` has independent `schemaVersion: 1` and a `codeVersions` array.
Every record requires `id`, `pipelineId`, `pso` (32 lowercase hex digits),
`stage` (`compute`), `simdWidth` (positive integer), `version`, `tier`, and `cache`.
Code IDs are consecutive, one-based array positions in first-observed order.
Pipeline IDs are capture-local, one-based pipeline-instance identities; two
instances with the same PSO hash are distinct. A pipeline/version pair is unique,
and its PSO hash and SIMD width cannot change. These are publication identities,
not shader-source or native-object content hashes.

An initially published entry has version 1: a freshly compiled entry is Tier 1
with cache `miss`, while a memory/disk cache hit is Tier 2 with cache `hit`.
Asynchronous promotion on the original pipeline publishes version 2, Tier 2,
cache `miss`. A capture may observe only the promoted version. IDs are assigned
only to entries actually selected for captured dispatches, not every compilation.
Compute span arguments add optional `pipelineId`/`codeVersionId` together; when
present, PSO/stage/SIMD/tier/cache must match the referenced record. Other spans
and legacy captures omit these IDs rather than implying an exact version.

Instrumented dispatches acquire one immutable entry/function/metadata snapshot.
Promotion publishes a separate record without changing snapshots held by
in-flight dispatches; workers inherit their dispatch's selected identity.
Both records live with the pipeline, and the existing JIT code lifetime is
unchanged. No generated-kernel hooks, additional code unloading, source embedding,
or address serialization are introduced. Instrumentation-OFF layouts and callable
selection remain unchanged.

At most 65,536 observed compute versions may be registered per capture. Exceeding
this limit reports a save error and produces no bundle, rather than silently
truncating metadata. Registry and serialization allocations are additional to the
event-buffer budget. Version tracking has not certified the performance targets.

Current native/browser readers accept legacy `[]` metadata and validate versioned
records and trace references. Unsupported shader schema versions are rejected;
missing required fields and inconsistent references are errors. Native readers
remain strict about unknown fields and extra entries; browser readers tolerate
additional object fields and manifest-declared payloads. Old native readers
explicitly reject new compute-version captures; the original browser reader can
display/export them but does not interpret shader metadata. No extra payloads or
manifest fields were added.

Captures include compile/import/lowering/optimization, draw/compute/mesh
dispatch, and coarse worker spans. They do not instrument generated kernels,
pixels, individual instructions, or every possible shader stage.
No sources, host file paths, resource contents, or replay state are embedded.
Nonzero capture flags are rejected: source embedding is not implemented yet.
Profiles are never labeled replayable. Heatmaps, counters, WARP comparison,
deterministic replay, and native/HLSL lane stepping are later work.

Readers reject malformed versions, paths, duplicates, corruption, and oversized
payloads. Native v1 requires exactly the three entries above; the browser can
accept additional manifest-declared payloads. Both cap archives and aggregate
decoded payloads at 256 MiB, manifests at 1 MiB, and trace events at one million.
Large offline profiles may occupy the browser briefly
while decoding.
The producer checks serialized metadata before constructing trace/compression
buffers and rechecks the final manifest after payload sizes are known. The limit
includes JSON escaping. Oversized metadata returns a save error without creating
a bundle; it is not silently discarded or truncated. This check does not bound
metadata allocations made earlier during capture or total serialization RSS.

Keep application captures private. Only viewer assets, never captures, are
staged by the Pages publishers. The accepted measurement targets are no OFF
regressions, at most 1% ON-idle median wall-time overhead and 5% active coarse
capture overhead on representative nontrivial workloads. These require
controlled measurements; they are not guarantees for every shader or host.

### Validation status

The initial implementation is experimental. Linux x86_64 validation covers
static/shared runtime capture with compute and graphics output checks, recorder
lifecycle/overflow tests, native bundle interoperability, and offline browser
upload/render/export. Windows and macOS have not been executed locally.
Static capture tests use the `Nacelle*` entry points; shared capture tests use
the DLL-exported `D3D12*` entry points, including on Windows, without expanding
the runtime export surface.
Compute-version tests additionally cover cold Tier 1, promotion on the original
pipeline, memory/disk cache hits, delayed selection across publication, repeated
captures, metadata limits and invalid references. Linux static/shared execution
has been checked at SIMD widths 4 and 8.

Performance acceptance remains open. A shared-host sample using Release builds
with identical compiler hashes, randomized sequential interleaving, 11 runs per
configuration, and 240 iterations per run produced these median wall times
(1024x1024, SIMD width 8):

| Workload | OFF | ON idle | ON active | Idle overhead | Active overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raymarch compute | 2.1066 ms | 2.0929 ms | 2.1056 ms | -0.65% | -0.05% |
| Textured filter pixel | 7.1846 ms | 7.1661 ms | 7.8094 ms | -0.26% | +8.70% |

Outputs matched across configurations. Active pixel capture exceeded the 5%
target; earlier samples also varied, so these results do not certify either
performance budget. Active captures added approximately 64 MiB to benchmark
peak RSS, consistent with the event allocation. That benchmark samples memory
before final bundle serialization and does not isolate serialization peak
memory. Instrumentation-OFF binaries were checked for absence of capture
symbols and support targets; this alone is not a universal no-regression proof.

The first recorder optimization for [#211](https://github.com/farzonl/Nacelle/issues/211)
batches ID allocation and moves recorded-event counting to capture drain. An
isolated test with eight threads, each recording 10,000 nested scope pairs,
reduced median recording time from 29.8003 ms to 15.1204 ms across ten interleaved
runs. Both versions recorded all 160,009 events. This diagnostic excludes capture
allocation, draining, and serialization; it is not the shader-overhead gate.

A separate matched-hash shader comparison used 12 balanced, interleaved rounds
per configuration, warm caches, and the same workload settings as above:

| Workload | OFF | ON idle | Original ON active | Batched ON active |
| --- | ---: | ---: | ---: | ---: |
| Raymarch compute | 2.1024 ms | 2.0996 ms | 2.1145 ms | 2.1081 ms |
| Textured filter pixel | 7.1973 ms | 7.1287 ms | 7.1564 ms | 7.1546 ms |

Outputs and recorded-event counts matched between active versions, with no
drops. End-to-end changes were within shared-host variability: paired pixel
overhead ranged from -9.0% to +16.9% for the original recorder. These samples do
not demonstrate a reliable shader speedup or certify the performance budgets;
further isolation and controlled-host measurements remain necessary.
