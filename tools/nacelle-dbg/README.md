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

## Format and current scope

A `.nacelle-profile` is a stored ZIP containing:

* `manifest.json`: schema version 1, capture options, completeness/event counts,
  producer revision/compiler/OS/architecture, and captured compiler hash,
  target triple, CPU/features when a pipeline is created during the capture.
* `trace.json.zst`: dictionary-free zstd-compressed Chrome Trace JSON.
* `shaders.json`: currently `[]`; PSO hashes, stage, SIMD width, compilation tier,
  cache outcome, and parent IDs live in trace arguments.

Manifest clock offsets are nanoseconds; Chrome Trace `ts`/`dur` are microseconds.
Trace IDs and thread IDs are capture-local, not addresses or OS thread IDs.
Overlapping spans are not additive CPU time. Tier 0 and cache `unknown` mean
unavailable, not Tier 1 or a miss; execution scopes do not yet identify the
exact promoted native entry version.

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
decoded payloads at 256 MiB; the browser additionally caps trace events at one
million. Large offline profiles may occupy the browser briefly while decoding.

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
