# Offline profiling and local shader debugging

## HLSL source and lane inspection (Linux x86_64, LLDB)

Source debugging is separately opt-in, requires the native-debug build described
below, and does not change ordinary profiling or native-instruction debugging.
Include [source_debug.h](../../include/nacelle/source_debug.h), link one
instrumented runtime, and configure before any pipeline compilation/cache lookup:

```cpp
NacelleSourceDebugOptions options;
options.enabled = 1;
options.consent = 3; // Explicit embedded-source AND local-value consent.
HRESULT status = NacelleConfigureSourceDebug(&options);
```

Check exact HRESULTs. Configuration enables the existing pinned native-debug
policy: no asynchronous optimized replacement and no optimized memory/disk cache
reuse or writes. Source mode has its own compiler/environment fingerprint.
Compile the supplied compute/pixel shader with DXC `-Od -Zi -Qembed_debug`.
Stripped, optimized or incompatible inputs fail explicitly. Debug and executable
code must agree after removing debug-only information; execution settings,
resources/signatures and entry metadata come from the executable part, not
untrusted alternative debug metadata. Shader identity hashes the original entire
input. Source debug input is limited to 32 MiB.

After successful pipeline creation, enumerate `NacelleGetSourceDebugSite` starting
at ID 1 until `S_FALSE`. A site's `available` field is set only after successful
native materialization. Query `NacelleGetSourceDebugSource` first for its size,
then copy its embedded bytes into a caller-owned buffer. No original source path
is opened, and nothing is automatically exported to disk.
`NacelleGetSourceDebugLocal` reports each site's supported value type or explicit
unavailability reason.

Before submitting work, while the host is quiescent:

1. Fill `NacelleSourceDebugSelection` with the exact compiled `shader_id` and
   global compute thread XYZ, or pixel XY/sample 0.
2. Call `NacelleBeginSourceDebug` and retain the returned session generation.
3. Call `NacelleSourceDebugBreakpoint(session, source_id, line)` for executable
   lines of that shader. Non-executable/unmapped lines fail explicitly.
4. Submit the original workload. The session observes only the first matching
   dispatch/draw of the first admitted submission. Later operations/readbacks
   are not substituted for a completed target. Concurrent submissions cancel
   observation with diagnostics but still execute normally.
5. After work returns, call `NacelleEndSourceDebug` to drain admitted work and
   inspect the summary. `S_FALSE` means cancelled, unavailable/inactive target,
   uncompleted source-step or rejected concurrency, **not debugger acceptance**.

Launch or attach LLDB using the native workflow below, then load the shipped
first-party command aliases:

```text
(lldb) command source tools/nacelle-dbg/source-debug.lldb
(lldb) run
(lldb) hlsl-location
(lldb) hlsl-lanes
(lldb) hlsl-values
(lldb) hlsl-code
(lldb) hlsl-arm-step
(lldb) continue
(lldb) hlsl-arm-continue
(lldb) continue
```

The source breakpoint resolves in the application API; LLDB's native breakpoint
on `nacelle_source_debug_stop` handles all-stop coordination. A snapshot contains
the exact source identity/line/column/discriminator/site and bounded inline
chain (leaf first, at most eight frames), native object/range/version,
coordinates, actual execution/real masks, local metadata and raw local bits for
all submitted lanes (up to 16). `active_mask & ~real_mask` distinguishes active
helper/companion lanes. Inactive lane values are invalid, not zero-valued locals.
Use `locals[i].kind/bits` to interpret `values[i][lane]`; float bits are not an
integer numerical value. Group XYZ/local thread coordinates can be derived
from global XYZ and `site.threads_per_group`; the SIMD lane index is the actual
packet slot, not a fictional independently executed HLSL thread.

The aliases **only write the stop's generation-tagged command mailbox**.
`continue` resumes the complete original program. Never call lock-taking runtime
query/control/end functions from LLDB while other host threads are all-stopped:
another frozen thread may own their mutex. Read the immutable snapshot directly.
The runtime checks session/stop generations and the mailbox action only after
LLDB resumes. A missing/detached debugger returns immediately from the rendezvous
and cancels future observations explicitly; there is no runtime parked-lane wait.

Source-step follows the next dynamic mapped source site for the same selected
invocation, including loop revisits, rather than native instruction count.
Site granularity is one source line per original basic block; different
control-flow paths/discriminators on one line can have distinct sites.
Source-step skips deeper inlined helper frames; explicit helper breakpoints still
apply. An overlapping primitive at the same pixel cannot become the stepped
fragment. Selection filters
observations only: all waves, barrier phases, shared-memory operations, quad
helpers, derivatives, divergence, discard and supported wave operations retain
the existing execution. A selected lane that completes/discards or never reaches
another site does not cause a different invocation to be stepped instead.

The initial local-value reader supports directly described scalar bool,
signed/unsigned integer and floating values up to 64 bits, including simple
typedef/const wrappers. Unsupported debug expressions, aggregates/vector
fragments, inlined locals, missing reaching definitions and out-of-scope
information are unavailable, never reconstructed. No arbitrary pointer
dereference, expression evaluation or resource-memory inspector is added.
This pinned workflow rejects optimized input. Call step-in/out, MSAA,
non-compute/pixel source selections and other platforms remain unsupported;
native instruction debugging is separate.

To cancel at a stop, use `hlsl-arm-cancel`, then `continue`. From a running
controlling host thread, use `NacelleControlSourceDebug(session, 0,
NACELLE_SOURCE_CANCEL)` and End. Cancellation disables observation, not original
shader execution or half a threadgroup. End waits for original work to drain;
calling End from an observation callback is rejected. While LLDB has the process
suspended, host cancellation/draining cannot progress until resume/detach.

Embedded source retention defaults to 8 MiB (cap 16 MiB), site/local metadata to
16 MiB (cap 32 MiB), and breakpoints to 4,096. There is one bounded snapshot of
at most 64 KiB, at most 64 local slots and 16 lanes. Admission failures are
explicit; live metadata is not silently evicted. Native object registration keeps
its separate budget; the event budget remains unchanged. These are retained-data
limits, **not total compiler/process RSS guarantees**. The opted-in compiler
does additional parsing, code validation and provenance work. No performance
certification for coarse capture or optimized-code debugging is implied.

The source-debug acceptance tests exercise real LLDB source stops, source-line
advancement, loop revisits, raw per-lane scalar values, native identities,
compute multi-wave barriers/wave operations and pixel derivatives/discard,
followed by complete output equality. Tests also cover absent debuggers, missing
targets, stripped/optimized input, budgets, cancellation/draining, stale IDs
and overlap. See `source_debug_*test` in the profiling test configuration.

For a reproducible **debug-only diagnostic**, run the static fixture with
`NACELLE_SIMD_WIDTH=4 NACELLE_TEST_SOURCE_COST=native
NACELLE_TEST_SOURCE_UNATTENDED=1`, then repeat with
`NACELLE_TEST_SOURCE_COST=source`, passing `unused source-compute`.
Both run the same unoptimized DXIL and check every output. The source run observes
one stop and cancels observation when no debugger acknowledges it; this excludes
interactive debugger pauses and is not the cost of a long stepping session.
Five fresh-process Linux x86_64 samples gave median first-PSO compilation
18.36/43.39 ms and submission 0.586/1.042 ms (pinned native/source debug).
The fixture's two PSO creations retained 40 sites, 105,840 bytes of site/local
payload and 1,618 embedded source bytes; the snapshot is 23,176 bytes.
Payload excludes registry/allocator slack and compiler/native object storage.
These small-host observations are not a benchmark certification, a total-memory
bound, or evidence that #211/#212 are complete.

## Optimized HLSL source and lane inspection

The separately versioned optimized configuration retains ordinary Tier-1
optimization and asynchronous Tier-2 O3 promotion. It does **not** make the
pinned configuration above accept optimized inputs. In a fresh instrumented
Linux x86_64 process, configure before pipeline compilation/cache admission:

```cpp
NacelleOptimizedSourceDebugOptions options;
options.enabled = 1;
options.consent = 3;
HRESULT status = NacelleConfigureOptimizedSourceDebug(&options);
```

Check the HRESULT. Do not additionally enable pinned native debugging.
Use DXC `-O3 -Zi -Qembed_debug`, or compare the same workflow with
`-Od -Zi -Qembed_debug`. Both require compatible embedded source/debug
information matching the executable; stripped inputs are not supported.
Native object registration is enabled, but registration no longer implies
pinning. Ordinary PSO/disk-cache reuse is bypassed, private debug objects are
not written to disk, and the optimized policy has distinct compiler/replay
fingerprints. The normal runtime and existing pinned APIs retain their behavior.

Use the same host selection/breakpoint APIs and submission/draining rules.
`NacelleGetOptimizedSourceDebugSite` supplements the shared site inventory with
the actual compilation identity, tier, executed-entry version, native object
identity and mapping classification. Inventory can include unavailable sites
and multiple compilations of the same shader. An `available` site belongs to a
successfully materialized, still-live native entry; it does not promise that
the next submission will choose that compilation or reach that line.

Load the first-party optimized aliases instead of the pinned aliases:

```text
(lldb) command source tools/nacelle-dbg/optimized-source-debug.lldb
(lldb) run
(lldb) hlsl-location
(lldb) hlsl-lanes
(lldb) hlsl-values
(lldb) hlsl-code
(lldb) hlsl-arm-step
(lldb) continue
```

The native all-stop target is `nacelle_optimized_source_debug_stop`.
`hlsl-arm-cancel` followed by `continue` cancels observations, not the workload.
As above, read the immutable snapshot and write its generation-tagged mailbox;
do not call lock-taking runtime APIs from an all-stopped debugger.

Optimized stops are **surviving observation points**, not promises to stop
before the original HLSL operation or its side effects. Optimization may delete,
move, merge, duplicate or unroll source operations. Lines can disappear, repeat
or jump; source stepping is not monotonically increasing line numbers. Mapping
classification distinguishes exact, merged and unavailable provenance.
Merged mappings include all proven origins (at most eight), each with its
source identity, line/discriminator and complete bounded inline chain; use
`hlsl-origins` to inspect them. Origin-cap exhaustion or missing origins cannot
be presented as a convenient single exact line. Breakpoints match the preserved
origins, and step-over uses their shallowest represented inline depth.
An unmapped breakpoint is rejected rather than moved silently to another line.
Step-over retains the bounded inline chain and cannot switch to another
invocation, primitive or compilation when an entry is promoted.

Non-observing debug provenance is carried through lowering and optimization;
executable observation calls are inserted only after normal Tier-1 passes or
the full Tier-2 O3 pipeline. Dead calculations are not resurrected, and missing
values are not recomputed to make a debugger display plausible data. Source
execution masks and invocation coordinates must remain provable; a host block
where an expression was hoisted is not sufficient evidence of source execution.
For pixel entry context, late identity reads may use the known immutable
internal runtime ABI and a surviving batch index. This is not a shader-memory
or general debug-expression dereferencer. An entry helper mask is recovered
only with explicit entry-context provenance; it never substitutes for an
unknown branch/discard predicate. Killed indices, invalid/padding batches,
unprovable primitive identity and multisample fallback remain unavailable.

`NacelleGetOptimizedSourceDebugSnapshot` adds per-slot `valid_mask` and
availability classifications to the shared snapshot. Interpret raw bits only
where the slot is available and the lane's validity bit is set. Bounded scalar
fragments can occupy separate slots. Optimized-out, out-of-scope, unsupported,
partial or otherwise unavailable bindings have explicit reasons, not fabricated
zeros or values from a different branch. Unsupported expressions/conversions
and inlined locals remain unavailable. Arbitrary dereferencing, source
expression evaluation and aggregate reconstruction are not supported.

The executed-entry version is distinct from the process-wide native
registration generation. In-flight Tier-1 work retains its original code and
metadata while subsequent work may select Tier 2. Vertex/pixel entry versions
remain independent. Native identity checks reject stale/unloaded objects.
`NacelleGetOptimizedSourceDebugStatus` reports retained source/metadata charges,
bound Tier-1/Tier-2 compilation counts and sticky process-wide compilation
failure. Compiler failures are logged explicitly; a failed asynchronous
compilation does not count as a successful optimized entry, status returns
`S_FALSE`, and session End cannot certify success. Restart for a clean status.

The source/site/snapshot/native-object budgets above also apply here; they do
not bound transient compiler RSS. Debug observation calls, spills and extended
live ranges can change optimized machine code and timing. This is not a claim
of byte-identical uninstrumented code or #211/#212 performance certification.
The optimized LLDB acceptance requires real O0/O3-input compute/pixel stepping,
per-lane values, unchanged full outputs and actual native Tier-2 entry versions.
Its promotion cases suspend compiler threads **in the test debugger only** to
select Tier 1 deterministically, then resume normal asynchronous compilation
and verify a separate Tier-2 submission without replacing the earlier entry.
The mixed graphics cases additionally execute VS version 2 with PS version 1,
then verify PS version 2. Compute checks inspect live floating-point values;
pixel checks inspect live unsigned coordinate locals while derivative/discard
outputs are checked over the complete image. Other values can legitimately be
optimized out; a constant-looking final color is not proof of a surviving
source-local binding.

### Debug-only compute cost diagnostic

The static fixture can compare the same DXIL with ordinary compilation and
optimized source debugging. Use `NACELLE_SIMD_WIDTH=4`,
`NACELLE_DISABLE_TIER2=0`, `NACELLE_TEST_SOURCE_OPTIMIZED_MODE=1`,
`NACELLE_TEST_SOURCE_UNATTENDED=1` and a fresh `NACELLE_CACHE_DIR` per process.
Pass `unused source-compute`, first with
`NACELLE_TEST_SOURCE_COST=ordinary`, then `NACELLE_TEST_SOURCE_COST=source`.
Set `NACELLE_TEST_SOURCE_INPUT_O0=1` for the O0-input pair; **unset** it for
the O3-input pair. Both still execute actual runtime Tier-2 code. Each process
checks every output; source mode records one unattended stop and explicitly
cancels further observations. These timings exclude interactive debugger pauses.

Five fresh-process Linux x86_64 samples gave these medians:

| DXC input | Runtime policy | First PSO (ms) | Remaining Tier-2 wait (ms) | Submission (ms) | Peak process RSS (KiB) |
|---|---|---:|---:|---:|---:|
| O0 | Ordinary | 18.61 | 49.75 | 0.827 | 88,064 |
| O0 | Optimized source | 44.61 | 173.92 | 2.183 | 111,856 |
| O3 | Ordinary | 17.49 | 37.75 | 0.622 | 88,064 |
| O3 | Optimized source | 39.68 | 156.88 | 1.847 | 109,252 |

The fixture creates two PSOs. Source mode bypasses ordinary caches and can
compile both; first-PSO time does not include the second creation. The wait
column measures only the remaining asynchronous drain after fixture setup,
not total O3 compiler CPU time. Source inventories include original unavailable
sites and separate surviving Tier-1/Tier-2 sites: 534/458 records for O0/O3
input, with 1,910,904/1,640,456 charged metadata bytes and 1,780 charged source
bytes (1,618 source-content bytes). The optimized snapshot type is 35,664 bytes.
RSS includes compiler/process memory and is not inferred from retention budgets.
These diagnostics use DXC 1.10 (`5201-3e6e1485`), LLVM 24.0.0git, LLDB 21.1.8
and GCC 13.3. They are small-workload observations, not #211/#212 certification
or a supported-compiler matrix beyond the actual acceptance tests.

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

Separately consented [compute/buffer replay](../nacelle-replay/README.md) uses a
different `.nacelle-replay` format and runtime-linked CLI. The standalone reader,
ordinary profile files, and browser viewer do not execute captured code.

## Native shader debugging (LLDB)

LLDB is the primary supported native shader debugger, with first-party
breakpoint, instruction-step, stack/frame, code-unload and output-preservation
tests. The initial verified platform is **Linux x86_64**, using the
repository-pinned LLVM 24 runtime toolchain. This is native debugging of the
lowered compute/VS/PS entry points, not HLSL source/lane/local-variable debugging.
macOS, Windows, other architectures, geometry/tessellation/mesh/raytracing
pipelines and optimized source debugging are not supported by this mode.
Unsupported configuration/pipelines fail explicitly.

Build the instrumented runtime as above, with LLDB installed. Linux runtime
profile-test configuration requires `lldb` (override its path with
`-DNACELLE_LLDB_EXECUTABLE=/path/to/lldb`); a missing debugger is not an acceptance
pass. The standalone offline reader still needs no debugger or LLVM.

Include [native_debug.h](../../include/nacelle/native_debug.h) in the application
and configure **before any pipeline compilation, hash or cache lookup**:

```cpp
NacelleNativeDebugOptions debug;
debug.enabled = 1;
HRESULT status = NacelleConfigureNativeDebug(&debug);
// Check status before creating pipelines; failures are also logged.
```

The mode is OFF by default, even with instrumentation ON. Configuration freezes
on first compiler use, including ordinary compilation; late changes and disabling
afterward are rejected. Restart the process to change policy. It applies to one
linked runtime instance; do not mix static/shared Nacelle runtimes.

The debug path skips the Tier-1 cleanup optimizer, uses native O0 emission,
retains frame pointers/unwind tables and pins code without asynchronous O3
promotion. Required DXIL/MLIR lowering and SIMD packetization still occur:
this is not a promise of DXC/HLSL O0 source mappings. Group/quad/wave execution
and scheduling are unchanged; the debugger stops real host threads, not an
independently executed selected shader lane. Host-thread selection is not HLSL
lane selection. Pausing all threads can delay application fences/timeouts.

Optimized disk/PSO caches and cached-blob import/export are bypassed/rejected;
debug objects are never written into them. Debug VS sharing is only between
pinned entries created in the same debug runtime. Pipeline hashes include the
debug policy and replay environment metadata identifies it explicitly.
Ordinary replay currently requires Tier 2 and therefore rejects pinned Tier-1
debug entries; do not treat native debugging as a replay guarantee.

### Launch, break, step, inspect, continue

The existing runtime fixture gives a reproducible repository-built workload:

```sh
cmake --build build/profile --target profile_runtime_static_test \
  profile_runtime_shared_test native_debug_test
lldb --no-lldbinit -- build/profile/tests/profile_runtime_static_test unused native-compute
```

Inside LLDB:

```text
settings set symbols.enable-external-lookup false
settings set symbols.enable-lldb-index-cache false
settings set target.disable-aslr false
settings set plugin.jit-loader.gdb.enable on
breakpoint set --func-regex ^nacelle_cs_
run
frame info
register read pc
thread step-inst
thread step-inst
thread backtrace
breakpoint disable
continue
```

A pending breakpoint before compilation is expected. On registration, LLDB adds
the JIT entry location; it must then stop in `JIT(...)` at the named shader.
The fixture must finish with `NATIVE DEBUG OUTPUT PASS` and exit 0.
Use `native-graphics` instead of `native-compute` and `^nacelle_vs_` or
`^nacelle_ps_` to reproduce VS/PS debugging. These fixtures use stripped DXIL,
demonstrating that native symbol debugging does not pretend HLSL information
exists. `frame variable` cannot provide unavailable HLSL locals or lane values.

External symbol lookup is disabled to keep the workflow local (and avoid
external-symbol-fetch failures in the tested distro LLDB 21). Index caching is
disabled to avoid persisting private JIT symbols. `target.disable-aslr false`
avoids requiring permission to change process ASLR; no host security setting
needs to be relaxed.

For **attach**, the same fixture can wait after compilation and before shader
execution, printing its PID:

```sh
NACELLE_TEST_NATIVE_ATTACH=1 build/profile/tests/profile_runtime_static_test unused native-compute
```

In another terminal start `lldb --no-lldbinit`, apply the same symbol/cache/JIT-loader settings,
then `process attach --pid <application-pid>`. LLDB discovers existing JIT
registrations. Use `image lookup -rn '^nacelle_cs_'` to find live entries and
`breakpoint set --name <exact-symbol>` (or `breakpoint set --address <address>`)
before continuing, then press Enter in the application terminal. Break, step,
inspect frames and continue as in the launch workflow. For a different
application, use an application-controlled safe point with compiled pipelines
still alive. Attach is subject to normal OS ptrace permissions;
launch the application under LLDB if attach is prohibited. Do not change
system-wide ptrace policy just for this workflow.
The local attach check was blocked by the shared host's `ptrace_scope=1`;
that policy was left unchanged. Launch, real shader stepping and unregister
acceptance passed with LLDB 21.1.8; attach is not claimed as locally verified.

The standard LLVM **`lldb-dap`** adapter can use the same LLDB settings as
initialization commands when launching/attaching to a configured application;
no custom adapter/protocol is introduced. CLI LLDB is the automated acceptance
surface here; a VS Code extension/editor integration is separate work.
GDB's standard JIT interface is compatible with the registration mechanism,
but GDB is secondary and is not substituted for LLDB acceptance.

### Identity, lifetime, limits and privacy

`NacelleGetNativeDebugCode(address, &code)` returns the live, half-open code range,
entry symbol, object SHA-256 and monotonically increasing code version. The
entry name contains its logical pipeline/stage hash; existing profile metadata
correlates shader identities and native object IDs. Interior-address queries
resolve to the same entry. `S_FALSE` explicitly means no live debug shader
range is available and clears the output; invalid queries return/log an error.
Addresses and versions are local-process data, not portable replay identities.

Registration uses LLVM's ORC RuntimeDyld layer and its standard JIT event
listener. LLDB's built-in `jit-loader.gdb` plugin handles this interface: its
name does **not** make LLDB support dependent on running GDB. The pinned LLVM
ELF JITLink plugin does not implement unregister notifications, so it is not
used for this opt-in path. Normal compilation/linking remains unchanged.

Registrations/backing debug objects follow ORC resource lifetime.
Nacelle currently retains JIT code for process lifetime: releasing a PSO or
clearing lookup caches is **not** code unload. No new per-PSO eviction is added.
Actual ORC removal unregisters symbols before releasing their memory and
invalidates live address records; replacement code gets a new version even if
an address is reused. Tests remove an isolated, quiescent resource and verify
that LLDB marks its old breakpoint unresolved. Never invoke an unloaded address;
an unresolved version breakpoint is not authorization to retarget it silently.
JIT teardown follows draining of outstanding compilation.

Object/registration accounting defaults to **32 MiB**, capped at **64 MiB** and
configurable downward via `object_bytes`. Admission conservatively charges
twice the raw ELF object size, twice the symbol-record storage and 4 KiB
overhead per object, including linked helper objects. Reservations follow the
runtime memory manager through success, failure and actual unload. A failed
materialization before memory-manager creation conservatively retains its
reservation until process teardown. Exhaustion fails compilation explicitly;
live registrations are never silently evicted or replaced by undebuggable
code. This bounds accounted retained objects/metadata, not compiler transient
memory or total process RSS, and is separate from the event-buffer budget.

Native debug emission strips inherited debug/source locations and replaces the
object source filename with a synthetic label. It does not read original source
paths, embed HLSL, dump objects to disk, upload data or open a remote listener.
Separately enabled profile artifacts retain their existing source-consent
rules. A debugger has trusted-local access to all application memory, not a
sandbox; OS debugger settings/third-party adapters remain the user's boundary.

Run the first-party checks:

```sh
ctest --test-dir build/profile -R '^native_debug_' --output-on-failure
```

The Linux profiling CI job runs real LLDB compute/VS/PS tests against both
static and shared runtimes, with exact PC-range/instruction-advancement,
caller-frame and output checks. Tests prepopulate optimized caches and verify
their files/digests remain unchanged. Configuration/budget failures and actual
ORC removal/replacement are covered separately. Normal OFF builds contain no
native-debug configuration, state, registration listener or API references.
Debugger-enabled costs are separate from profiler/performance certification.

A single shared Linux x86_64 host observation of the small fixture, **without
LLDB attached**, measured 0.05 s / 67.5 MiB peak RSS for native compute (two PSO
creations plus execution) and 0.04 s / 67.5 MiB for native graphics. These are
whole-process synthetic measurements, not isolated compiler overhead or
representative application benchmarks. Real LLDB startup/symbol/stop costs are
additional and highly debugger/host dependent. No profiler performance or
memory acceptance is certified by these observations.

## CI coverage

The Linux, macOS, and Windows CI jobs retain their normal OFF build and test
run, checking OFF compile commands, build targets, and static-library symbols
for instrumentation leakage using the repository-built `llvm-nm` without
fetching profiling dependencies. A dedicated profiling matrix on Linux,
macOS, and Windows runs concurrently with the normal OFF jobs to build
`build-profile/` with instrumentation enabled,
run the recorder and real static/shared capture tests, and run native-to-viewer
interoperability tests and the viewer build.
The matrix shares dependency checkout, toolchain cache restoration/build,
capture configuration/build/tests, and native-to-viewer validation. SDK installs
and Windows compiler options are platform-specific. Profiling jobs reuse the
normal jobs' OS/architecture-specific LLVM/MLIR/Clang and DXC SDK cache keys;
on a cache miss they install the same pinned toolchains independently.
Profiling failures do not affect the normal OFF job results, and no new Pages
publisher is added. Windows adds the instrumented runtime directory to
the DLL search path, exercising shared API exports as well as static linkage.
The instrumented configure steps report the actual runner OS/architecture.

The integration tests use fresh caches and exercise compute/graphics output
checks and asynchronous Tier-2 capture. Captures remain temporary CI workspace
data, not Pages assets. The Linux profiling job also runs offline Chromium upload/render/download tests,
comparing Chrome trace downloads with the native CLI export. Only production
viewer assets are served from local disk; all other requests are rejected.
These jobs do not certify performance budgets or guarantee a separate ARM64
runner. An ARM64 runner must actually pass the instrumented tests before ARM64
coverage is claimed; an OS label alone is not certification. Platform coverage is only verified
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
event or spatial-observation loss. Errors are returned and logged to stderr. After a write failure the
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
Nested synchronous scopes borrow the session held by their live stack parent,
without shared ownership updates. Root scopes retain the session; asynchronous
scopes retain their owning work context. Exporting a context from any nesting
depth acquires independent ownership, so it may safely outlive the original root
and continue recording while capture end waits for it. Worker coverage, event
counts, and draining semantics are unchanged.

### Opt-in shader text and source retention

This is separate from event capture and defaults to **OFF** even in an
instrumented runtime. Configure it **before creating pipelines** (and before
beginning execution-only capture of pre-existing pipelines):

```cpp
NacelleProfileArtifactOptions artifacts;
artifacts.flags = NACELLE_PROFILE_RETAIN_SHADER_TEXT;
// Add NACELLE_PROFILE_EMBED_SHADER_SOURCES only with explicit source consent.
HRESULT status = NacelleConfigureProfileArtifacts(&artifacts);
```

Check the HRESULT. Unknown flags, incompatible struct/version, or invalid budgets
are rejected and logged. Retention defaults to 32 MiB artifact text and 8 MiB
source data; caps are 64 MiB and 16 MiB respectively, configurable downward.
Accounting includes IDs, labels and mapping storage as well as text. It is not
a total-process RSS limit: compiler parsing/cloning, containers, serialization,
CodeMirror and decoded objects use additional memory. Opted-in compilation
does extra textual IR/disassembly work; ordinary ON-idle performance thresholds
do **not** certify this separately enabled collection mode.

Only debug data already present in the supplied DXC container is inspected.
For embedded source and preserved HLSL-to-DXIL locations, compile with DXC
`-Zi -Qembed_debug` (tests additionally use `-Od`). Nacelle never opens referenced
source paths, invokes background DXC, or recompiles a pre-existing pipeline to
invent missing artifacts. Text-only consent does not embed HLSL. Stripped input,
late opt-in and unsupported mappings are explicitly unavailable. Filename
references become capture-local `source-N.hlsl`; source **contents** can still
contain private information. Treat the entire archive as private.
Debug references are compared with normalized path separators, without reading
those paths or guessing a match from a basename.

Retained views include DXIL instruction text, lifted MLIR when that compilation
path ran, optimized LLVM for emitted native objects, and disassembly of exact
emitted or cached object bytes. Disk cache hits can have native text without
retained LLVM/MLIR. Tier-1 entries without object identity do not invent assembly.
Native offsets are relative to their labeled text section, never live JIT
addresses. Lifetimes, promotion, cache selection and unloading remain unchanged.
The panes resolve immutable shader/native owners for each captured code version,
including older entries retained across asynchronous promotion. Optimizations
can remove mappings: only supplied HLSL/DXIL mappings are currently preserved;
MLIR/optimized LLVM/native source and lane mappings are unavailable, not inferred.

Enabled sessions snapshot their retention store. Reconfiguring replaces the
store for future compiler work/captures; existing sessions retain their snapshot.
Quiesce compilation before reconfiguration for consistent coverage. Set `flags`
to zero to release global retention after captures finish. Overflow fails saving
with an explicit error and writes nothing; reconfigure and begin a **new**
capture. Reconfiguring cannot repair a stopped handle's overflowed snapshot.

Archives without retention keep manifest version 1. Enabled captures use
manifest version 2 and a declared `artifacts.json` payload (artifact schema 1).
The existing SHA-256 shader/native identities remain authoritative owners;
payload IDs and the declared artifact checksum use XXH64 of the exact text/JSON
bytes for corruption detection, **not cryptographic authenticity**. Native and
browser readers reject owner/selection disagreement, duplicate or unused
payloads, unsafe labels, checksum/size changes, absent source consent, mapping
bounds and declared text-budget violations. Both archive/decoded aggregate
limits remain 256 MiB. Updated readers accept versions 1 and 2; old readers
reject version 2 rather than silently dropping artifacts.

### Opt-in spatial and submitted-lane capture

Spatial collection is independently **OFF** by default, including when coarse
capture or shader text retention is enabled. Configure before Begin, not during
an active session:

```cpp
NacelleProfileSpatialOptions spatial;
spatial.flags = NACELLE_PROFILE_SPATIAL_COMPUTE | NACELLE_PROFILE_SPATIAL_PIXEL;
spatial.sample_stride = 1; // Observe every eligible coordinate.
HRESULT status = NacelleConfigureProfileSpatial(&spatial);
```

Check the HRESULT. Invalid flags, struct/version, stride, half-open ROI or budget
are logged and rejected. `flags = 0` disables future spatial captures. Sessions
own their settings and preallocated pool; allocation failure admits no capture.
The default pool is 8 MiB, configurable from 424 bytes through 16 MiB. Each
admitted entry call occupies one 424-byte record before post-drain aggregation:
8 MiB admits 19,784 calls and 16 MiB admits 39,568. The additional 131,072-cell
limit is not an independent allocation entitlement. At most 4,096 operations
are represented. The pool excludes preallocated operation counters/metadata,
strings, aggregation indices/cells, JSON, compression and browser objects;
**neither the pool nor the archive limit is a total-process RSS guarantee**.

ROI uses `min_xyz` (inclusive) and `max_xyz` (exclusive) in **cell coordinates**,
not pixel coordinates. A coordinate is selected when every axis is in range
and `(coordinate - minimum) % sample_stride == 0`. Pixel cells are global
32-by-32 raster tiles, top-left origin with Y downward; compute cells are group
XYZ with logical threads-per-group. Selection only suppresses observations:
all original shader calls, companion work, barriers and output writes still run.
Hot observations reserve a shared fixed-pool slot atomically and update bounded
operation counters; no per-call heap allocation, map or mutex is introduced.
Aggregation runs only after captured work drains.

Metrics intentionally describe **host entry submission**, not shader instruction
execution:

- `wallNs` sums wall time enclosing the actual entry function calls. A pixel
  batch's full time belongs to its tile, never divided into per-pixel estimates.
  It includes host scheduling/preemption and is not CPU cycles or GPU timing.
- `calls` counts observed entry calls. Compute `invocations` counts logical
  launched threads, not instruction activity. A dispatch-wide entry ABI reports
  `dispatch-wide-entry-unavailable` rather than inventing per-group observations.
- Pixel `invocations` includes every submitted SIMD slot, including padding.
  One entry call may submit multiple SIMD batches; lane totals count those
  batches, not just calls.
  Covered/quad-companion/padding counts classify coverage **before** depth and
  shader discard; they do not measure surviving fragments, dynamic divergence
  or lane values. Packing is quad-major, with lane `(y & 1) * 2 + (x & 1)`
  inside each quad. MSAA, overdraw and repeated calls count repeated submissions.
  Compute lane layout and instruction/source-lane precision remain unavailable.

Operations retain exact pipeline/PSO/stage/code-version and event identities,
including cached or asynchronously promoted entries. Separate raster calls or
dispatches remain separate operations; no cross-version averaging occurs.
The sparse payload preserves observed numeric zero. A missing cell is **not
zero**: it may have no submission, be outside ROI/sampling, or have lost records.
Without losses its conservative missing status is `unsampled`; with losses it
is `dropped-or-unsampled`. Dropped coordinates are not retained, so every retained
cell in that operation is conservatively `partial`, not claimed complete.
Unsupported operations use `unavailable` and no fabricated cells.

Pool exhaustion records exact observation-drop counts, preserves execution and
saves an explicitly incomplete bundle (`S_FALSE` / CLI exit 2), even with zero
event drops. Event drops remain separate in `NacelleProfileSummary`. Operation
metadata overflow or aggregation beyond the exact JSON integer range
(`2^53 - 1`) fails saving explicitly, without output. A stopped overflowed
capture cannot be repaired by reconfiguration; begin a new capture.

Spatial captures use manifest **version 3**, declaring checksummed `spatial.json`
(spatial schema 1), with optional artifact data. Native and browser readers
validate identities, event references when event recording is complete,
dimensions/ROI/stride, per-slot submission counts, capacity, exact integers and
combined completeness without dense allocation from dimensions. Legacy
versions 1/2 remain supported; older readers reject the required new capability.
The 256 MiB archive/decoded aggregate bounds remain unchanged.
Spatial opt-in embeds no resource contents, source files or host addresses.
Coordinates and counts can disclose private workload structure; keep captures
and exports private. No uploads, telemetry or external assets are added.

The instrumented benchmark worker supports `NACELLE_PROFILE_SPATIAL=1` together
with `NACELLE_PROFILE_OUTPUT`; it explicitly reserves the 16 MiB maximum for
repeated benchmark submissions. `0` leaves spatial collection disabled. Invalid
values or enabling without an output path fail explicitly. Its stderr reports
End's final drain/aggregation/serialization/write time separately from kernel
samples; stdout benchmark JSON is unchanged.

#### Spatial cost diagnostic

Six alternating-order matched pairs per workload used the same instrumented
Linux x86_64 executable, fresh per-process caches, SIMD 8, 256-by-256 output,
64 raymarch steps and five hot samples after Tier-2 warmup/drain. Coarse event
capture was active in both modes. All 24 processes produced complete captures;
each matched pair preserved its output checksum. Values are medians of six
process results. End includes final drain, not isolated serializer time.

| Workload | Spatial OFF / ON kernel | Change | OFF / ON End | OFF / ON peak RSS | ON spatial JSON |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compute raymarch | 0.2555 / 0.2733 ms | +6.97% | 0.970 / 15.412 ms | 156,776 / 173,262 KiB | 1,529,662 bytes |
| State-heavy pixel | 6.8113 / 6.8736 ms | +0.91% | 0.645 / 5.769 ms | 186,392 / 203,354 KiB | 153,265 bytes |

ON archives were approximately 1.54 MB / 0.167 MB respectively. The 16 MiB
benchmark pool is a separate spatial budget, not an increase to the 64 MiB
event budget. This shared-host diagnostic measures the separately opted-in
mode; it does **not** certify or relax #211's coarse idle/active thresholds or
#212's bookkeeping/RSS acceptance. OFF contains no spatial hooks/state or
profiling dependencies. Cross-platform correctness requires passing the
dedicated profiling matrix; these timings are Linux-only diagnostics.

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
for live application work. These historical measurements predate the bounded
streaming implementation and current stage measurements below. Deterministic recorder tests fail the actual event-pool allocation
and a serialization allocation using a test-only thread-local allocator probe.
They require `E_OUTOFMEMORY`, no partial output or stale active session, and a
successful retry of the stopped capture with identical event/drop counts. No
allocation-failure hooks or global allocator replacement enter the runtime.

#### Bounded serialization and current metadata scaling

The approved [#212 design](https://github.com/farzonl/Nacelle/issues/212#issuecomment-7115020880)
replaces simultaneous full trace JSON, `compressBound` storage and heap ZIP with:

- A sorted pointer index for completed event slots.
- Exact JSON-size preflight over the same Glaze-reflected values, then Glaze
  serialization into pre-sized storage. No output-buffer growth/allocation occurs
  inside Glaze's `noexcept` writer. JSON exceeding a format limit is rejected
  before allocating its output. Unsupported control bytes that the pinned Glaze
  writer does not escape are explicitly rejected; supported escapes and UTF-8
  round-trip. No custom JSON parser/encoder or source-content transformation is added.
- Two bounded event passes: count the exact decoded trace length, then encode
  one event at a time (at most 4 KiB plus library padding) into zstd streaming
  compression. The frame pledges the decoded size, remains dictionary-free,
  and uses level 1 with window log 19. Existing reader requirements and exact
  typed Chrome Trace bytes remain unchanged.
- A 128 KiB compression output buffer, private compressed spool, and miniz's
  stored-file archive APIs. The archive is staged privately, then copied in
  64 KiB chunks to an exclusively created destination. Neither a compressed
  payload nor the ZIP is held in a whole-archive heap buffer.

POSIX staging is owner-only, unlinked `tmpfile` storage. Windows uses native
cryptographic random names, exclusive creation, an owner-rights-only protected
ACL and delete-on-close handles, preserving UTF-16 paths. Every exit releases
both staging handles. A failed final copy/close removes only the destination
this attempt created; pre-existing destinations are never overwritten.
Transient serialization/allocation/write failures remain retryable with exact
counts. Actual metadata allocation failure is sticky: execution continues,
End returns `E_OUTOFMEMORY`, summary completeness is false, and a new capture is
required rather than saving missing identities. Code/version/selection limits
and invalid SHA-256 owners are also explicitly diagnosed. Synchronous compiler
configuration records through its existing counted root, without allocating an
unnecessary escaping Work context.

**Working-memory model (not total process RSS):** let `E <= 64 MiB` be the
event pool, `N <= floor(E / sizeof(Event))` its completed events, `C + S <= 65536`
the code-version/selection counts, and `P <= C + S` the capture-local pipeline
count. Recorder metadata scales as `O(C + S + P)`, with one bounded key/map node
per version/pipeline, at most eight IDs per selection, bounded vector capacity,
32-character PSO and optional 64-character shader/native owners, and fixed
stage/role/cache labels. CPU/compiler strings and feature-array serialization
are constrained by the 1 MiB manifest; raw contents and the minimum encoded
feature-array length are checked before copying them into a session. In-flight
roots/escaping contexts and thread stacks are owned by application work:
their memory is proportional to live ownership units, **not a fixed event-budget
entitlement**. TLS keeps fixed ID/slot cursors per thread, not additional event pools.

After drain, the serializer retains the admitted session metadata plus its
bounded metadata JSON `M <= 256 MiB` in aggregate, up to two 1 MiB manifest
buffers while updating descriptors, `N * sizeof(Event*)` for sorting,
one bounded event JSON buffer, the 128 KiB output buffer and zstd's level-1
512 KiB-window workspace. Archive construction and final writing run after
the sorting/compression workspace is released. Miniz keeps only its small
3..5-entry directory and bounded file-I/O scratch. Allocator/container capacity
and library workspace are reported separately; these formulas do not limit
compiler RSS, application work, OS page cache or browser decoded objects.
Both compressed spool and staged archive are individually limited to 256 MiB,
so staging disk can require up to 512 MiB, excluding the explicit final export.
The unchanged native/browser decoded aggregate and final archive cap is
256 MiB; the manifest cap is 1 MiB and event cap is 64 MiB.

The test-only [stage diagnostic](../../tests/profile_memory_test.cpp) recompiles
the production writer with private phase observations. No callbacks, timers,
allocator probe or diagnostic targets enter shipping libraries or OFF builds.
It reports current RSS, process lifetime high-water marks, 2 ms sampled stage
peaks, retained/requested storage and elapsed time. Short stages may be missed
by sampling; cumulative high-water marks are **never subtracted and mislabeled
as stage allocations**. Test-only allocation counts are cumulative requests,
including reallocation, not live bytes or allocator overhead.

Five fresh GNU 13.3 Release Linux x86_64 processes per matched baseline/new
synthetic case used the same 136-byte Event and version-tracked output. Every
bundle passed the native reader with exact event/drop/completeness counts:

| Budget / workload | Recorded / dropped | Baseline / streaming save peak RSS | Baseline / streaming save time |
| --- | ---: | ---: | ---: |
| 1 MiB nearly full | 7,697 / 0 | 8.89 / 6.00 MiB | 6.48 / 9.13 ms |
| 16 MiB nearly full | 123,348 / 0 | 82.83 / 22.00 MiB | 98.38 / 129.91 ms |
| 64 MiB nearly full | 493,434 / 0 | 320.83 / 72.50 MiB | 399.53 / 518.06 ms |
| 64 MiB overflow | 493,447 / 4,096 | 320.83 / 72.50 MiB | 396.95 / 523.76 ms |

This trades approximately 30% additional synthetic End time for about 77%
lower peak RSS at 64 MiB. It is not kernel overhead or a universal latency
guarantee. Final writing remains I/O-dependent; End still waits for real live
work, while these synthetic workers are drained before saving.

Five separate phase-observed processes per case measured current RSS, including
the diagnostic sampler, rather than isolated allocation deltas:

| 64 MiB case | After event allocation | After bookkeeping | After metadata JSON | During compression | After archive | After save |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearly full | 68.27 | 68.28 | 68.36 | 73.43 | 68.90 | 68.96 |
| 32,768 versions + 32,768 selections | 68.28 | 89.04 | 100.71 | 105.81 | 105.87 | 94.33 |
| 65,536 versions | 68.29 | 104.81 | 124.59 | 129.69 | 129.75 | 110.08 |

Units are MiB. On this STL, Session/Context/Work/CodeVersion/CodeSelection are
832/72/16/248/40 bytes; shared control blocks, map nodes, strings and allocator
overhead are additional. Maximum mixed/all-version bookkeeping requested
28,409,312/50,003,472 bytes cumulatively during construction; this includes
released vector growth and is not resident memory.
At 64 MiB, ordering retains 3,947,472 bytes; measured ordering + live zstd +
output/event buffers retained about 5,448,679 bytes. In the sampler run,
trace encoding and zstd/spool totals were separately approximately 284/63 ms.
Metadata-heavy archive copying increased End time and is not hidden in shader
timings. With 512 short-lived threads and retained contexts, a 1 MiB pool
recorded 121 events and explicitly dropped 391 from stranded blocks; context
creation cumulatively requested 90,040 bytes. Every output's counts were checked.

Reproduce the current stage matrix after enabling profile tests:

```sh
cmake --build build/runtime-check --target profile_memory_test
build/runtime-check/profile_memory_test 1 near-full
build/runtime-check/profile_memory_test 16 near-full
build/runtime-check/profile_memory_test 64 near-full
build/runtime-check/profile_memory_test 64 overflow
build/runtime-check/profile_memory_test 1 threads
build/runtime-check/profile_memory_test 64 metadata
build/runtime-check/profile_memory_test 64 versions
ctest --test-dir build/runtime-check \
  -R '^(instrumentation_test|profile_memory_.*test)$' --output-on-failure
```

Run each measurement in a fresh process; retain stdout JSON lines. Output
captures are private temporary files and removed after native validation.
Focused tests cover phase failures, including failure after the first final
write chunk, cleanup/retry, metadata allocator faults, output exclusivity,
manifest boundaries, Unicode and typed-trace byte identity. The existing
static/shared artifact/spatial and native/browser suites remain authoritative
format/execution checks. Raw comparative/stage samples and reproduction
evidence are published under [#212](https://github.com/farzonl/Nacelle/issues/212).

### Reproducible coarse-capture performance acceptance

The recorder contention improvements above are not shader-overhead acceptance.
Use [the comparison harness](../../scripts/measure_profile_overhead.py) for
paired, sequential fresh-process comparisons of a pre-change OFF worker,
current OFF, ON-idle, and active **coarse-only** capture. It runs the existing
1024x1024 raymarch compute and eight-tap textured graphics workloads, with the
same DXIL, SIMD width, inputs, and full worker coverage. Every output checksum
must agree. Every active archive must pass the native reader, be complete, have
zero drops, and identify the expected instrumented compiler. Source retention,
spatial data, replay, source/native debugging, and hardware counters are not
enabled or certified by this test.

Build both variants in Release using the same host compiler, LLVM installation,
flags, and compiler/ABI/runtime-bitcode inputs. The harness verifies all common
input SHA-256 values against the generated compiler stamps, rejecting stale
headers or unexpected differences. Since native/source debugging was added,
the ON stamp additionally covers seven debugger-only files; both complete
cache fingerprints and those extra inputs are reported, **not made equal or
overridden**. A separate matching ordinary-input digest identifies the common
compiler implementation. Cache namespaces and compiler/cache-key behavior are
unchanged.

Before making a recorder change, preserve the statically linked OFF worker from
the baseline revision. After rebuilding both current variants:

```sh
python3 scripts/measure_profile_overhead.py \
  --off-build build/runtime-off --on-build build/runtime-check \
  --baseline-off-worker /absolute/path/to/baseline-off-worker \
  --baseline-revision BASELINE_GIT_SHA \
  --shader-dir build/runtime-check/tests/shaders \
  --inspector build/runtime-check/tools/nacelle-dbg/nacelle-dbg \
  --output-dir /absolute/path/to/new-measurement-directory
```

The output directory must not exist. Ambient `NACELLE_*` settings are removed
from workers; all variants warm the private disk cache before recorded rounds.
All 24 four-variant permutations occur once per 24-round block in seeded
random order, balancing position and adjacent-variant effects. There are 240
timed iterations per process by default; `--iters` changes repetition count,
not work size, capture coverage, or the workload's short/long checks.
Do not run builds or other benchmarks concurrently.

`identity.json` records build inputs/flags, both full compiler fingerprints,
the common-input digest, baseline revision, harness/worker/DXIL hashes,
parameters and platform. `runs.jsonl` retains every warmup and timed result,
commands and complete active manifests; individual logs and captures remain
local. `summary.json` reports median paired ratios, observed ranges, and
20,000-resample paired bootstrap 95% intervals. Ratios compare **same-round**
per-process medians, not unpaired aggregate medians or pooled iterations.
The intervals are estimates for these workloads/host, not universal bounds.

An idle/active comparison passes only when its entire interval is at or below
1.01/1.05. Intervals crossing a threshold are **inconclusive**, not passes.
OFF regression is excluded by an identical pre-change/current static worker
binary, or otherwise requires a paired interval at or below 1.00. Identical
binary evidence does not erase the reported scheduling variability or turn a
negative timing estimate into proof of zero cost.

A shared development host is always non-certifying. Only on an exclusively used
benchmark environment add `--controlled-host`; this is an explicit operator
attestation, not automatic host reservation. Exit codes are 0 for accepted
controlled measurements, 1 for invalid evidence/exceeded budgets, and 2 for
inconclusive evidence. Preserve the report even when a run is not accepted.

The CI workflow's manual **Run controlled Linux coarse-capture performance
acceptance** input enables this separately from ordinary correctness CI:

```sh
gh workflow run ci.yml --ref YOUR_BRANCH -f profile_performance=true
```

After all correctness/browser work finishes, the Linux runner sequentially
builds the first-parent OFF baseline and current OFF in the same source/build
paths, verifies OFF isolation, then measures with 24 rounds and 60 iterations
per process. No build/test runs concurrently with timing within that runner.
The `coarse-profile-performance` artifact retains identity, raw results,
summary and logs on success or failure; it contains no source/resource payloads.
An unavailable or noisy runner leaves acceptance open. Normal CI does not
silently convert inconclusive performance into success.

The [controlled Linux x86_64 run on 2026-11-23](https://github.com/farzonl/Nacelle/actions/runs/45268760467)
passed all coarse performance gates on the first measurement attempt, using
1024x1024, SIMD8, 96 raymarch steps, 24 rounds and 60 iterations per process.
Its independently built first-parent/current OFF workers are byte-identical.
All 200 fresh-process outputs matched; all 50 active archives were complete,
zero-drop and native-reader validated. The ordinary compiler/ABI/bitcode
digest matched between variants; both distinct full debug-policy fingerprints
are preserved in the report.

| Workload | Paired idle overhead (95% interval) | Paired active overhead (95% interval) |
| --- | ---: | ---: |
| Raymarch compute | +0.041% [-0.145%, +0.153%] | +0.044% [-0.083%, +0.352%] |
| Textured graphics | -0.249% [-0.282%, -0.222%] | -0.243% [-0.291%, -0.205%] |

Negative estimates are not claims of zero cost or speedups. These qualified
results meet the coarse idle/active thresholds; they do not certify every
shader/host, other-platform performance, optional observation modes or
serialization memory. Both this run and normal PR CI passed all six
Linux/macOS/Windows normal/profiling jobs; Linux ran 288 native capture tests.
Raw results and identity evidence are retained as the run's
`coarse-profile-performance` artifact and published under
[#211](https://github.com/farzonl/Nacelle/issues/211).

## Format and current scope

### Profile reader fuzzing

The offline reader accepts a bounded in-memory span through `readProfileBytes`;
the file API uses the same parser. It validates stored ZIP local/directory
headers, names, flags/methods, descriptors, bounds/overlaps and CRCs before
accepting payloads. Native v1 remains exactly three entries, non-ZIP64. Zstd
requires one dictionary-free frame with matching content size, a window no
larger than 256 MiB, and valid decompression/checksum. No new runtime hooks,
payload schemas, or producer behavior are introduced.

Build the opt-in, standalone Clang libFuzzer entry with coverage and sanitizers
on the reader and its dependencies:

```sh
cmake -S tools/nacelle-dbg -B build/profile-reader-fuzz -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNACELLE_BUILD_PROFILE_FUZZER=ON \
  -DCMAKE_C_FLAGS="-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build/profile-reader-fuzz --target profile_reader_fuzz
build/profile-reader-fuzz/profile_reader_fuzz build/profile-fuzz-corpus \
  -runs=10000 -max_len=65536 -rss_limit_mb=2048 -timeout=10
```

Generate the sanitized shared corpus using the
[viewer instructions](../nacelle-view/README.md#compatibility-parser-parity-and-fuzzing).
The fuzz option exists only in the standalone tools configure and defaults OFF;
normal runtime/OFF builds do not acquire fuzz dependencies. Keep corpus/crash
files local and out of Pages artifacts. A finite successful run is not exhaustive
parser certification. Reader and browser object graphs still use memory beyond
the bounded 256 MiB payload; see the viewer's memory/retention documentation.

A `.nacelle-profile` is a stored ZIP containing:

* `manifest.json`: schema version 1, capture options, completeness/event counts,
  producer revision/compiler/OS/architecture, and captured compiler hash,
  target triple, CPU/features when a pipeline is created or an existing compute,
  raster draw, or mesh pipeline executes during the capture. Execution-only
  captures do not require recompilation; empty captures leave these fields
  unavailable. Configuration is recorded once per session, including concurrent
  compilation/execution, rather than collected for every dispatch.
* `trace.json.zst`: dictionary-free zstd-compressed Chrome Trace JSON.
* `shaders.json`: `[]` when no entries or graphics selections were observed,
  otherwise the independently versioned metadata object described below.
  PSO hashes, stage, SIMD width,
  compilation tier, cache outcome, and parent IDs also live in trace arguments.

Manifest clock offsets are nanoseconds; Chrome Trace `ts`/`dur` are microseconds.
Trace IDs and thread IDs are capture-local, not addresses or OS thread IDs.
Overlapping spans are not additive CPU time. Tier 0 and cache `unknown` mean
unavailable, not Tier 1 or a miss. Compute dispatch/worker spans identify the
selected native entry version, including pre-existing and cached pipelines.
Vertex, pixel, amplification and mesh regions identify their selected native
entry. Draws and multi-stage geometry regions reference composite selections
rather than claiming that one entry/tier describes the entire draw.

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

At most 65,536 code-version and composite-selection records combined may be registered per capture. Exceeding
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

### Graphics, mesh and content identities

Shader schema v2 retains the v1 compute rules and adds graphics entries with
`stage` and required `entryRole`: `vertex`, `pixel`, `geometry`, `domain`, `mesh`,
`amplification`, or stage `hull` with `hull-control-point`/`hull-patch-constant`.
The unique identity is `(pipelineId, entryRole, version)`; hull entries therefore
remain distinct even when they share a shader and publication number.
Graphics versions are monotonically numbered per native entry, starting at 1.
Independent stage promotion/cache selection is preserved, including a Tier-1
vertex cache hit combined with a newly compiled pixel entry. Unknown tier/cache
values remain explicit rather than being inferred from another stage.

Optional `shaderId` is a lowercase SHA-256 digest of the exact supplied shader
bytecode bytes (including its DXContainer), not a source filename or PSO hash.
It is retained before cache lookup, including for execution-only captures.
Optional `artifactId` is SHA-256 of the complete native object bytes supplied to
the JIT. Several entries may share one object digest. Tier-1 IR compilation does
not expose an object buffer here, so its artifact ID is omitted; optimized and
disk/memory-cache entries retain the actual object's digest. Digests identify
content, not anonymized source or relocatable function addresses. No object,
source, resource, or path is embedded.

Optional `codeSelections` contains consecutive one-based selection `id`,
`pipelineId`, and up to eight unique `codeVersionIds`. Trace `codeSelectionId`
references a draw's or geometry region's observed entries. Children contribute
to their enclosing selection, including old versions executed after promotion;
capture draining completes this metadata before serialization. An empty
selection explicitly represents no observed entry. Exact-entry spans may also
reference a selection, and must reference one of its entries. Composite spans
omit `codeVersionId` and use tier 0/cache `unknown`, not a fabricated latest tier.
Extended VS/HS/DS/GS work remains one coarse composite region; it is not
instruction-exclusive per-stage timing. Vertex/pixel/mesh regions also include
fixed-function work and must not be read as isolated shader CPU samples.

Instrumentation-ON graphics callables and metadata are published as one
immutable record at each existing selection boundary. Retained records live
with the PSO, and native object identity bookkeeping lives with the existing JIT;
both use additional memory outside the bounded event pool. No draw-wide freezing,
per-invocation hooks, or change to quad/wave/threadgroup execution is introduced.
OFF builds keep their atomic callable fields and exclude all identity state.
The existing event-buffer byte cap is unchanged; extra identity fields reduce
the number of events fitting a given byte budget.

Current native/browser readers accept `[]`, v1 and v2. Older strict native readers
reject the new trace/schema fields explicitly. v1 producers remain readable;
neither manifest schema nor the three-entry bundle contract changes. Native
readers still reject extra entries/unknown fields, while browser readers tolerate
manifest-declared additional entries and unknown object fields. Unsupported
schema versions, roles, digests and inconsistent selection references are errors.
Static/shared tests cover cold, promoted, memory/disk-cache, mixed cache outcomes,
SIMD 4/8, mesh/amplification, tessellation/geometry roles and output checks.
This does not certify the overhead budgets, instruction timing, or debugging.

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

### Lower-memory trace serialization

The writer sorts pointers to completed event slots and lets Glaze serialize a
lazy range of individual typed events. It no longer materializes a second
`TraceEvent` vector with per-event strings before constructing JSON. The pointer
index is released before compression; its storage is at most
`recordedEvents * sizeof(Event*)`. Capture buffers, chronological trace ordering,
JSON field names/encoding, zstd/ZIP format, limits, exclusive output creation,
and retry behavior are unchanged. This is not streaming file I/O: full JSON,
compression, and archive buffers still exist.

A fresh-process comparison used Clang 21.1.8 `-O3` on Linux x86_64, three
sequentially interleaved processes per writer/workload, with reverse-ordered
synthetic compute events carrying valid code-version references:

| Workload | Event budget | Recorded / dropped | Previous save peak RSS | Lazy save peak RSS | Previous / lazy save time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nearly full | 1 MiB | 8,179 / 0 | 14.21 MiB | 11.39 MiB | 11.12 / 8.98 ms |
| Nearly full | 16 MiB | 131,059 / 0 | 162.84 MiB | 116.84 MiB | 181.42 / 148.90 ms |
| Nearly full | 64 MiB | 524,275 / 0 | 640.34 MiB | 456.33 MiB | 752.57 / 605.96 ms |
| Overflow | 64 MiB | 524,288 / 4,096 | 640.34 MiB | 456.33 MiB | 714.68 / 600.40 ms |

Numbers are medians; save peaks are process-lifetime high-water marks sampled
immediately after serialization/write, before reader validation. Recording RSS
was about 67.7 MiB for both 64 MiB variants. Every saved bundle passed native
validation with exact completeness and counts. Raw samples, diagnostic source,
and reproduction instructions are published under
[#212](https://github.com/farzonl/Nacelle/issues/212).
These code-version workloads are not comparable to the earlier smaller-event
measurements. These historical results predate the bounded streaming/stage work above.
The event limit is still not a process RSS guarantee.

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

The historical shared-host samples below did not establish performance
acceptance; the controlled run above supplies the current qualified evidence.
A shared-host sample using Release builds
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
they are superseded for acceptance by the controlled measurements above.

A subsequent recorder diagnostic separately evaluated scope ownership and
counter false sharing (Clang 21.1.8, Linux x86_64, `-O3`, 12 balanced sequential
rounds per variant/thread count, 10,000 nested scope pairs per worker):

| Threads | Baseline | Borrowed scopes only | Borrowed scopes + counter alignment |
| --- | ---: | ---: | ---: |
| 1 | 1.4381 ms | 1.3887 ms | 1.4583 ms |
| 8 | 13.0082 ms | 2.7754 ms | 1.8525 ms |

Every run retained all events with zero drops (20,002 for one worker, 160,009
for eight). The timed region covers barrier release, recording, and worker join,
not event allocation, final count scanning, or serialization. Individual
eight-worker samples ranged from 11.2588-14.3048 ms (baseline),
2.2250-4.5322 ms (ownership only), and 1.4936-3.2550 ms (aligned counters).
The single-worker differences are within host variability.

The recorder retains both changes: nested scopes borrow their live owning
ancestor, async scopes retain Work without a redundant Session reference, and
shared counters use the runtime's existing 64-byte alignment convention.
On this build, Session grows from 480 to 768 bytes per capture and Scope from
176 to 184 bytes per live scope; Event size and the 64 MiB event pool are unchanged.
Exporting a deeply nested context walks its owning scope chain before acquiring
independent ownership. Raw samples and the reproducible diagnostic are published
under [#211](https://github.com/farzonl/Nacelle/issues/211).
These are recorder-contention measurements, not matched-hash shader acceptance
or a portable promise about cache-line sizes. Controlled OFF/idle/active shader
budget certification remains open.
