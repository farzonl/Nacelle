# Local compute replay

This is a separate, default-OFF capture capability. Ordinary `.nacelle-profile`
files remain `replayable:false`, contain no buffer snapshots, and cannot be
executed. The browser viewer and `nacelle-dbg` remain non-executing.

Build with `NACELLE_ENABLE_INSTRUMENTATION=ON` and
`NACELLE_BUILD_PROFILE_TOOLS=ON`, using the existing
[profiling build](../nacelle-dbg/README.md). Build the `nacelle-replay` target.
Unlike the standalone profile reader, this CLI links the runtime and LLVM.
There are no new dependencies, uploads, or network permissions.

## Trust and privacy

Snapshots contain complete application buffer contents, including initial bytes
of writable buffers. Shader binaries can reveal proprietary algorithms and
source paths. Enable snapshot and shader-binary consent separately; both are
required for replay. Embedded sources/debug-file metadata additionally require
`NACELLE_REPLAY_EMBEDDED_SOURCES`. Source-bearing binaries are rejected without
that consent, not silently stripped or rewritten.

Output is local, exclusive-create, and never overwrites an existing file.
Archive entry names are fixed; live pointers, resource handles, and host paths
are not serialized as provenance. Explicitly consented original shader binaries
can still contain paths/source text. Do not capture production secrets or upload
private bundles. The CLI is **not an OS sandbox**; execution is only for trusted
local shaders.

## Capture contract

Include [`nacelle/replay.h`](../../include/nacelle/replay.h) in an instrumented
application:

1. Configure `NacelleReplayConfiguration` with
   `NACELLE_REPLAY_SNAPSHOTS | NACELLE_REPLAY_SHADER_BINARIES` before creating
   participating buffers, root signatures, PSOs, descriptor heaps, and queues.
   This retains owned creation inputs and bounded object provenance.
2. Set `NACELLE_DISABLE_GEMM_FASTPATH=1` **before compilation**. External BLAS
   execution is excluded from this capability. This setting is checked, not
   forced. Keep SIMD, compiler options, and FP settings unchanged.
3. Finish all PSO creation and call `NacelleWaitForReplayCompilation()`.
   This drains existing asynchronous compilation, without forcing promotion.
   Participating code must already be stable Tier-2 with an exact native identity.
4. Initialize and unmap buffers. Declare every participating resource and its
   read/write access in `NacelleReplayCaptureOptions`. Explicitly declare
   `NACELLE_REPLAY_RACE_FREE_INPUTS | NACELLE_REPLAY_HOST_QUIESCENT`.
5. Begin capture, reset/record supported lists, and submit to one queue.
   No host mutation, object/descriptor creation, Map/Unmap, or cross-queue work
   is permitted during the transaction. Do not call queue Signal/Wait within it.
6. End capture to a new `.nacelle-replay` path and destroy the capture handle.
   End closes admission, drains admitted submissions, checks exact executed
   entry identity, snapshots expected writable output, and validates the bundle.
   A known unsupported operation invalidates capture but **does not suppress or
   change ordinary shader execution**. Invalid capture writes no runnable bundle.
7. Set configuration flags to zero to release retained provenance.

The caller must guarantee defined, in-bounds, race-free behavior, no inter-group
communication, and no external/host mutation. These assertions are not proven by
repeated matching outputs. No arbitrary shader determinism is claimed.

## Supported boundary

One explicit single-queue compute transaction supports:

- Owned committed buffers; complete initial snapshots and byte-exact expected
  final snapshots of declared writable resources.
- Compute PSOs, owned root-signature/DXIL bytes, root constants and buffer root
  CBV/SRV/UAV descriptors.
- Bounded CBV and raw/structured/typed buffer SRV/UAV tables, descriptor copies,
  and 32-bit-component FLOAT/UINT/SINT typed formats.
- Direct Dispatch XYZ, sequential buffer copies, whole-buffer UAV clears, and
  semantic transition/UAV barriers on the unified-host-memory runtime.
- Fixed SIMD 4/8/16 execution, wave operations admitted by the structured
  operation allowlist, and group-local shared-memory/barrier semantics.

Unsupported: graphics/draw/mesh/raytracing, textures/MSAA, placed/reserved/aliased,
shared/external resources, unbounded/sampler tables, UAV counters, bundles,
indirect commands, queries, predication, immediate writes, multiple queues,
overlapping same-buffer copies, partial clears, unsupported typed formats,
atomics (including race-free atomics), timing-sensitive calls, unknown DXIL
operations, and unapproved external/LLVM intrinsic calls.

The conditional guarantee requires the same runtime build/revision,
compiler/toolchain/backend, OS/architecture and dependent libraries, CPU/features,
SIMD and FP environment, complete inputs, and the caller contract above. Known
fingerprint mismatches fail before executable code; recompilation must produce
the exact captured stable native content identity before any dispatch. Repeated
output comparison catches differences, not their absence for every possible input.

## Format and limits

Schema 1 is a stored, non-ZIP64 archive with exactly `manifest.json`, `replay.json`,
and `blobs.bin`. The manifest declares capability, consent/contracts, environment,
budgets, and payload sizes/checksums. Typed metadata stores capture-local IDs,
checked offsets, original code inputs, and command order/submission boundaries.
Contiguous blobs have SHA-256 identities; no missing/unreferenced blobs are
allowed. Strict readers reject unknown/missing JSON keys, corrupt data, bad
references, invalid root/table/view semantics, and archive/aggregate violations.

| Budget | Default | Hard cap |
| --- | ---: | ---: |
| Initial + expected snapshots | 64 MiB | 192 MiB |
| Owned DXIL + root binaries | 16 MiB | 32 MiB |
| Command/provenance JSON | 8 MiB | 16 MiB |
| Manifest | 1 MiB | 1 MiB |
| Archive/decoded aggregate | 256 MiB | 256 MiB |

There are at most 4,096 resources/roots/PSOs/heaps, 65,536 commands/views, and
65,536 retained descriptor slots across heaps. Creation provenance conservatively
limits total retained buffer capacity to half the snapshot budget, even for
read-only buffers. JSON size and archive overhead are additionally checked at End;
overflow is an explicit failure, not a truncated runnable bundle.

## Validate or execute

```sh
build/profile/tools/nacelle-replay capture.nacelle-replay
# Validates archive, snapshots, shaders and binding semantics; never executes.

NACELLE_SIMD_WIDTH=4 NACELLE_DISABLE_TIER2=0 NACELLE_DISABLE_GEMM_FASTPATH=1 \
  build/profile/tools/nacelle-replay capture.nacelle-replay \
  --execute-trusted-local --repetitions 3
```

Use the settings captured by your application, not necessarily SIMD 4.
Every repetition creates fresh resources/descriptors/roots/PSOs, restores all
initial bytes, drains compilation, verifies code identity, executes original
command order, and checks complete writable output byte-for-byte. Failure reports
the bounded resource ID/repetition/byte offset, never buffer contents. Exit status
is zero only on successful validation/replay.

With profile tests enabled, `replay_runtime_static_test <new-path> table` creates a
sanitized synthetic vector fixture when `NACELLE_REPLAY_KEEP_OUTPUT=1` is set.
Use the environment above. Static/shared fixtures cover SIMD 4/8/16, 3D dispatch,
raw/typed/structured buffers, root bindings/constants, copies/clears, group barriers,
execution preservation, missing/corrupt snapshots, incompatible code/environment,
output mismatch, source consent, open maps and unsupported/atomic rejection.
`replay_cli_test` verifies the CLI's non-executing default and trusted repetitions.
Portable `replay_format_test` and opt-in `replay_reader_fuzz` need no LLVM runtime.

Snapshot retention/copying/serialization and fresh-state replay have independent
costs. Capture/replay timings are not ordinary execution benchmarks; these tests
do not certify epic #211/#212 overhead or memory thresholds.

On a shared Linux x64 host, five warm matched synthetic-fixture pairs measured
median Begin at 0.072 ms, End (including semantic validation/serialization) at
0.651 ms, and three fresh-state replays at 66.4 ms. The fixture captures 8 KiB
initial + 4 KiB expected bytes in a 21,133-byte archive. Median process peak RSS
was 43.5 MiB with replay disabled, 47.5 MiB for capture + non-executing validation,
and 71 MiB for the full replay acceptance process, including negative checks.
These small-fixture/shared-host observations are not production overhead,
large-budget memory certification, or isolation guarantees.
