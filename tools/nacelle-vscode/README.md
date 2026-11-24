# Nacelle for VS Code

Optional, local-first integration for the existing Nacelle profile viewer and
standard LLVM `lldb-dap`. It adds no native runtime dependency, custom debugger,
language server, network listener, automatic elevation, or telemetry.

## Build and install locally

Use Node 22.18 or newer and VS Code 1.106 or newer:

```sh
npm ci --prefix tools/nacelle-view
npm ci --prefix tools/nacelle-vscode
npm run build --prefix tools/nacelle-vscode
npm test --prefix tools/nacelle-vscode
npm run package --prefix tools/nacelle-vscode
code --install-extension tools/nacelle-vscode/nacelle-vscode-0.1.0.vsix
```

The VSIX embeds the same built frontend, decoding, artifact/heatmap panes and
Speedscope implementation as the browser. Third-party notices are packaged.
This private development package is not a Marketplace publication or a change
to the repository's licensing.

Open a `.nacelle-profile` file with **Nacelle Profile**. The editor is read-only;
invalid/incomplete captures retain their ordinary explicit diagnostics. Profile
versions follow the shared viewer built from the same source revision. Source
and resource retention are never enabled by opening a capture.

Only packaged assets are webview resources. The host reads only its selected
document and transfers bounded, ordered 64 KiB chunks over VS Code's existing
message channel; read messages cannot supply another URI. No capture is staged
to disk. This avoids widening resource roots to the capture's parent directory.
The decoding worker uses a local blob bundle. File changes cancel older loads;
closing disposes workers, views and active export transactions. Captures are
bounded by the shared 256 MiB archive/decoded limits; this is not a total VS Code
RSS guarantee.
Hidden panels retain their context to preserve the viewed capture and transfer
identities; close a panel to release its state.

Exports use an explicit VS Code Save dialog. No capture/message supplies a target
path. A single transaction uses ordered 64 KiB chunks with acknowledgements,
the existing 256 MiB ceiling, an unpredictably named staging sibling and rename.
Filesystem provider rename/permission semantics apply, not universal atomicity
or private-ACL guarantees; failures/cleanup errors are surfaced. Read-only
viewing works in untrusted workspaces; export writes and debugger execution
require workspace trust. Virtual filesystem workspaces are unsupported.
Idle transfers expire after 120 seconds. Speedscope runs unchanged in a
nonce-controlled, same-origin frame using its existing file-import boundary;
packaged script/styles/fonts are embedded to avoid cross-origin blob access.

## Native and HLSL debugging

Genuine Nacelle shader debug support follows the runtime: **Linux x86_64** and
**Windows x64**, with an instrumented runtime configured before compilation.
macOS profile viewing is independent; its Nacelle runtime shader debugging is
explicitly unsupported, not approximated with host stepping. On Windows x64,
enabled native/source debugging emits Win64-ABI ELF JIT objects registered
through LLDB's GDB JIT loader, and host debug targets require PDB symbols
(`/Z7` + `/DEBUG`). Ensure the Python runtime required by your `lldb-dap.exe`
build (e.g., Python 3.10 for upstream LLVM 21.1.8 Windows binaries) is on
`PATH`.

Install upstream `lldb-dap` and set `nacelle.debug.lldbDapPath` if needed. The
adapter runs on the workspace extension host, including an existing VS Code
Remote-SSH host. It is not a #226 remote RPC implementation. Missing adapters,
symbols, mappings and unsupported pipelines remain errors/unavailable data.

Example native fixture launch:

```json
{
  "type": "nacelle-lldb",
  "request": "launch",
  "name": "Nacelle compute",
  "program": "${workspaceFolder}/build/profile/tests/profile_runtime_static_test",
  "args": ["unused", "native-compute"],
  "mode": "native"
}
```

For your application, explicitly configure the
[native/source debugging APIs](../nacelle-dbg/README.md). Launch settings do not
magically configure an ordinary runtime. External symbol lookup and JIT index
caching are disabled by default; the JIT loader is enabled while preserving
ASLR by default and without changing ptrace permissions. Disabling target-process
ASLR requires explicit `disableASLR: true`. Native pending shader-symbol breakpoints resolve after
materialization. Standard native instruction stepping is separate from HLSL
source stepping.

Use `mode: "hlsl"` for the pinned source workflow, or
`mode: "optimized-hlsl"` for the separately opted-in optimized workflow. Both
reuse the shipped LLDB aliases. The application supplies source/local consent,
the selected invocation and initial HLSL source breakpoints before submission.
VS Code editor-line HLSL breakpoint insertion and source step-in/out are not
claimed.

At a genuine source-snapshot stop:

- **Nacelle: Inspect HLSL Source Location** prints exact source metadata and
  optionally opens a user-selected source file only after its SHA-256 matches
  the stopped embedded-source identity. No captured path is opened implicitly.
- **Inspect HLSL Lane** displays a captured SIMD slot, active/real/helper status,
  coordinates and raw local bits. It does not retarget the selected invocation.
  Optimized validity/availability is displayed separately; inactive/unsupported
  values must not be interpreted as successful zeros.
- **Inspect HLSL Values and Code Version** uses the existing immutable snapshot.
- **Step Selected HLSL Invocation**, **Continue HLSL Observation**, and
  **Cancel HLSL Observation and Resume** write only the existing generation-tagged
  mailbox through LLDB and then resume the whole program. They never call
  lock-taking runtime APIs while other threads are frozen.

Stopped-frame/session generations are discarded on resume, exit and disconnect;
stale or non-snapshot controls fail explicitly. Generic VS Code Stop can terminate
the application; use HLSL cancellation/resume when preserved workload completion
is desired. Attach is not claimed as tested support by this integration.

## Upstream HLSL language intelligence

Install `llvm-vs-code-extensions.vscode-clangd` separately and point its
`clangd.path` at a HLSL-enabled upstream clangd with matching resource headers.
This extension neither installs nor starts another language server. HLSL driver,
shader-model, stage and default-header flags must be supplied through your
clangd compilation database/configuration; DXC command lines are not automatically
valid clang command lines. Unsupported intrinsics/stages produce genuine clangd
diagnostics, not a fake complete HLSL service.

Verified baseline: upstream clangd 21.1.8 with its matching experimental HLSL
headers, compute shader model 6.0, basic scalar/vector types, dispatch semantics,
semantic highlighting/inlay checks, and genuine undeclared-identifier diagnostics.
Some distribution packages omit `hlsl.h`; provide that header and the `hlsl/`
directory from the matching upstream Clang release. Example `.clangd`:

```yaml
If:
  PathMatch: .*\.hlsl
CompileFlags:
  Add:
    - -x
    - hlsl
    - -std=hlsl202x
    - --target=dxil-pc-shadermodel6.0-compute
    - -I/path/to/matching/clang/resource/include
    - -include
    - hlsl.h
    - -Xclang
    - -finclude-default-header
    - -Xclang
    - -hlsl-entry
    - -Xclang
    - main
    - -Wno-invalid-constexpr
```

The last flag works around clang 21's invalid-constexpr diagnostics in its
experimental compatibility overload header; it also suppresses that diagnostic
group in your source. It does not suppress type/name errors or provide missing
intrinsics. This is a tested subset, not complete DXC compatibility. Set the
actual entry name, stage/model and include paths per file; do not apply compute
flags to pixel/graphics shaders. Verify your configuration with
`clangd --enable-config --check=/absolute/path/to/shader.hlsl` before relying on it.

## Validation

```sh
npm test --prefix tools/nacelle-view
npm run build --prefix tools/nacelle-vscode
npm test --prefix tools/nacelle-vscode
xvfb-run -a npm run test:smoke --prefix tools/nacelle-vscode
NACELLE_EXTENSION_SMOKE_UNTRUSTED=1 xvfb-run -a npm run test:smoke --prefix tools/nacelle-vscode
```

The Linux smoke uses an isolated VS Code 1.106.3 Extension Host, a real native
capture, actual shared-webview and embedded Speedscope rendering, explicit Save
UI/export, editor teardown, and real native/pinned/optimized shader steps through
standard `lldb-dap`, including source-identity navigation and SIMD-slot inspection,
followed by preserved output. The restricted-mode run verifies read-only rendering
and denied export/HLSL controls without trusting the workspace. Override
`NACELLE_EXTENSION_DEBUG_PROGRAM` and `NACELLE_EXTENSION_PROFILE_READER` for
another build directory, and `NACELLE_EXTENSION_SOURCE_FILE` for its exact source.
It requires Xvfb, standard lldb-dap with command REPL mode (verified 21.1.8), and
the existing native fixtures/tools;
missing prerequisites are failures, not accepted skips. Test profiles, caches
and VS Code user state are private temporary files and removed after the run.
Remote-SSH uses VS Code's workspace-host facilities; an actual SSH session and
Windows/macOS editor hosts are not certified by the Linux smoke.
