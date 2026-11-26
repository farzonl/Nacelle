# Nacelle Desktop Profile Viewer

Optional, local-only Tauri 2 desktop wrapper for `.nacelle-profile` archives.
It reuses the exact same built [`tools/nacelle-view/`](../nacelle-view/README.md)
frontend, `FileProfileDataSource` / Web Worker decoder, CodeMirror artifact
panes, spatial heatmaps, CPU counter summary, and embedded Speedscope timeline
inside the operating system's native WebView:

- **Windows x64**: Microsoft Edge **WebView2** (Evergreen runtime)
- **macOS ARM64**: Apple **WKWebView**
- **Linux x86_64**: **WebKitGTK 4.1** (`libwebkit2gtk-4.1`)

It adds no CMake targets, native runtime dependencies, instrumentation-OFF
cost, bundled browser runtime, auto-updater, crash reporter, or telemetry.

## Prerequisites

- **Node.js 22+** and **npm**
- **Rust 1.90+** (`cargo` / `rustc`)
- **Platform WebView prerequisites**:
  - **Windows x64**: MSVC Build Tools and system Microsoft Edge WebView2 Runtime
    (pre-installed on Windows 11 and Edge-updated Windows 10; packaging uses
    `webviewInstallMode: { type: "skip" }` so the installer never downloads a
    runtime silently).
  - **Linux x86_64**: `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`,
    `libappindicator3-dev`, `librsvg2-dev`, `patchelf`, `pkg-config`,
    `libssl-dev` (plus `xvfb` for headless smoke testing).
  - **macOS ARM64**: Xcode Command Line Tools (WKWebView is built into macOS).

## Build, test, and package

```sh
npm ci --prefix tools/nacelle-view
npm ci --prefix tools/nacelle-desktop
npm run build:frontend --prefix tools/nacelle-desktop
npm test --prefix tools/nacelle-desktop
npm run package --prefix tools/nacelle-desktop
```

Packaging targets by platform:
- **Windows x64**: portable `src-tauri/target/release/nacelle-desktop.exe` and
  NSIS installer under `src-tauri/target/release/bundle/nsis/`.
- **Linux x86_64**: standalone binary and `.deb` package under
  `src-tauri/target/release/bundle/deb/`.
- **macOS ARM64**: `.app` bundle under
  `src-tauri/target/release/bundle/macos/`.

Launch with an optional local capture path:

```sh
./tools/nacelle-desktop/src-tauri/target/release/nacelle-desktop path/to/capture.nacelle-profile
```

## Security, privacy, and IPC boundaries

| Boundary | Policy |
| :--- | :--- |
| Native file access | No `tauri-plugin-fs` and no `assetProtocol`. The WebView never passes arbitrary filesystem paths to Rust. Captures are authorized exclusively by user-mediated native Open dialogs, CLI arguments, macOS file-open events, or native OS window drag-and-drop. Rust validates `.nacelle-profile` extension, regular-file descriptor metadata, and the 22 B .. 256 MiB bounds on the open `File` handle, then streams bounded 64 KiB raw binary IPC chunks (`tauri::ipc::Response`) directly to the shared `FileProfileDataSource` Web Worker. No archive is extracted or staged to disk. |
| IPC capabilities | `src-tauri/capabilities/main.json` is bound strictly to window `"main"` with `"local": true`, no `"remote"` URLs, `withGlobalTauri: false`, and only the nine custom read/export commands plus `core:event:default`. No `dialog:*`, `fs:*`, `shell:*`, `http:*`, `opener:*`, `process:*`, `os:*`, or `clipboard:*` permissions are granted to the WebView. |
| Navigation & CSP | `tauri.conf.json` enforces `default-src 'none'`, `script-src 'self' 'wasm-unsafe-eval'`, `worker-src 'self' blob:`, `style-src 'self' 'nonce-nacelle-editor'`, `connect-src ipc: http://ipc.localhost`, `object-src 'none'`, `base-uri 'none'`, `form-action 'none'`. `on_navigation` denies all non-bundled origins (including external `http:`/`https:`, `file:`, `javascript:`, `data:`, `ws:`). `on_new_window` unconditionally returns `Deny`. |
| Exports | Only allowlisted filenames (`nacelle-trace.json`, `nacelle-speedscope.json`) up to 256 MiB are accepted. Destination selection uses a Rust-owned native Save dialog. Raw 64 KiB binary chunks (`InvokeBody::Raw`) stream directly into an unpredictably named sibling `.nacelle-export-<32-hex>.tmp` (mode `0600` on Unix), `sync_all()` + close + replace-rename (`MoveFileExW` on Windows, `rename` on Unix). Cancel, error, window close, or 120-second idle expiry removes the staging file and surfaces cleanup errors. OS filesystem rename/ACL semantics apply. |
| Remote capabilities (#226) | **Disabled / unavailable by default.** Zero network crates, outbound sockets, or remote capabilities are shipped. Future #226 integration requires an explicit opt-in build/runtime flag, explicit per-session user confirmation, and trusted-host loopback/SSH transport feeding the same `ProfileDataSource`/chunked-handle IPC boundary without widening WebView CSP or granting WebView network access. |

## Real-WebView smoke verification

Run the real-WebView open/decode/Speedscope/navigation-block/export smoke test
against a real native capture and `nacelle-dbg` CLI:

```sh
NACELLE_NATIVE_PROFILE="$PWD/build/profile/tests/static.nacelle-profile" \
NACELLE_NATIVE_READER="$PWD/build/profile/tools/nacelle-dbg/nacelle-dbg" \
  npm run test:smoke --prefix tools/nacelle-desktop
```

On headless Linux, wrap with `xvfb-run -a`. The smoke harness compiles only when
`--features smoke-harness` is explicitly passed; default and packaged production
builds never compile or expose `desktop_smoke_report`.

## Honest capability gaps

- **Verified platforms**: CI validates Linux x86_64 (WebKitGTK 4.1), Windows x64
  (WebView2), and macOS ARM64 (WKWebView). Linux ARM64, Windows ARM64, and
  macOS x86_64 are not certified until verified on matching runners.
- **Signing & notarization**: Packages are unsigned/ad-hoc development builds
  (Windows SmartScreen and macOS Gatekeeper warnings apply).
- **WebView prerequisites**: Linux requires system WebKitGTK 4.1; Windows
  requires system Edge WebView2 Runtime.
- **Memory**: The 256 MiB archive and decoded payload limits match
  `tools/nacelle-view` and are not a total OS WebView process RSS guarantee.
- **Debugging & remote**: Local shader debugging remains in
  [`tools/nacelle-dbg/`](../nacelle-dbg/README.md) and
  [`tools/nacelle-vscode/`](../nacelle-vscode/README.md); authenticated remote
  capture/debugging is deferred to #226.
