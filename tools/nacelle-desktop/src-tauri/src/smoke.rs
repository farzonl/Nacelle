#![cfg(feature = "smoke-harness")]

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};

static BLOCKED_NAVIGATIONS: AtomicUsize = AtomicUsize::new(0);
static TRACE_EXPORTED: AtomicBool = AtomicBool::new(false);
static SPEEDSCOPE_EXPORTED: AtomicBool = AtomicBool::new(false);
static LOADED_REPORT: Mutex<Option<Value>> = Mutex::new(None);
static SMOKE_STARTED: AtomicBool = AtomicBool::new(false);

fn smoke_dir() -> Option<PathBuf> {
    std::env::var_os("NACELLE_DESKTOP_SMOKE_DIR").map(PathBuf::from)
}

pub fn record_blocked_navigation() {
    BLOCKED_NAVIGATIONS.fetch_add(1, Ordering::SeqCst);
}

pub fn override_export_path(allowlisted_name: &str) -> Option<PathBuf> {
    smoke_dir().map(|dir| dir.join(allowlisted_name))
}

pub fn record_export_commit(path: &Path) {
    if let Some(name) = path.file_name().and_then(|v| v.to_str()) {
        if name == "nacelle-trace.json" {
            TRACE_EXPORTED.store(true, Ordering::SeqCst);
        } else if name == "nacelle-speedscope.json" {
            SPEEDSCOPE_EXPORTED.store(true, Ordering::SeqCst);
        }
    }
}

#[tauri::command]
pub fn desktop_smoke_report(app: AppHandle, payload: Value) -> Result<(), String> {
    eprintln!("[smoke] report: {payload}");
    let Some(dir) = smoke_dir() else {
        return Ok(());
    };
    let stage = payload.get("stage").and_then(Value::as_str).unwrap_or("");
    if stage == "loaded" {
        if let Ok(mut guard) = LOADED_REPORT.lock() {
            *guard = Some(payload.clone());
        }
        if SMOKE_STARTED.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let app_handle = app.clone();
        std::thread::spawn(move || {
            // 1. Trigger a real external navigation inside the WebView and confirm on_navigation blocks it.
            if let Some(window) = app_handle.get_webview_window("main") {
                let _ = window.eval(
                    "(() => { const link = document.createElement('a'); link.href = 'https://example.invalid/blocked'; document.body.appendChild(link); link.click(); link.remove(); })();",
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(300));

            // 2. Drive real Trace and Speedscope exports through the real menu/IPC/staging path.
            let _ = app_handle.emit("nacelle://menu-action", "export-trace");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while !TRACE_EXPORTED.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }

            let _ = app_handle.emit("nacelle://menu-action", "export-speedscope");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while !SPEEDSCOPE_EXPORTED.load(Ordering::SeqCst)
                && std::time::Instant::now() < deadline
            {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }

            // 3. Report current WebView URL and complete summary back to the smoke runner.
            let current_url = app_handle
                .get_webview_window("main")
                .and_then(|window| window.url().ok())
                .map(|url| url.to_string())
                .unwrap_or_default();
            let loaded = LOADED_REPORT
                .lock()
                .ok()
                .and_then(|guard| guard.clone())
                .unwrap_or(Value::Null);
            let summary = json!({
                "loaded": loaded,
                "blockedNavigationCount": BLOCKED_NAVIGATIONS.load(Ordering::SeqCst),
                "currentUrl": current_url,
                "traceExported": TRACE_EXPORTED.load(Ordering::SeqCst),
                "speedscopeExported": SPEEDSCOPE_EXPORTED.load(Ordering::SeqCst),
            });
            let _ = std::fs::write(
                dir.join("smoke-result.json"),
                serde_json::to_vec_pretty(&summary).unwrap_or_default(),
            );
            app_handle.exit(0);
        });
    }
    Ok(())
}
