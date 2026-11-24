mod commands;
mod policy;
#[cfg(feature = "smoke-harness")]
mod smoke;
mod transfer;

use commands::AppState;
use std::path::PathBuf;
use tauri::menu::{MenuBuilder, MenuItemBuilder, SubmenuBuilder};
use tauri::webview::NewWindowResponse;
use tauri::{AppHandle, DragDropEvent, Emitter, Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};

fn open_authorized_path(app: &AppHandle, path: PathBuf) {
    let state = app.state::<AppState>();
    let result = match state.transfers.lock() {
        Ok(mut transfers) => transfers.open_capture(path),
        Err(_) => Err("Desktop state lock poisoned.".to_string()),
    };
    match result {
        Ok(meta) => {
            let _ = app.emit("nacelle://open-capture", meta);
        }
        Err(reason) => {
            let _ = app.emit("nacelle://host-error", reason);
        }
    }
}

fn build_menu(app: &AppHandle) -> tauri::Result<tauri::menu::Menu<tauri::Wry>> {
    let open = MenuItemBuilder::with_id("open", "Open Profile...")
        .accelerator("CmdOrCtrl+O")
        .build(app)?;
    let reload = MenuItemBuilder::with_id("reload", "Reload Profile")
        .accelerator("CmdOrCtrl+R")
        .build(app)?;
    let export_trace = MenuItemBuilder::with_id("export-trace", "Export Chrome Trace JSON...")
        .accelerator("CmdOrCtrl+E")
        .build(app)?;
    let export_speedscope = MenuItemBuilder::with_id(
        "export-speedscope",
        "Export Filtered Speedscope Profile...",
    )
    .accelerator("CmdOrCtrl+Shift+E")
    .build(app)?;

    let file_menu = SubmenuBuilder::new(app, "File")
        .item(&open)
        .item(&reload)
        .separator()
        .item(&export_trace)
        .item(&export_speedscope)
        .separator()
        .quit()
        .build()?;

    let edit_menu = SubmenuBuilder::new(app, "Edit")
        .copy()
        .select_all()
        .build()?;

    MenuBuilder::new(app)
        .item(&file_menu)
        .item(&edit_menu)
        .build()
}

pub fn run() {
    let initial_arg: Option<PathBuf> = std::env::args_os().nth(1).map(PathBuf::from);
    let app_state = AppState::default();
    if let Some(path) = initial_arg {
        let initial_result = app_state
            .transfers
            .lock()
            .map_err(|_| "Desktop state lock poisoned.".to_string())
            .and_then(|mut transfers| transfers.open_capture(path));
        if let Ok(mut slot) = app_state.initial_capture.lock() {
            *slot = Some(initial_result);
        }
    }

    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
        .on_menu_event(|app, event| {
            let action = match event.id().as_ref() {
                "open" => Some("open"),
                "reload" => Some("reload"),
                "export-trace" => Some("export-trace"),
                "export-speedscope" => Some("export-speedscope"),
                _ => None,
            };
            if let Some(action) = action {
                let _ = app.emit("nacelle://menu-action", action);
            }
        })
        .setup(|app| {
            #[cfg(feature = "smoke-harness")]
            app.add_capability(
                tauri::ipc::CapabilityBuilder::new("smoke-capability")
                    .window("main")
                    .permission("allow-desktop-smoke-report"),
            )?;

            let menu = build_menu(app.handle())?;
            app.set_menu(menu)?;

            let app_handle = app.handle().clone();
            let window = WebviewWindowBuilder::new(
                app,
                "main",
                WebviewUrl::App("index.html".into()),
            )
            .title("Nacelle Profile Viewer")
            .inner_size(1280.0, 840.0)
            .min_inner_size(720.0, 520.0)
            .on_navigation(|url| {
                let allowed = policy::is_allowed_navigation(url);
                #[cfg(feature = "smoke-harness")]
                if !allowed {
                    smoke::record_blocked_navigation();
                }
                allowed
            })
            .on_new_window(|_url, _features| NewWindowResponse::Deny)
            .build()?;

            window.on_window_event(move |event| match event {
                WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) => {
                    if paths.len() != 1 {
                        let _ = app_handle.emit(
                            "nacelle://host-error",
                            "Drop a single .nacelle-profile archive at a time.",
                        );
                        return;
                    }
                    open_authorized_path(&app_handle, paths[0].clone());
                }
                WindowEvent::CloseRequested { .. } | WindowEvent::Destroyed => {
                    let state = app_handle.state::<AppState>();
                    if let Ok(mut transfers) = state.transfers.lock() {
                        transfers.shutdown();
                    };
                }
                _ => {}
            });

            Ok(())
        });

    #[cfg(feature = "smoke-harness")]
    let builder = builder.invoke_handler(tauri::generate_handler![
        commands::desktop_get_initial_capture,
        commands::desktop_open_dialog,
        commands::desktop_reload_capture,
        commands::desktop_read_chunk,
        commands::desktop_cancel_read,
        commands::desktop_export_start,
        commands::desktop_export_chunk,
        commands::desktop_export_finish,
        commands::desktop_export_cancel,
        smoke::desktop_smoke_report,
    ]);

    #[cfg(not(feature = "smoke-harness"))]
    let builder = builder.invoke_handler(tauri::generate_handler![
        commands::desktop_get_initial_capture,
        commands::desktop_open_dialog,
        commands::desktop_reload_capture,
        commands::desktop_read_chunk,
        commands::desktop_cancel_read,
        commands::desktop_export_start,
        commands::desktop_export_chunk,
        commands::desktop_export_finish,
        commands::desktop_export_cancel,
    ]);

    let app = builder
        .build(tauri::generate_context!())
        .expect("error while building Nacelle desktop profile viewer");

    app.run(|app_handle, event| {
        #[cfg(target_os = "macos")]
        if let tauri::RunEvent::Opened { urls } = event {
            if let Some(first) = urls.into_iter().next() {
                if let Ok(path) = first.to_file_path() {
                    open_authorized_path(app_handle, path);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (app_handle, event);
    });
}
