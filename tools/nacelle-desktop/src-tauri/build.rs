fn main() {
    #[cfg(feature = "smoke-harness")]
    let commands: &[&str] = &[
        "desktop_get_initial_capture",
        "desktop_open_dialog",
        "desktop_reload_capture",
        "desktop_read_chunk",
        "desktop_cancel_read",
        "desktop_export_start",
        "desktop_export_chunk",
        "desktop_export_finish",
        "desktop_export_cancel",
        "desktop_smoke_report",
    ];
    #[cfg(not(feature = "smoke-harness"))]
    let commands: &[&str] = &[
        "desktop_get_initial_capture",
        "desktop_open_dialog",
        "desktop_reload_capture",
        "desktop_read_chunk",
        "desktop_cancel_read",
        "desktop_export_start",
        "desktop_export_chunk",
        "desktop_export_finish",
        "desktop_export_cancel",
    ];
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(commands)),
    )
    .expect("failed to build Nacelle desktop Tauri manifest");
}
