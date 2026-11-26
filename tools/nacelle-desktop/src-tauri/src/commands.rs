use crate::policy::{validate_export_name, validate_export_size};
use crate::transfer::{CaptureHandleMeta, TransferState};
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::ipc::{InvokeBody, Request, Response};
use tauri::{AppHandle, State};
use tauri_plugin_dialog::DialogExt;

pub struct AppState {
    pub transfers: Mutex<TransferState>,
    pub initial_capture: Mutex<Option<Result<CaptureHandleMeta, String>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            transfers: Mutex::new(TransferState::default()),
            initial_capture: Mutex::new(None),
        }
    }
}

#[tauri::command]
pub fn desktop_get_initial_capture(
    state: State<'_, AppState>,
) -> Result<Option<CaptureHandleMeta>, String> {
    let mut slot = state
        .initial_capture
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    match slot.take() {
        None => Ok(None),
        Some(Ok(meta)) => Ok(Some(meta)),
        Some(Err(err)) => Err(err),
    }
}

#[tauri::command]
pub async fn desktop_open_dialog(
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<Option<CaptureHandleMeta>, String> {
    let app_handle = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        app_handle
            .dialog()
            .file()
            .set_title("Open Nacelle Profile")
            .add_filter("Nacelle Profile", &["nacelle-profile"])
            .blocking_pick_file()
    })
    .await
    .map_err(|err| format!("Open file dialog task failed: {err}"))?;

    let Some(file_path) = picked else {
        return Ok(None);
    };
    let path = file_path
        .into_path()
        .map_err(|_| "Selected capture must be a local filesystem path.".to_string())?;

    let mut transfers = state
        .transfers
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    transfers.open_capture(path).map(Some)
}

#[tauri::command]
pub fn desktop_reload_capture(state: State<'_, AppState>) -> Result<CaptureHandleMeta, String> {
    let mut transfers = state
        .transfers
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    transfers.reload_capture()
}

#[tauri::command]
pub fn desktop_read_chunk(
    state: State<'_, AppState>,
    id: u64,
    offset: u64,
) -> Result<Response, String> {
    let mut transfers = state
        .transfers
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    let bytes = transfers.read_chunk(id, offset)?;
    Ok(Response::new(bytes))
}

#[tauri::command]
pub fn desktop_cancel_read(state: State<'_, AppState>, id: u64) -> Result<(), String> {
    let mut transfers = state
        .transfers
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    transfers.cancel_read(id);
    Ok(())
}

#[tauri::command]
pub async fn desktop_export_start(
    app: AppHandle,
    state: State<'_, AppState>,
    name: String,
    size: u64,
) -> Result<u64, String> {
    let allowlisted = validate_export_name(&name)?;
    validate_export_size(size)?;

    let target: PathBuf = {
        #[cfg(feature = "smoke-harness")]
        if let Some(path) = crate::smoke::override_export_path(allowlisted) {
            path
        } else {
            choose_save_target(&app, allowlisted).await?
        }

        #[cfg(not(feature = "smoke-harness"))]
        {
            choose_save_target(&app, allowlisted).await?
        }
    };

    let mut transfers = state
        .transfers
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    transfers.begin_export(target, size)
}

async fn choose_save_target(app: &AppHandle, allowlisted_name: &'static str) -> Result<PathBuf, String> {
    let app_handle = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        app_handle
            .dialog()
            .file()
            .set_title("Export Nacelle Profile Data")
            .set_file_name(allowlisted_name)
            .add_filter("JSON", &["json"])
            .blocking_save_file()
    })
    .await
    .map_err(|err| format!("Save dialog task failed: {err}"))?;

    let Some(file_path) = picked else {
        return Err("Export cancelled.".to_string());
    };
    file_path
        .into_path()
        .map_err(|_| "Export destination must be a local filesystem path.".to_string())
}

#[tauri::command]
pub fn desktop_export_chunk(
    state: State<'_, AppState>,
    request: Request<'_>,
) -> Result<u64, String> {
    let id = parse_u64_header(&request, "x-nacelle-export-id")?;
    let offset = parse_u64_header(&request, "x-nacelle-export-offset")?;
    let bytes = match request.body() {
        InvokeBody::Raw(data) => data.as_slice(),
        InvokeBody::Json(_) => return Err("Expected raw binary export chunk.".to_string()),
    };

    let mut transfers = state
        .transfers
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    transfers.write_export_chunk(id, offset, bytes)
}

#[tauri::command]
pub fn desktop_export_finish(state: State<'_, AppState>, id: u64) -> Result<(), String> {
    let mut transfers = state
        .transfers
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    let _committed = transfers.finish_export(id)?;
    #[cfg(feature = "smoke-harness")]
    crate::smoke::record_export_commit(&_committed);
    Ok(())
}

#[tauri::command]
pub fn desktop_export_cancel(state: State<'_, AppState>, id: u64) -> Result<(), String> {
    let mut transfers = state
        .transfers
        .lock()
        .map_err(|_| "Desktop state lock poisoned.".to_string())?;
    transfers.cancel_export(id)
}

fn parse_u64_header(request: &Request<'_>, name: &str) -> Result<u64, String> {
    let raw = request
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| format!("Missing required export header {name}."))?;
    raw.parse::<u64>()
        .map_err(|_| format!("Invalid integer value for header {name}."))
}
