use crate::policy::{
    validate_capture_filename, validate_export_size, validate_open_capture_file, CHUNK_BYTES,
    IDLE_TIMEOUT_SECS,
};
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CaptureHandleMeta {
    pub id: u64,
    pub name: String,
    pub size: u64,
}

struct ReadSession {
    id: u64,
    file: File,
    size: u64,
    offset: u64,
    last_activity: Instant,
}

struct ExportSession {
    id: u64,
    target: PathBuf,
    staging: PathBuf,
    file: Option<File>,
    declared_size: u64,
    written: u64,
    committed: bool,
    last_activity: Instant,
}

impl ExportSession {
    fn cleanup(&mut self) -> Result<(), String> {
        self.file = None;
        if self.committed {
            return Ok(());
        }
        match std::fs::remove_file(&self.staging) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(format!("Export staging cleanup failed: {err}")),
        }
    }
}

impl Drop for ExportSession {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

pub struct TransferState {
    last_authorized_path: Option<PathBuf>,
    next_read_id: u64,
    active_read: Option<ReadSession>,
    next_export_id: u64,
    active_export: Option<ExportSession>,
    idle_timeout: Duration,
}

impl Default for TransferState {
    fn default() -> Self {
        Self::with_timeout(Duration::from_secs(IDLE_TIMEOUT_SECS))
    }
}

impl TransferState {
    pub fn with_timeout(idle_timeout: Duration) -> Self {
        Self {
            last_authorized_path: None,
            next_read_id: 0,
            active_read: None,
            next_export_id: 0,
            active_export: None,
            idle_timeout,
        }
    }

    pub fn expire_stale(&mut self) {
        self.expire_stale_at(Instant::now());
    }

    fn expire_stale_at(&mut self, now: Instant) {
        if let Some(read) = &self.active_read {
            if now.saturating_duration_since(read.last_activity) >= self.idle_timeout {
                self.active_read = None;
            }
        }
        if let Some(export) = &self.active_export {
            if now.saturating_duration_since(export.last_activity) >= self.idle_timeout {
                if let Some(mut stale) = self.active_export.take() {
                    let _ = stale.cleanup();
                }
            }
        }
    }

    pub fn open_capture(&mut self, path: PathBuf) -> Result<CaptureHandleMeta, String> {
        self.expire_stale();
        let name = validate_capture_filename(&path)?;
        let file = File::open(&path)
            .map_err(|err| format!("Unable to open selected capture: {err}"))?;
        let size = validate_open_capture_file(&file)?;

        self.next_read_id = self
            .next_read_id
            .checked_add(1)
            .ok_or_else(|| "Read transfer counter overflow.".to_string())?;
        let id = self.next_read_id;

        self.last_authorized_path = Some(path);
        self.active_read = Some(ReadSession {
            id,
            file,
            size,
            offset: 0,
            last_activity: Instant::now(),
        });

        Ok(CaptureHandleMeta { id, name, size })
    }

    pub fn reload_capture(&mut self) -> Result<CaptureHandleMeta, String> {
        self.expire_stale();
        let path = self
            .last_authorized_path
            .clone()
            .ok_or_else(|| "No capture is currently open to reload.".to_string())?;
        self.open_capture(path)
    }

    pub fn read_chunk(&mut self, id: u64, offset: u64) -> Result<Vec<u8>, String> {
        self.expire_stale();
        let finished = {
            let session = self
                .active_read
                .as_mut()
                .ok_or_else(|| "No active capture read transfer.".to_string())?;
            if session.id != id || session.offset != offset || offset >= session.size {
                return Err("Read chunk is stale or out of order.".to_string());
            }
            let remaining = (session.size - session.offset) as usize;
            let want = remaining.min(CHUNK_BYTES);
            let mut buf = vec![0u8; want];
            let mut filled = 0usize;
            while filled < want {
                let n = session
                    .file
                    .read(&mut buf[filled..])
                    .map_err(|err| format!("Failed reading capture chunk: {err}"))?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            if filled == 0 {
                return Err("Selected capture was truncated while reading.".to_string());
            }
            buf.truncate(filled);
            session.offset += filled as u64;
            session.last_activity = Instant::now();
            let done = session.offset == session.size;
            return {
                if done {
                    self.active_read = None;
                }
                Ok(buf)
            };
        };
        #[allow(unreachable_code)]
        Ok(finished)
    }

    pub fn cancel_read(&mut self, id: u64) {
        if self.active_read.as_ref().is_some_and(|session| session.id == id) {
            self.active_read = None;
        }
    }

    pub fn begin_export(&mut self, target: PathBuf, declared_size: u64) -> Result<u64, String> {
        self.expire_stale();
        validate_export_size(declared_size)?;
        if self.active_export.is_some() {
            return Err("Another export is already in progress.".to_string());
        }
        let parent = target
            .parent()
            .ok_or_else(|| "Export destination has no parent directory.".to_string())?;
        let (staging, file) = create_staging_file(parent)?;

        self.next_export_id = self
            .next_export_id
            .checked_add(1)
            .ok_or_else(|| "Export transfer counter overflow.".to_string())?;
        let id = self.next_export_id;

        self.active_export = Some(ExportSession {
            id,
            target,
            staging,
            file: Some(file),
            declared_size,
            written: 0,
            committed: false,
            last_activity: Instant::now(),
        });
        Ok(id)
    }

    pub fn write_export_chunk(
        &mut self,
        id: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<u64, String> {
        self.expire_stale();
        if bytes.is_empty() || bytes.len() > CHUNK_BYTES {
            return Err("Export chunk is empty or exceeds 64 KiB.".to_string());
        }
        let valid = self.active_export.as_ref().is_some_and(|session| {
            session.id == id
                && session.written == offset
                && offset.saturating_add(bytes.len() as u64) <= session.declared_size
        });
        if !valid {
            return Err("Export chunk is stale, out of order, or exceeds declared size.".to_string());
        }

        let write_result = {
            let session = self.active_export.as_mut().unwrap();
            let file = session
                .file
                .as_mut()
                .ok_or_else(|| "Export staging file is closed.".to_string())?;
            file.write_all(bytes)
                .map_err(|err| format!("Failed writing export chunk: {err}"))
                .map(|()| {
                    session.written += bytes.len() as u64;
                    session.last_activity = Instant::now();
                    session.written
                })
        };

        match write_result {
            Ok(next_offset) => Ok(next_offset),
            Err(write_err) => {
                if let Some(mut failed) = self.active_export.take() {
                    if let Err(cleanup_err) = failed.cleanup() {
                        return Err(format!("{write_err} ({cleanup_err})"));
                    }
                }
                Err(write_err)
            }
        }
    }

    pub fn finish_export(&mut self, id: u64) -> Result<PathBuf, String> {
        self.expire_stale();
        let matches = self
            .active_export
            .as_ref()
            .is_some_and(|session| session.id == id);
        if !matches {
            return Err("Export transaction is stale or not active.".to_string());
        }
        let mut session = self.active_export.take().unwrap();
        if session.written != session.declared_size {
            let cleanup_result = session.cleanup();
            return match cleanup_result {
                Ok(()) => Err("Export is incomplete.".to_string()),
                Err(cleanup_err) => Err(format!("Export is incomplete. ({cleanup_err})")),
            };
        }

        if let Some(file) = session.file.as_mut() {
            if let Err(err) = file.sync_all() {
                let reason = format!("Failed to flush exported file: {err}");
                return match session.cleanup() {
                    Ok(()) => Err(reason),
                    Err(cleanup_err) => Err(format!("{reason} ({cleanup_err})")),
                };
            }
        }
        // Close handle before rename (required on Windows).
        session.file = None;

        if let Err(err) = replace_file(&session.staging, &session.target) {
            let reason = format!("Failed to commit exported file: {err}");
            return match session.cleanup() {
                Ok(()) => Err(reason),
                Err(cleanup_err) => Err(format!("{reason} ({cleanup_err})")),
            };
        }

        session.committed = true;
        Ok(session.target.clone())
    }

    pub fn cancel_export(&mut self, id: u64) -> Result<(), String> {
        if self
            .active_export
            .as_ref()
            .is_some_and(|session| session.id == id)
        {
            if let Some(mut session) = self.active_export.take() {
                session.cleanup()?;
            }
        }
        Ok(())
    }

    pub fn shutdown(&mut self) {
        self.active_read = None;
        if let Some(mut session) = self.active_export.take() {
            let _ = session.cleanup();
        }
    }
}

fn create_staging_file(parent: &Path) -> Result<(PathBuf, File), String> {
    let mut random_bytes = [0u8; 16];
    getrandom::getrandom(&mut random_bytes)
        .map_err(|err| format!("Failed generating staging filename entropy: {err}"))?;
    let mut hex = String::with_capacity(32);
    for byte in random_bytes {
        hex.push_str(&format!("{byte:02x}"));
    }
    let staging = parent.join(format!(".nacelle-export-{hex}.tmp"));

    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(&staging)
        .map_err(|err| format!("Unable to create export staging file: {err}"))?;
    Ok((staging, file))
}

#[cfg(unix)]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::rename(from, to)
}

#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let from_wide: Vec<u16> = from.as_os_str().encode_wide().chain(Some(0)).collect();
    let to_wide: Vec<u16> = to.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: Both pointers are valid null-terminated UTF-16 buffers for the duration of the call.
    let ok = unsafe {
        MoveFileExW(
            from_wide.as_ptr(),
            to_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if ok == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(label: &str) -> PathBuf {
        let mut random = [0u8; 8];
        getrandom::getrandom(&mut random).unwrap();
        let hex: String = random.iter().map(|b| format!("{b:02x}")).collect();
        let dir = std::env::temp_dir().join(format!("nacelle-desktop-{label}-{hex}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reads_multi_chunk_capture_and_rejects_out_of_order_or_stale() {
        let dir = temp_dir("read");
        let path = dir.join("sample.nacelle-profile");
        let payload: Vec<u8> = (0..(CHUNK_BYTES * 2 + 123)).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &payload).unwrap();

        let mut state = TransferState::default();
        let meta = state.open_capture(path.clone()).unwrap();
        assert_eq!(meta.name, "sample.nacelle-profile");
        assert_eq!(meta.size as usize, payload.len());

        // Reject out-of-order offset and wrong ID.
        assert!(state.read_chunk(meta.id, 1).is_err());
        assert!(state.read_chunk(meta.id + 1, 0).is_err());

        let mut received = Vec::new();
        let mut offset = 0u64;
        while offset < meta.size {
            let chunk = state.read_chunk(meta.id, offset).unwrap();
            assert!(!chunk.is_empty() && chunk.len() <= CHUNK_BYTES);
            offset += chunk.len() as u64;
            received.extend_from_slice(&chunk);
        }
        assert_eq!(received, payload);
        // Handle closes automatically on EOF.
        assert!(state.read_chunk(meta.id, offset).is_err());

        // Reload re-authorizes the last opened file cleanly.
        let reloaded = state.reload_capture().unwrap();
        assert!(reloaded.id > meta.id);
        state.cancel_read(reloaded.id);
        assert!(state.read_chunk(reloaded.id, 0).is_err());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_directory_and_truncated_files() {
        let dir = temp_dir("invalid");
        let fake_dir = dir.join("dir.nacelle-profile");
        fs::create_dir(&fake_dir).unwrap();

        let tiny = dir.join("tiny.nacelle-profile");
        fs::write(&tiny, b"too short").unwrap();

        let mut state = TransferState::default();
        assert!(state.open_capture(fake_dir).is_err());
        assert!(state.open_capture(tiny).is_err());
        assert!(state.open_capture(dir.join("missing.nacelle-profile")).is_err());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn streams_export_to_staging_replaces_existing_and_cleans_up_on_cancel_or_error() {
        let dir = temp_dir("export");
        let target = dir.join("nacelle-trace.json");
        fs::write(&target, b"old-content").unwrap();

        let mut state = TransferState::default();
        let data = b"{\"traceEvents\":[],\"displayTimeUnit\":\"ms\"}";
        let id = state.begin_export(target.clone(), data.len() as u64).unwrap();
        // Concurrent export is rejected.
        assert!(state.begin_export(target.clone(), 4).is_err());
        // Out-of-order offset is rejected.
        assert!(state.write_export_chunk(id, 1, b"x").is_err());

        let mid = 16;
        assert_eq!(state.write_export_chunk(id, 0, &data[..mid]).unwrap(), mid as u64);
        assert_eq!(
            state.write_export_chunk(id, mid as u64, &data[mid..]).unwrap(),
            data.len() as u64
        );
        state.finish_export(id).unwrap();
        assert_eq!(fs::read(&target).unwrap(), data);

        // No staging sibling left behind.
        let leftover: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".nacelle-export-"))
            .collect();
        assert!(leftover.is_empty(), "expected zero staging files");

        // Cancel removes staging file.
        let cancel_id = state.begin_export(dir.join("cancel.json"), 10).unwrap();
        state.write_export_chunk(cancel_id, 0, b"12345").unwrap();
        state.cancel_export(cancel_id).unwrap();

        // Truncated finish removes staging file and errors.
        let trunc_id = state.begin_export(dir.join("trunc.json"), 10).unwrap();
        state.write_export_chunk(trunc_id, 0, b"12345").unwrap();
        assert!(state.finish_export(trunc_id).is_err());

        let leftover_after: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".nacelle-export-"))
            .collect();
        assert!(leftover_after.is_empty(), "expected cleanup after cancel/error");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn expires_idle_read_and_export_transfers() {
        let dir = temp_dir("idle");
        let path = dir.join("sample.nacelle-profile");
        fs::write(&path, vec![b'P'; 64]).unwrap();

        let mut state = TransferState::with_timeout(Duration::from_secs(120));
        let meta = state.open_capture(path).unwrap();
        let export_id = state.begin_export(dir.join("out.json"), 10).unwrap();

        let future = Instant::now() + Duration::from_secs(121);
        state.expire_stale_at(future);

        assert!(state.read_chunk(meta.id, 0).is_err());
        assert!(state.write_export_chunk(export_id, 0, b"hello").is_err());

        let leftover: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".nacelle-export-"))
            .collect();
        assert!(leftover.is_empty(), "expected expired export staging cleanup");

        fs::remove_dir_all(&dir).unwrap();
    }
}
