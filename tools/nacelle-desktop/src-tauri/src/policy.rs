use std::fs::File;
use std::path::Path;
use url::Url;

pub const MAX_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_EXPORT_BYTES: u64 = 256 * 1024 * 1024;
pub const MIN_ARCHIVE_BYTES: u64 = 22;
pub const CHUNK_BYTES: usize = 64 * 1024;
pub const IDLE_TIMEOUT_SECS: u64 = 120;
pub const ALLOWED_EXPORT_NAMES: [&str; 2] = ["nacelle-trace.json", "nacelle-speedscope.json"];

/// Allows navigation exclusively to the bundled Tauri custom-protocol origin
/// and empty/srcdoc iframe initializations. Blocks all external, file, and
/// script/data top-level navigations.
pub fn is_allowed_navigation(url: &Url) -> bool {
    match url.scheme() {
        "about" => matches!(url.path(), "blank" | "srcdoc"),
        "tauri" => {
            url.host_str() == Some("localhost")
                && url.port().is_none()
                && url.username().is_empty()
                && url.password().is_none()
        }
        "http" | "https" => {
            url.host_str() == Some("tauri.localhost")
                && url.port().is_none()
                && url.username().is_empty()
                && url.password().is_none()
        }
        _ => false,
    }
}

pub fn validate_export_name(name: &str) -> Result<&'static str, String> {
    for allowed in ALLOWED_EXPORT_NAMES {
        if name == allowed {
            return Ok(allowed);
        }
    }
    Err("Export filename is not allowlisted.".to_string())
}

pub fn validate_export_size(size: u64) -> Result<u64, String> {
    if size > MAX_EXPORT_BYTES {
        return Err("Export exceeds the 256 MiB limit.".to_string());
    }
    Ok(size)
}

pub fn validate_capture_filename(path: &Path) -> Result<String, String> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Selected capture must have a valid UTF-8 filename.".to_string())?;
    if name.is_empty() || name.len() > 256 || name.contains('/') || name.contains('\\') {
        return Err("Selected capture filename is invalid.".to_string());
    }
    if !name.to_ascii_lowercase().ends_with(".nacelle-profile") {
        return Err("Only .nacelle-profile archives are supported.".to_string());
    }
    Ok(name.to_string())
}

pub fn validate_open_capture_file(file: &File) -> Result<u64, String> {
    let metadata = file
        .metadata()
        .map_err(|err| format!("Unable to inspect selected capture: {err}"))?;
    if !metadata.is_file() {
        return Err("Selected capture is not a regular file.".to_string());
    }
    let size = metadata.len();
    if size < MIN_ARCHIVE_BYTES {
        return Err("Selected capture is truncated (smaller than a ZIP header).".to_string());
    }
    if size > MAX_ARCHIVE_BYTES {
        return Err("Selected capture exceeds the 256 MiB limit.".to_string());
    }
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_only_bundled_app_origins() {
        let allowed = [
            "tauri://localhost",
            "tauri://localhost/index.html",
            "tauri://localhost/speedscope/index.html?load=1#title=test",
            "http://tauri.localhost/index.html",
            "https://tauri.localhost/speedscope/index.html",
            "about:blank",
            "about:srcdoc",
        ];
        for raw in allowed {
            let parsed = Url::parse(raw).expect("valid test url");
            assert!(is_allowed_navigation(&parsed), "expected allow for {raw}");
        }

        let denied = [
            "https://example.com/",
            "http://example.com/profile.nacelle-profile",
            "http://127.0.0.1:8080/",
            "http://localhost:5173/",
            "http://tauri.localhost:8080/index.html",
            "http://user@tauri.localhost/index.html",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<h1>x</h1>",
            "ws://127.0.0.1:9000/",
            "wss://example.com/",
            "blob:http://tauri.localhost/1234",
            "about:config",
        ];
        for raw in denied {
            let parsed = Url::parse(raw).expect("valid test url");
            assert!(!is_allowed_navigation(&parsed), "expected deny for {raw}");
        }
    }

    #[test]
    fn enforces_export_allowlist_and_bounds() {
        assert_eq!(validate_export_name("nacelle-trace.json").unwrap(), "nacelle-trace.json");
        assert_eq!(
            validate_export_name("nacelle-speedscope.json").unwrap(),
            "nacelle-speedscope.json"
        );
        for bad in [
            "../nacelle-trace.json",
            "subdir/nacelle-trace.json",
            "C:\\temp\\nacelle-trace.json",
            "arbitrary.json",
            "",
        ] {
            assert!(validate_export_name(bad).is_err(), "expected rejection for {bad}");
        }
        assert!(validate_export_size(0).is_ok());
        assert!(validate_export_size(MAX_EXPORT_BYTES).is_ok());
        assert!(validate_export_size(MAX_EXPORT_BYTES + 1).is_err());
    }

    #[test]
    fn validates_capture_extension_and_basename() {
        assert_eq!(
            validate_capture_filename(Path::new("/tmp/static.nacelle-profile")).unwrap(),
            "static.nacelle-profile"
        );
        assert!(validate_capture_filename(Path::new("/tmp/not-a-profile.json")).is_err());
        assert!(validate_capture_filename(Path::new("/tmp/")).is_err());
    }
}
