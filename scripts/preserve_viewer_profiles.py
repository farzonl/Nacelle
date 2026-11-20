#!/usr/bin/env python3
"""Preserve preloaded viewer benchmark profiles from live GitHub Pages across deployments."""

import argparse
import json
import re
import urllib.request
from pathlib import Path

SAFE_PLATFORM_KEY = re.compile(r"^[a-z0-9_-]+$")
SAFE_PROFILE_PATH = re.compile(r"^(?:[a-z0-9_-]+/)?[A-Za-z0-9_.-]+\.nacelle-profile$")


def fetch_bytes(url: str, timeout: int = 20) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Nacelle-Pages-CI"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def normalize_platform_key(os_name: str) -> str:
    key = re.sub(r"[^a-z0-9_-]+", "-", (os_name or "macos").strip().lower()).strip("-")
    return key or "macos"


def extract_live_platforms(manifest: dict) -> dict[str, tuple[dict, bool]]:
    """Return {platform_key: (platform_manifest_dict, is_legacy_v1)}."""
    version = manifest.get("schemaVersion")
    if version == 2 and isinstance(manifest.get("platforms"), dict):
        result: dict[str, tuple[dict, bool]] = {}
        for key, val in manifest["platforms"].items():
            if isinstance(key, str) and SAFE_PLATFORM_KEY.fullmatch(key) and isinstance(val, dict) and isinstance(val.get("profiles"), list):
                result[key] = (val, False)
        return result
    if version == 1 and isinstance(manifest.get("profiles"), list):
        os_label = manifest.get("os") if isinstance(manifest.get("os"), str) and manifest.get("os") else "macOS"
        key = normalize_platform_key(os_label)
        return {
            key: ({
                "os": os_label,
                "generatedAt": manifest.get("generatedAt", ""),
                "commitSha": manifest.get("commitSha", ""),
                "shortSha": manifest.get("shortSha", ""),
                "simdWidth": manifest.get("simdWidth", 8),
                "profiles": manifest["profiles"],
            }, True)
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-url", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--exclude-platform", action="append", default=[])
    args = parser.parse_args()

    excluded = {p.strip().lower() for p in args.exclude_platform if p and p.strip()}
    base_url = args.pages_url.rstrip("/") + "/viewer/profiles"
    manifest_bytes = fetch_bytes(f"{base_url}/manifest.json")
    if not manifest_bytes:
        print("Info: no live viewer/profiles/manifest.json found on Pages; skipping preservation.")
        return

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        print(f"Warning: invalid live manifest.json ({exc}); skipping preservation.")
        return

    if not isinstance(manifest, dict):
        print("Warning: unexpected manifest.json schema; skipping preservation.")
        return

    live_platforms = extract_live_platforms(manifest)
    if not live_platforms:
        print("Warning: unexpected manifest.json schema; skipping preservation.")
        return

    preserved_platforms: dict[str, dict] = {}
    total_kept = 0
    for key, (plat, is_legacy) in live_platforms.items():
        if key in excluded:
            print(f"Info: skipping excluded platform '{key}' (will be freshly captured).")
            continue
        kept = []
        for entry in plat["profiles"]:
            if not isinstance(entry, dict):
                continue
            remote_file = entry.get("file")
            if not isinstance(remote_file, str) or not SAFE_PROFILE_PATH.fullmatch(remote_file) or ".." in remote_file:
                continue
            data = fetch_bytes(f"{base_url}/{remote_file}")
            if not data:
                continue
            local_file = f"{key}/{Path(remote_file).name}" if is_legacy else remote_file
            dest = args.out_dir / local_file
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            updated_entry = dict(entry)
            updated_entry["file"] = local_file
            kept.append(updated_entry)

        if kept:
            updated_plat = dict(plat)
            updated_plat["profiles"] = kept
            preserved_platforms[key] = updated_plat
            total_kept += len(kept)

    if not preserved_platforms:
        print("Info: no preloaded profile files downloaded; skipping manifest.")
        return

    ordered_platforms = {}
    for key in ("windows", "macos"):
        if key in preserved_platforms:
            ordered_platforms[key] = preserved_platforms[key]
    for key in sorted(preserved_platforms):
        if key not in ordered_platforms:
            ordered_platforms[key] = preserved_platforms[key]

    default_platform = "windows" if "windows" in ordered_platforms else next(iter(ordered_platforms))
    out_manifest = {
        "schemaVersion": 2,
        "defaultPlatform": default_platform,
        "platforms": ordered_platforms,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"Preserved {total_kept} preloaded benchmark profiles across {list(ordered_platforms.keys())} from {base_url}")


if __name__ == "__main__":
    main()
