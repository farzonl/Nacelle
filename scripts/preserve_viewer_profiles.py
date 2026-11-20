#!/usr/bin/env python3
"""Preserve preloaded viewer benchmark profiles from live GitHub Pages across deployments."""

import argparse
import json
import re
import urllib.request
from pathlib import Path

SAFE_PROFILE_FILE = re.compile(r"^[A-Za-z0-9_.-]+\.nacelle-profile$")


def fetch_bytes(url: str, timeout: int = 20) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Nacelle-Pages-CI"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-url", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

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

    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1 or not isinstance(manifest.get("profiles"), list):
        print("Warning: unexpected manifest.json schema; skipping preservation.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for entry in manifest["profiles"]:
        if not isinstance(entry, dict):
            continue
        fname = entry.get("file")
        if not isinstance(fname, str) or not SAFE_PROFILE_FILE.fullmatch(fname) or ".." in fname:
            continue
        data = fetch_bytes(f"{base_url}/{fname}")
        if not data:
            continue
        (args.out_dir / fname).write_bytes(data)
        kept.append(entry)

    if not kept:
        print("Info: no preloaded profile files downloaded; skipping manifest.")
        return

    manifest["profiles"] = kept
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Preserved {len(kept)} preloaded benchmark profiles from {base_url}")


if __name__ == "__main__":
    main()
