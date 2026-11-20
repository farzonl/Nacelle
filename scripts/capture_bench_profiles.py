#!/usr/bin/env python3
"""Capture .nacelle-profile traces for each shader in the benchmark suite (GitHub Actions only)."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def discover_benchmarks(bench_json: Path) -> list[dict]:
    with open(bench_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("suites") or raw.get("results") or raw.get("benchmarks") or []
    seen = set()
    specs = []
    for item in items:
        bench_id = item.get("bench_id") or item.get("benchId")
        if not bench_id or bench_id in seen:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", bench_id):
            continue
        seen.add(bench_id)
        specs.append({
            "id": bench_id,
            "name": item.get("bench_name") or item.get("benchName") or bench_id,
            "category": item.get("category") or "Other",
        })
    return specs


def normalize_platform_key(os_name: str, explicit_platform: str = "") -> str:
    raw = (explicit_platform or os_name or "default").strip().lower()
    key = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
    return key or "default"


def load_existing_platforms(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    if raw.get("schemaVersion") == 2 and isinstance(raw.get("platforms"), dict):
        return {
            k: v for k, v in raw["platforms"].items()
            if isinstance(k, str) and isinstance(v, dict) and isinstance(v.get("profiles"), list)
        }
    if raw.get("schemaVersion") == 1 and isinstance(raw.get("profiles"), list):
        os_label = raw.get("os") if isinstance(raw.get("os"), str) and raw.get("os") else "macOS"
        key = normalize_platform_key(os_label)
        return {
            key: {
                "os": os_label,
                "generatedAt": raw.get("generatedAt", ""),
                "commitSha": raw.get("commitSha", ""),
                "shortSha": raw.get("shortSha", ""),
                "simdWidth": raw.get("simdWidth", 8),
                "profiles": raw["profiles"],
            }
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture .nacelle-profile archives for benchmark shaders for GitHub Pages viewer preloading."
    )
    parser.add_argument("--bench-json", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--shader-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sha", default="")
    parser.add_argument("--os", dest="os_name", default="macOS")
    parser.add_argument("--platform", default="")
    parser.add_argument("--simd-width", type=int, default=8)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()

    if not args.worker.exists():
        raise SystemExit(f"Instrumented worker not found: {args.worker}")

    specs = discover_benchmarks(args.bench_json)
    if not specs:
        raise SystemExit(f"No benchmarks discovered in {args.bench_json}")

    platform_key = normalize_platform_key(args.os_name, args.platform)
    platform_dir = args.out_dir / platform_key
    platform_dir.mkdir(parents=True, exist_ok=True)
    backend_flag = f"nacelle-w{args.simd_width}"
    captured = []

    with tempfile.TemporaryDirectory(prefix="nacelle-prof-cache-") as tmp_root:
        for spec in specs:
            bench_id = spec["id"]
            filename = f"{bench_id}.nacelle-profile"
            rel_file = f"{platform_key}/{filename}"
            out_path = platform_dir / filename
            if out_path.exists():
                out_path.unlink()

            shader_cache = Path(tmp_root) / bench_id
            shader_cache.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["NACELLE_PROFILE_OUTPUT"] = str(out_path)
            env["NACELLE_CACHE_DIR"] = str(shader_cache)

            cmd = [
                str(args.worker),
                "--single-bench", bench_id,
                "--single-backend", backend_flag,
                "--shader-dir", str(args.shader_dir),
                "--iters", str(args.iters),
            ]
            print(f"[capture] {platform_key}/{bench_id} ({spec['category']}: {spec['name']})...", flush=True)
            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=args.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                print(f"  [skip] {bench_id} timed out after {args.timeout}s", flush=True)
                if out_path.exists():
                    out_path.unlink()
                continue

            if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
                stderr_preview = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
                tail = stderr_preview[-1] if stderr_preview else f"exit={proc.returncode}"
                print(f"  [skip] {bench_id} failed ({tail})", flush=True)
                if out_path.exists():
                    out_path.unlink()
                continue

            size_bytes = out_path.stat().st_size
            print(f"  [ok]   {rel_file} ({size_bytes} bytes)", flush=True)
            captured.append({
                "id": bench_id,
                "name": spec["name"],
                "category": spec["category"],
                "file": rel_file,
                "simdWidth": args.simd_width,
                "sizeBytes": size_bytes,
            })

    manifest_path = args.out_dir / "manifest.json"
    platforms = load_existing_platforms(manifest_path)
    platforms[platform_key] = {
        "os": args.os_name,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commitSha": args.sha,
        "shortSha": args.sha[:7] if args.sha else "",
        "simdWidth": args.simd_width,
        "profiles": captured,
    }
    ordered_platforms = {}
    for key in ("windows", "macos"):
        if key in platforms:
            ordered_platforms[key] = platforms[key]
    for key in sorted(platforms):
        if key not in ordered_platforms:
            ordered_platforms[key] = platforms[key]

    default_platform = "windows" if "windows" in ordered_platforms else next(iter(ordered_platforms))
    manifest = {
        "schemaVersion": 2,
        "defaultPlatform": default_platform,
        "platforms": ordered_platforms,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Captured {len(captured)}/{len(specs)} {args.os_name} benchmark shader profiles into {platform_dir}")


if __name__ == "__main__":
    main()
