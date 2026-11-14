#!/usr/bin/env python3
"""Parse nacelle-bench JSON results and update GitHub Pages benchmark history."""

import argparse
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def fetch_json_url(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Nacelle-Bench-CI"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def load_existing_history(pages_url: str | None, cache_file: Path | None) -> list:
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    return loaded
        except Exception as exc:
            print(f"Warning: failed to read cached bench history {cache_file}: {exc}")

    if pages_url:
        loaded = fetch_json_url(pages_url.rstrip("/") + "/data/bench-history.json")
        if isinstance(loaded, list):
            return loaded

    return []


def get_num(r: dict, camel: str, snake: str, default: float = 0.0) -> float:
    val = r.get(camel)
    if val is None:
        val = r.get(snake, default)
    return float(val)


def normalize_result(r: dict) -> dict:
    return {
        "benchId": r.get("benchId") or r.get("bench_id", "bench_raymarch"),
        "benchName": r.get("benchName") or r.get("bench_name", "Raymarcher"),
        "category": r.get("category", "Compute"),
        "backend": r.get("backend", ""),
        "shaderFormat": r.get("shaderFormat") or r.get("shader_format", ""),
        "simdWidth": int(get_num(r, "simdWidth", "simd_width", 0)),
        "ran": bool(r.get("ran", False)),
        "compileMs": round(get_num(r, "compileMs", "compile_ms"), 3),
        "firstDispatchMs": round(get_num(r, "firstDispatchMs", "first_dispatch_ms"), 3),
        "coldMs": round(get_num(r, "coldMs", "cold_ms"), 3),
        "meanMs": round(get_num(r, "meanMs", "mean_ms"), 3),
        "medianMs": round(get_num(r, "medianMs", "median_ms"), 3),
        "minMs": round(get_num(r, "minMs", "min_ms"), 3),
        "shortJobMs": round(get_num(r, "shortJobMs", "short_job_ms"), 3),
        "longJobMs": round(get_num(r, "longJobMs", "long_job_ms"), 3),
        "rssBeforeMb": round(get_num(r, "rssBeforeMb", "rss_before_mb"), 2),
        "rssAfterCompileMb": round(get_num(r, "rssAfterCompileMb", "rss_after_compile_mb"), 2),
        "rssAfterRunMb": round(get_num(r, "rssAfterRunMb", "rss_after_run_mb"), 2),
        "rssDeltaMb": round(get_num(r, "rssDeltaMb", "rss_delta_mb"), 2),
        "peakRssMb": round(get_num(r, "peakRssMb", "peak_rss_mb"), 2),
        "checksum": round(float(r.get("checksum", 0.0)), 2),
        "status": r.get("status", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-json", required=True, type=Path)
    parser.add_argument("--site-src", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--message", default="")
    parser.add_argument("--author", default="")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--pages-url", default="")
    parser.add_argument("--history-cache", type=Path, default=None)
    args = parser.parse_args()

    with open(args.bench_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    results = raw.get("results")
    if results is None:
        results = raw.get("benchmarks", [])
    suites = raw.get("suites")
    if suites is None:
        suites = results

    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    entry = {
        "version": int(raw.get("version", 1)),
        "sha": args.sha,
        "shortSha": args.sha[:7],
        "message": args.message.splitlines()[0][:120] if args.message else "",
        "author": args.author,
        "timestamp": ts,
        "runUrl": args.run_url,
        "os": "Windows",
        "overallWinner": raw.get("overall_winner") or raw.get("overallWinner", ""),
        "overallGeomeanMs": raw.get("overall_geomean_ms") or raw.get("overallGeomeanMs", {}),
        "categoryWinners": raw.get("category_winners") or raw.get("categoryWinners", []),
        "regimeWinners": raw.get("regime_winners") or raw.get("regimeWinners", []),
        "results": [normalize_result(r) for r in results],
        "suites": [normalize_result(r) for r in suites],
    }

    history = load_existing_history(args.pages_url, args.history_cache)
    history = [h for h in history if h.get("sha") != args.sha]
    history.append(entry)
    history.sort(key=lambda h: h.get("timestamp", ""))

    if args.history_cache:
        args.history_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_cache, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.site_src.iterdir():
        dst = args.out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "bench-history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Preserve compliance history.json, latest.json, and badge.json from live Pages if not already in _site/data
    if args.pages_url:
        base_url = args.pages_url.rstrip("/")
        if not (data_dir / "history.json").exists():
            compliance = fetch_json_url(base_url + "/data/history.json")
            if isinstance(compliance, list):
                with open(data_dir / "history.json", "w", encoding="utf-8") as f:
                    json.dump(compliance, f, indent=2)
        for fname in ("latest.json", "badge.json"):
            if not (data_dir / fname).exists():
                obj = fetch_json_url(base_url + f"/data/{fname}")
                if isinstance(obj, dict):
                    with open(data_dir / fname, "w", encoding="utf-8") as f:
                        json.dump(obj, f, indent=2)

    ran_backends = [f"{r['backend']}={r['medianMs']}ms" for r in entry["results"] if r["ran"]]
    print(f"Recorded benchmark {entry['shortSha']}: {', '.join(ran_backends)}. Suites: {len(entry['suites'])}. History length: {len(history)}")


if __name__ == "__main__":
    main()
