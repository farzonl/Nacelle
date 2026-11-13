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
    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def get_num(r: dict, camel: str, snake: str) -> float:
        val = r.get(camel)
        if val is None:
            val = r.get(snake, 0.0)
        return float(val)

    entry = {
        "sha": args.sha,
        "shortSha": args.sha[:7],
        "message": args.message.splitlines()[0][:120] if args.message else "",
        "author": args.author,
        "timestamp": ts,
        "runUrl": args.run_url,
        "os": "Windows",
        "results": [
            {
                "backend": r.get("backend", ""),
                "shaderFormat": r.get("shaderFormat") or r.get("shader_format", ""),
                "ran": bool(r.get("ran", False)),
                "compileMs": round(get_num(r, "compileMs", "compile_ms"), 3),
                "medianMs": round(get_num(r, "medianMs", "median_ms"), 3),
                "minMs": round(get_num(r, "minMs", "min_ms"), 3),
                "checksum": round(float(r.get("checksum", 0.0)), 2),
                "status": r.get("status", ""),
            }
            for r in results
        ],
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

    # Preserve compliance history.json from live Pages if not already in _site/data
    if args.pages_url and not (data_dir / "history.json").exists():
        compliance = fetch_json_url(args.pages_url.rstrip("/") + "/data/history.json")
        if isinstance(compliance, list):
            with open(data_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(compliance, f, indent=2)

    ran_backends = [f"{r['backend']}={r['medianMs']}ms" for r in entry["results"] if r["ran"]]
    print(f"Recorded benchmark {entry['shortSha']}: {', '.join(ran_backends)}. History length: {len(history)}")


if __name__ == "__main__":
    main()
