#!/usr/bin/env python3
"""Parse llvm-lit JSON results and update GitHub Pages compliance history."""

import argparse
import json
import os
import shutil
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def category_for_test(test_name: str) -> str:
    # Example: "OffloadTest-d3d12 :: Feature/HLSLLib/mad.fp64.test"
    rel = test_name.split("::", 1)[-1].strip()
    parts = rel.split("/")
    if len(parts) >= 2 and parts[0] in ("Feature", "Basic", "Tools"):
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def parse_lit_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    counts = {"PASS": 0, "FAIL": 0, "XFAIL": 0, "XPASS": 0, "UNSUPPORTED": 0}
    categories = defaultdict(lambda: {"passed": 0, "failed": 0, "xfail": 0, "unsupported": 0})
    failures = []

    for t in tests:
        name = t.get("name", "")
        code = t.get("code", "UNKNOWN")
        cat = category_for_test(name)

        if code in ("PASS", "FLAKYPASS"):
            counts["PASS"] += 1
            categories[cat]["passed"] += 1
        elif code in ("XFAIL",):
            counts["XFAIL"] += 1
            categories[cat]["xfail"] += 1
        elif code in ("UNSUPPORTED",):
            counts["UNSUPPORTED"] += 1
            categories[cat]["unsupported"] += 1
        else:
            counts["FAIL"] += 1
            categories[cat]["failed"] += 1
            failures.append({
                "name": name.split("::", 1)[-1].strip(),
                "category": cat,
                "code": code,
                "elapsed": round(float(t.get("elapsed", 0.0)), 3),
            })

    total = len(tests)
    executed = counts["PASS"] + counts["FAIL"] + counts["XFAIL"]
    pass_rate = round((100.0 * counts["PASS"] / total), 2) if total > 0 else 0.0

    cat_list = []
    for cat, c in sorted(categories.items(), key=lambda kv: (-kv[1]["failed"], kv[0])):
        cat_total = c["passed"] + c["failed"] + c["xfail"] + c["unsupported"]
        cat_rate = round((100.0 * c["passed"] / cat_total), 1) if cat_total > 0 else 0.0
        cat_list.append({
            "name": cat,
            "passed": c["passed"],
            "failed": c["failed"],
            "xfail": c["xfail"],
            "unsupported": c["unsupported"],
            "passRate": cat_rate,
        })

    return {
        "total": total,
        "executed": executed,
        "passed": counts["PASS"],
        "failed": counts["FAIL"],
        "xfail": counts["XFAIL"],
        "unsupported": counts["UNSUPPORTED"],
        "passRate": pass_rate,
        "elapsed": round(float(data.get("elapsed", 0.0)), 2),
        "categories": cat_list,
        "failures": sorted(failures, key=lambda x: x["name"]),
    }


def load_existing_history(pages_url: str | None, cache_file: Path | None) -> list:
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    return loaded
        except Exception as exc:
            print(f"Warning: failed to read cached history {cache_file}: {exc}")

    if pages_url:
        url = pages_url.rstrip("/") + "/data/history.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Nacelle-Compliance-CI"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                loaded = json.loads(resp.read().decode("utf-8"))
                if isinstance(loaded, list):
                    return loaded
        except Exception as exc:
            print(f"Info: no prior history fetched from {url} ({exc}); starting fresh.")

    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lit-json", required=True, type=Path)
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

    summary = parse_lit_json(args.lit_json)
    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    entry = {
        "sha": args.sha,
        "shortSha": args.sha[:7],
        "message": args.message.splitlines()[0][:120] if args.message else "",
        "author": args.author,
        "timestamp": ts,
        "runUrl": args.run_url,
        "total": summary["total"],
        "executed": summary["executed"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "xfail": summary["xfail"],
        "unsupported": summary["unsupported"],
        "passRate": summary["passRate"],
        "elapsed": summary["elapsed"],
        "categories": summary["categories"],
    }

    history = load_existing_history(args.pages_url, args.history_cache)
    # Replace if same SHA already exists, otherwise append
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
    with open(data_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(data_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump({**entry, "failures": summary["failures"]}, f, indent=2)
    with open(data_dir / "badge.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "schemaVersion": 1,
                "label": "Offload D3D12 Suite",
                "message": f"{entry['passed']} Passed ({entry['passRate']}%)",
                "color": "brightgreen" if entry["passRate"] >= 85.0 else "green",
            },
            f,
            indent=2,
        )

    # Preserve Windows benchmark history if already deployed on Pages
    if args.pages_url and not (data_dir / "bench-history.json").exists():
        try:
            req = urllib.request.Request(
                args.pages_url.rstrip("/") + "/data/bench-history.json",
                headers={"User-Agent": "Nacelle-Compliance-CI"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                bench_hist = json.loads(resp.read().decode("utf-8"))
                if isinstance(bench_hist, list):
                    with open(data_dir / "bench-history.json", "w", encoding="utf-8") as f:
                        json.dump(bench_hist, f, indent=2)
        except Exception:
            pass

    print(
        f"Recorded {entry['shortSha']}: {entry['passed']}/{entry['total']} passed "
        f"({entry['passRate']}%, {entry['executed']} executed), {entry['failed']} failed, "
        f"{entry['xfail']} xfail, {entry['unsupported']} unsupported. History length: {len(history)}"
    )


if __name__ == "__main__":
    main()
