#!/usr/bin/env python3
"""Cross-platform CTest for nacelle-bench progress, skip, checkpoint/resume, and timeout."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


def run(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nacelle-bench", required=True)
    args = ap.parse_args()

    bench_exe = os.path.abspath(args.nacelle_bench)
    if not os.path.exists(bench_exe):
        print(f"Missing nacelle-bench binary: {bench_exe}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "bench.ckpt")
        out_json = os.path.join(tmp, "bench.json")

        # 1. Test progress bar, --skip, --checkpoint, --timeout, and --json
        res = run([
            bench_exe,
            "--filter", "matmul",
            "--simd-width=4",
            "--iters", "1",
            "--allow-partial",
            "--timeout", "30",
            "--skip", "linalg_matmul@nacelle-w4",
            "--checkpoint", ckpt,
            "--json", out_json,
        ])
        if res.returncode != 0:
            print("Initial run failed:\n" + res.stdout)
            return 1

        for expected in (
            "START matmul @ nacelle-w4",
            "STOP  matmul @ nacelle-w4",
            "SKIP  linalg_matmul @ nacelle-w4 (--skip)",
            "100%",
        ):
            if expected not in res.stdout:
                print(f"Missing expected output {expected!r}:\n{res.stdout}")
                return 1

        with open(out_json, "r", encoding="utf-8") as f:
            report = json.load(f)
        statuses = {(r["bench_id"], r["backend"]): r["status"] for r in report["suites"]}
        if statuses.get(("linalg_matmul", "Nacelle (W=4)")) != "SKIPPED (user)":
            print(f"Unexpected linalg_matmul status: {statuses}")
            return 1

        # 2. Test --resume with an interrupted task (dangling START without DONE)
        with open(ckpt, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        # Replace linalg_matmul @ nacelle-w4 DONE/RESULT with a dangling START
        filtered = [
            ln for ln in lines
            if not (ln.startswith("RESULT\tlinalg_matmul\tnacelle-w4\t") or ln == "DONE\tlinalg_matmul\tnacelle-w4")
        ]
        filtered.append("START\tlinalg_matmul\tnacelle-w4")
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write("\n".join(filtered) + "\n")

        res2 = run([
            bench_exe,
            "--filter", "matmul",
            "--simd-width=4",
            "--iters", "1",
            "--allow-partial",
            "--checkpoint", ckpt,
            "--resume",
            "--json", out_json,
        ])
        if res2.returncode != 0:
            print("Resume run failed:\n" + res2.stdout)
            return 1

        for expected in (
            "SKIP  matmul @ nacelle-w4 (already in checkpoint)",
            "SKIP  linalg_matmul @ nacelle-w4 (hung/interrupted in prior run)",
        ):
            if expected not in res2.stdout:
                print(f"Missing expected resume output {expected!r}:\n{res2.stdout}")
                return 1

        with open(out_json, "r", encoding="utf-8") as f:
            report2 = json.load(f)
        statuses2 = {(r["bench_id"], r["backend"]): r["status"] for r in report2["suites"]}
        if statuses2.get(("linalg_matmul", "Nacelle (W=4)")) != "SKIPPED (hung in prior run)":
            print(f"Unexpected resumed linalg_matmul status: {statuses2}")
            return 1

        # 3. Test --timeout kill recovery on POSIX using a mock sleeping vulkan worker
        if os.name != "nt":
            mock_dir = os.path.join(tmp, "mock_bin")
            os.makedirs(mock_dir, exist_ok=True)
            bin_dir = os.path.dirname(bench_exe)
            shutil.copy2(bench_exe, os.path.join(mock_dir, os.path.basename(bench_exe)))
            for w in ("bench-worker-nacelle", "bench-worker-metal"):
                src = os.path.join(bin_dir, w)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(mock_dir, w))
            vk_mock = os.path.join(mock_dir, "bench-worker-vulkan")
            with open(vk_mock, "w", encoding="utf-8") as f:
                f.write('#!/bin/sh\nif [ "$1" = "--list-devices" ]; then exit 0; fi\nsleep 10\n')
            os.chmod(vk_mock, 0o755)

            res3 = run([
                os.path.join(mock_dir, os.path.basename(bench_exe)),
                "--filter", "matmul",
                "--simd-width=4",
                "--iters", "1",
                "--allow-partial",
                "--timeout", "1",
            ])
            if res3.returncode != 0:
                print("Timeout recovery run failed:\n" + res3.stdout)
                return 1
            if "TIMED OUT (1s)" not in res3.stdout:
                print(f"Expected TIMED OUT (1s) in output:\n{res3.stdout}")
                return 1

    print("nacelle_bench_harness_test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
