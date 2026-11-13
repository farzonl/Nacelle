#!/usr/bin/env python3
"""Minimal portable lit-style RUN + FileCheck test runner for Nacelle."""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile


def expand_substitutions(cmd: str, subs: dict[str, str]) -> str:
    # Sort longest keys first to avoid prefix collisions
    for k in sorted(subs.keys(), key=len, reverse=True):
        cmd = cmd.replace(k, subs[k])
    return cmd


def run_pipeline(line: str, env: dict[str, str], cwd: str) -> None:
    stages = [s.strip() for s in line.split("|")]
    prev_stdout = None
    for idx, stage in enumerate(stages):
        parts = shlex.split(stage)
        if not parts:
            raise RuntimeError(f"Empty pipeline stage in: {line}")
        stdout_mode = subprocess.PIPE if idx + 1 < len(stages) else subprocess.PIPE
        proc = subprocess.run(
            parts,
            input=prev_stdout,
            stdout=stdout_mode,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"Command failed ({proc.returncode}): {stage}\n")
            if proc.stdout:
                sys.stderr.write(proc.stdout.decode("utf-8", errors="replace"))
            if proc.stderr:
                sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
            sys.exit(proc.returncode)
        prev_stdout = proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nacelle-compile", required=True)
    ap.add_argument("--dxc", required=True)
    ap.add_argument("--filecheck", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("test_file")
    args = ap.parse_args()

    test_path = os.path.abspath(args.test_file)
    with open(test_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    run_lines = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        for prefix in ("// RUN:", "# RUN:"):
            if stripped.startswith(prefix):
                cmd = stripped[len(prefix):].strip()
                while cmd.endswith("\\") and i + 1 < len(lines):
                    i += 1
                    nxt = lines[i].strip()
                    for p in ("// RUN:", "# RUN:", "//", "#"):
                        if nxt.startswith(p):
                            nxt = nxt[len(p):].strip()
                            break
                    cmd = cmd[:-1].strip() + " " + nxt
                run_lines.append(cmd)
                break
        i += 1

    if not run_lines:
        sys.stderr.write(f"No RUN lines found in {test_path}\n")
        return 1

    with tempfile.TemporaryDirectory(prefix="nacelle_lit_") as tmpdir:
        t_prefix = os.path.join(tmpdir, "t")
        subs = {
            "%nacelle_compile": shlex.quote(args.nacelle_compile),
            "%dxc": shlex.quote(args.dxc),
            "%FileCheck": shlex.quote(args.filecheck),
            "%split_file": shlex.quote(args.split_file),
            "%s": shlex.quote(test_path),
            "%S": shlex.quote(os.path.dirname(test_path)),
            "%t": shlex.quote(t_prefix),
        }
        env = os.environ.copy()
        for raw_cmd in run_lines:
            expanded = expand_substitutions(raw_cmd, subs)
            run_pipeline(expanded, env, tmpdir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
