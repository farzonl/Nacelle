#!/usr/bin/env python3
"""Paired coarse-capture measurements; never certify uncontrolled/noisy runs."""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEBUG_INPUTS = {
    "src/compiler/native_debug.cpp",
    "src/compiler/native_debug.h",
    "include/nacelle/native_debug.h",
    "src/compiler/source_debug.cpp",
    "src/compiler/source_debug_compiler.cpp",
    "src/compiler/source_debug.h",
    "include/nacelle/source_debug.h",
}
WORKLOADS = {
    "bench_raymarch": ("Compute", ("bench_raymarch.dxil",)),
    "textured_filter": ("Pixel", ("fullscreen_quad_vs.dxil", "textured_filter.dxil")),
}
VARIANTS = ("baseline", "off", "idle", "active")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_identity(build: Path, enabled: bool) -> dict:
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.fullmatch(r"([^/#][^:]*):[^=]*=(.*)", line)
        if match:
            cache[match[1]] = match[2]
    if cache["NACELLE_ENABLE_INSTRUMENTATION"] != ("ON" if enabled else "OFF"):
        raise ValueError(f"{build}: unexpected instrumentation configuration")
    if cache["CMAKE_BUILD_TYPE"] != "Release":
        raise ValueError(f"{build}: a single-config Release build is required")
    if cache.get("NACELLE_PGO_MODE", "OFF") != "OFF":
        raise ValueError(f"{build}: this acceptance configuration requires PGO OFF")
    inputs = {}
    stamp = ""
    for line in (build / "generated/compiler_hash_inputs.txt").read_text().splitlines():
        if line.startswith("LLVM_PACKAGE_VERSION="):
            label, digest = line.split("=", 1)
            stamp += line + "\n"
        else:
            path = Path(line).resolve()
            digest = sha256(path)
            original_label = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
            stamp += f"{original_label}:{digest}\n"
            label = f"generated/{path.name}" if path.parent == build / "generated" else original_label
        if label in inputs:
            raise ValueError(f"{build}: duplicate compiler input {label}")
        inputs[label] = digest
    header = (build / "generated/compiler_version.h").read_text()
    match = re.search(r'kCompilerBuildHash\[\] = "([0-9a-f]{64})";', header)
    if not match or match[1] != hashlib.sha256(stamp.encode()).hexdigest():
        raise ValueError(f"{build}: stale compiler hash; rebuild before measuring")
    flags = {key: cache.get(key, "") for key in (
        "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE",
        "LLVM_DIR", "MLIR_DIR", "NACELLE_ENABLE_THIN_LTO", "NACELLE_PGO_MODE",
    )}
    return {"compiler_hash": match[1], "inputs": inputs, "flags": flags}


def compare_builds(off: dict, on: dict) -> str:
    if off["flags"] != on["flags"]:
        raise ValueError("ON/OFF compiler paths, toolchains or build flags differ")
    extra = set(on["inputs"]) - set(off["inputs"])
    if extra != DEBUG_INPUTS or set(off["inputs"]) & DEBUG_INPUTS:
        raise ValueError("Unexpected ON/OFF compiler input difference")
    ordinary = {key: value for key, value in on["inputs"].items() if key not in DEBUG_INPUTS}
    if ordinary != off["inputs"]:
        raise ValueError("ON/OFF ordinary compiler, ABI or runtime-bitcode inputs differ")
    return hashlib.sha256(json.dumps(ordinary, sort_keys=True).encode()).hexdigest()


def parse_result(stdout: str, bench: str, simd: int) -> dict:
    blocks = re.findall(r"<<<BENCH_JSON_BEGIN>>>\s*(.*?)\s*<<<BENCH_JSON_END>>>", stdout, re.S)
    if len(blocks) != 1:
        raise ValueError("Worker must return exactly one benchmark JSON block")
    result = json.loads(blocks[0])
    if not isinstance(result, dict) or (
        result.get("ran") is not True or result.get("status") != "OK"
        or result.get("bench_id") != bench or result.get("simd_width") != simd
        or result.get("category") != WORKLOADS[bench][0] or result.get("shader_format") != "DXIL"
    ):
        raise ValueError(f"{bench}: missing, failed or mismatched worker result")
    for key in ("median_ms", "checksum"):
        value = result.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{bench}: invalid {key}")
    if result["median_ms"] <= 0:
        raise ValueError(f"{bench}: nonpositive timing")
    return result


def paired_interval(baseline: list[float], measured: list[float], seed: int, resamples: int = 20000) -> dict:
    if len(baseline) != len(measured) or not baseline:
        raise ValueError("Paired measurements must be nonempty and have identical lengths")
    if any(not math.isfinite(x) or x <= 0 for x in baseline + measured):
        raise ValueError("Paired timings must be finite and positive")
    ratios = [value / reference for reference, value in zip(baseline, measured)]
    rng = random.Random(seed)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(resamples))
    return {
        "paired_median_ratio": statistics.median(ratios),
        "ratio_ci95": [boot[int(resamples * 0.025)], boot[math.ceil(resamples * 0.975) - 1]],
        "paired_ratio_range": [min(ratios), max(ratios)],
        "reference_median_ms": statistics.median(baseline),
        "measured_median_ms": statistics.median(measured),
    }


def verdict(interval: dict, limit: float) -> str:
    low, high = interval["ratio_ci95"]
    if high <= 1 + limit:
        return "pass"
    if low > 1 + limit:
        return "fail"
    return "inconclusive"


def conclusion(results: dict, controlled_host: bool) -> str:
    if set(results) != set(WORKLOADS) or any(
        set(variants) != {"off", "idle", "active"} for variants in results.values()
    ):
        raise ValueError("Both workloads and all OFF/idle/active acceptance checks are required")
    if not controlled_host:
        return "inconclusive"
    states = [item["verdict"] for variants in results.values() for item in variants.values()]
    if any(state not in ("pass", "fail", "inconclusive") for state in states):
        raise ValueError("Missing or invalid acceptance checks")
    return "fail" if "fail" in states else "inconclusive" if "inconclusive" in states else "pass"


def inspect_capture(path: Path, inspector: Path, compiler_hash: str, timeout: int) -> dict:
    proc = subprocess.run([str(inspector), str(path)], capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"Native capture validation failed: {proc.stdout}\n{proc.stderr}")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    if (manifest["complete"] is not True or manifest["droppedEvents"] != 0
            or manifest["recordedEvents"] <= 0 or manifest["schemaVersion"] != 1
            or manifest["build"]["compilerHash"] != compiler_hash):
        raise ValueError("Capture is incomplete, empty, not coarse-only or has mismatched compiler identity")
    return manifest


def run_worker(worker, variant, bench, number, args, env, compiler_hash):
    capture = args.output_dir / f"{bench}-{number}-{variant}.nacelle-profile"
    child_env = dict(env)
    if variant == "active":
        child_env["NACELLE_PROFILE_OUTPUT"] = str(capture)
    command = [
        str(worker), "--single-bench", bench, "--single-backend", f"nacelle-w{args.simd}",
        "--shader-dir", str(args.shader_dir), "--width", str(args.width),
        "--height", str(args.height), "--steps", str(args.steps), "--iters", str(args.iters),
    ]
    proc = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=args.timeout)
    log = args.output_dir / f"{bench}-{number}-{variant}.log"
    log.write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"{variant}/{bench}: worker exited {proc.returncode}; see {log}")
    record = {
        "variant": variant, "bench": bench, "round": number, "command": command,
        "result": parse_result(proc.stdout, bench, args.simd),
    }
    if variant == "active":
        record["capture"] = inspect_capture(capture, args.inspector, compiler_hash, args.timeout)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("off-build", "on-build", "baseline-off-worker", "shader-dir", "inspector", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--baseline-revision", required=True, help="Git revision used to build the baseline OFF worker")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--iters", type=int, default=240)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--simd", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--controlled-host", action="store_true",
                        help="Attest exclusive host use during timing; never set on a shared development host")
    args = parser.parse_args()
    if args.rounds < 24 or args.rounds % 24 or min(
        args.iters, args.width, args.height, args.steps, args.timeout
    ) <= 0:
        parser.error("Use positive parameters and a multiple of 24 rounds (all four-variant permutations)")
    for name in ("off_build", "on_build", "baseline_off_worker", "shader_dir", "inspector", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    off = build_identity(args.off_build, False)
    on = build_identity(args.on_build, True)
    ordinary_hash = compare_builds(off, on)
    suffix = ".exe" if os.name == "nt" else ""
    workers = {
        "baseline": args.baseline_off_worker,
        "off": args.off_build / f"bench-worker-nacelle{suffix}",
        "idle": args.on_build / f"bench-worker-nacelle{suffix}",
        "active": args.on_build / f"bench-worker-nacelle{suffix}",
    }
    worker_hashes = {key: sha256(value) for key, value in workers.items()}
    unchanged_off = worker_hashes["baseline"] == worker_hashes["off"]
    shader_hashes = {name: sha256(args.shader_dir / name)
                     for _, names in WORKLOADS.values() for name in names}
    args.output_dir.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("NACELLE_")}
    env["NACELLE_CACHE_DIR"] = str(args.output_dir / "cache")
    metadata = {
        "schema_version": 1, "parameters": vars(args), "platform": platform.platform(),
        "harness_sha256": sha256(Path(__file__)),
        "processor": platform.processor(), "off_build": off, "on_build": on,
        "ordinary_compiler_inputs_sha256": ordinary_hash, "worker_sha256": worker_hashes,
        "shader_sha256": shader_hashes, "off_binary_unchanged": unchanged_off,
        "cleared_environment": sorted(key for key in os.environ if key.startswith("NACELLE_")),
        "method": "Median paired ratios of per-process medians; 20000 paired bootstrap resamples, 95% intervals",
    }
    (args.output_dir / "identity.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    records = {bench: {variant: [] for variant in VARIANTS} for bench in WORKLOADS}
    checksums = {}
    rng = random.Random(args.seed)
    with (args.output_dir / "runs.jsonl").open("x") as raw:
        for bench in WORKLOADS:
            orders = []
            for _ in range(args.rounds // 24):
                block = list(itertools.permutations(VARIANTS))
                rng.shuffle(block)
                orders.extend(block)
            for number, order in [(-1, VARIANTS)] + list(enumerate(orders)):
                for variant in order:
                    record = run_worker(workers[variant], variant, bench, number, args, env, on["compiler_hash"])
                    raw.write(json.dumps(record) + "\n")
                    raw.flush()
                    checksum = record["result"]["checksum"]
                    if bench in checksums and checksums[bench] != checksum:
                        raise ValueError(f"{bench}: output checksum differs; see runs.jsonl")
                    checksums[bench] = checksum
                    if number >= 0:
                        records[bench][variant].append(record["result"]["median_ms"])
                print(f"{bench}: {'warmup' if number < 0 else str(number + 1) + '/' + str(args.rounds)}", flush=True)
    results = {}
    for bench, samples in records.items():
        results[bench] = {}
        for variant, limit in (("off", 0), ("idle", 0.01), ("active", 0.05)):
            reference = samples["baseline"] if variant == "off" else samples["off"]
            interval = paired_interval(reference, samples[variant], args.seed)
            interval["limit_ratio"] = 1 + limit
            interval["verdict"] = "pass" if variant == "off" and unchanged_off else verdict(interval, limit)
            interval["proof"] = "identical static OFF worker" if variant == "off" and unchanged_off else "paired timing"
            results[bench][variant] = interval
    status = conclusion(results, args.controlled_host)
    summary = {"status": status, "controlled_host": args.controlled_host, "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if status == "pass" else 1 if status == "fail" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print(f"measure_profile_overhead: {error}", file=sys.stderr)
        sys.exit(1)
