#!/usr/bin/env python3
"""Focused tests for coarse-capture comparison admission and acceptance."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "measure_profile_overhead", Path(__file__).resolve().parents[1] / "scripts/measure_profile_overhead.py"
)
assert SPEC is not None and SPEC.loader is not None
overhead = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overhead)


class ProfileOverheadTest(unittest.TestCase):
    def test_uncontrolled_and_incomplete_acceptance_cannot_pass(self):
        results = {bench: {variant: {"verdict": "pass"} for variant in ("off", "idle", "active")}
                   for bench in overhead.WORKLOADS}
        self.assertEqual(overhead.conclusion(results, False), "inconclusive")
        self.assertEqual(overhead.conclusion(results, True), "pass")
        results["bench_raymarch"]["idle"]["verdict"] = "inconclusive"
        self.assertEqual(overhead.conclusion(results, True), "inconclusive")
        results["bench_raymarch"]["idle"]["verdict"] = "fail"
        self.assertEqual(overhead.conclusion(results, True), "fail")
        with self.assertRaises(ValueError):
            overhead.conclusion({}, True)
        del results["bench_raymarch"]["off"]
        with self.assertRaises(ValueError):
            overhead.conclusion(results, True)

    def test_paired_ratios_not_ratio_of_unpaired_medians(self):
        interval = overhead.paired_interval([1, 10, 100], [2, 20, 10], 211, resamples=1000)
        self.assertEqual(interval["paired_median_ratio"], 2)
        self.assertEqual(interval["paired_ratio_range"], [0.1, 2])
        self.assertEqual(overhead.verdict(interval, 0.05), "inconclusive")

    def test_acceptance_thresholds_and_uncertainty(self):
        for ratio, limit, expected in (
            (1.01, 0.01, "pass"), (1.011, 0.01, "fail"),
            (1.05, 0.05, "pass"), (1.051, 0.05, "fail"),
        ):
            with self.subTest(ratio=ratio):
                interval = overhead.paired_interval([100] * 24, [100 * ratio] * 24, 211, resamples=100)
                self.assertEqual(overhead.verdict(interval, limit), expected)
        self.assertEqual(overhead.verdict({"ratio_ci95": [0.99, 1.02]}, 0.01), "inconclusive")

    def test_invalid_pairs_are_not_accepted(self):
        for baseline, measured in (([], []), ([1], [1, 2]), ([1], [0]), ([1], [float("nan")])):
            with self.subTest(measured=measured), self.assertRaises(ValueError):
                overhead.paired_interval(baseline, measured, 211)

    def test_compiler_identity_allows_only_declared_debug_inputs(self):
        off = {"flags": {"LLVM_DIR": "llvm"}, "inputs": {"generated/math_simd.bc": "same"}}
        on = copy.deepcopy(off)
        on["inputs"].update({key: "debug" for key in overhead.DEBUG_INPUTS})
        self.assertEqual(len(overhead.compare_builds(off, on)), 64)
        for key in ("generated/math_simd.bc", "src/compiler/jit_engine.cpp"):
            changed = copy.deepcopy(on)
            changed["inputs"][key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                overhead.compare_builds(off, changed)
        changed = copy.deepcopy(on)
        changed["flags"]["LLVM_DIR"] = "another-toolchain"
        with self.assertRaises(ValueError):
            overhead.compare_builds(off, changed)

    def test_stale_compiler_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            (build / "generated").mkdir(parents=True)
            source = root / "compiler.cpp"
            source.write_text("original compiler input")
            (build / "CMakeCache.txt").write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\nNACELLE_ENABLE_INSTRUMENTATION:BOOL=OFF\n"
            )
            (build / "generated/compiler_hash_inputs.txt").write_text(
                f"LLVM_PACKAGE_VERSION=24\n{source}\n"
            )
            stamp = "LLVM_PACKAGE_VERSION=24\ncompiler.cpp:" + overhead.sha256(source) + "\n"
            digest = overhead.hashlib.sha256(stamp.encode()).hexdigest()
            (build / "generated/compiler_version.h").write_text(f'kCompilerBuildHash[] = "{digest}";')
            with patch.object(overhead, "ROOT", root):
                self.assertEqual(overhead.build_identity(build, False)["compiler_hash"], digest)
                source.write_text("changed compiler input")
                with self.assertRaises(ValueError):
                    overhead.build_identity(build, False)

    def test_result_rejects_success_shaped_failures(self):
        result = {
            "ran": True, "status": "OK", "bench_id": "bench_raymarch",
            "simd_width": 8, "category": "Compute", "shader_format": "DXIL",
            "median_ms": 2, "checksum": 123,
        }

        def encoded(value):
            return "<<<BENCH_JSON_BEGIN>>>\n" + json.dumps(value) + "\n<<<BENCH_JSON_END>>>"

        self.assertEqual(overhead.parse_result(encoded(result), "bench_raymarch", 8), result)
        for key, value in (
            ("ran", False), ("status", "SKIPPED"), ("median_ms", 0),
            ("median_ms", True), ("checksum", float("nan")), ("simd_width", 16),
        ):
            changed = dict(result, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                overhead.parse_result(encoded(changed), "bench_raymarch", 8)
        with self.assertRaises(ValueError):
            overhead.parse_result(encoded(result) * 2, "bench_raymarch", 8)

    def test_capture_validation_rejects_loss_and_wrong_compiler(self):
        manifest = {
            "schemaVersion": 1, "complete": True, "recordedEvents": 20,
            "droppedEvents": 0, "build": {"compilerHash": "expected"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.nacelle-profile"
            for key, value in (("complete", False), ("droppedEvents", 1), ("recordedEvents", 0),
                               ("schemaVersion", 3), ("complete", True)):
                changed = dict(manifest, **{key: value})
                with overhead.zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("manifest.json", json.dumps(changed))
                with patch.object(overhead.subprocess, "run") as run:
                    run.return_value.returncode = 0
                    if key == "complete" and value is True:
                        self.assertEqual(overhead.inspect_capture(path, Path("inspector"), "expected", 10), manifest)
                        with self.assertRaises(ValueError):
                            overhead.inspect_capture(path, Path("inspector"), "wrong", 10)
                    else:
                        with self.subTest(key=key), self.assertRaises(ValueError):
                            overhead.inspect_capture(path, Path("inspector"), "expected", 10)


if __name__ == "__main__":
    unittest.main()
