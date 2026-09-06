"""Synthetic GGUF integration tests. No model downloads or capability scores."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import convert_euclidean as cli


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_fixture(path, *, arch="llama", qtype=1, extra=None, bias=False,
                  vector=False, zero=False, nonfinite=False):
    """Write a tiny structural fixture, deliberately not an executable language model."""
    core = cli.load_core()
    rng = np.random.default_rng(717)
    matrix = rng.normal(size=(4, 32 if qtype == 2 else 6)).astype(np.float32)
    if zero:
        matrix[:] = 0
    if nonfinite:
        matrix[0, 0] = np.nan
    metadata = {"general.architecture": arch, f"{arch}.block_count": 2, **(extra or {})}
    arrays = [(f"blk.{i}.ffn_down.weight", matrix.copy(), qtype) for i in range(2)]
    if vector:
        arrays[1] = (arrays[1][0], matrix[0], qtype)
    arrays.append(("token_embd.weight", np.arange(8, dtype=np.float32).reshape(2, 4), 0))
    if bias:
        arrays.append(("blk.1.ffn_down.bias", np.ones(4, dtype=np.float32), 0))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(arrays), len(metadata))
    for key, value in metadata.items():
        header += string(key)
        header += struct.pack("<I", 8) + string(value) if isinstance(value, str) else struct.pack("<II", 4, value)
    descriptors = b""
    data = bytearray()
    for name, array, typ in arrays:
        data.extend(b"\0" * (-len(data) % 32))
        descriptors += string(name) + struct.pack("<I", array.ndim)
        descriptors += b"".join(struct.pack("<Q", size) for size in reversed(array.shape))
        descriptors += struct.pack("<IQ", typ, len(data))
        if typ == 2:
            raw = b"\0" * (array.size // 32 * 18)
        elif nonfinite and typ in (0, 1):
            raw = array.astype("<f4" if typ == 0 else "<f2").tobytes()
        else:
            raw = core.encode_native(array, typ)
        data.extend(raw)
    header += descriptors
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + data)


class EuclideanCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.gguf"
        self.out = self.root / "output"
        write_fixture(self.source)
        self.core = cli.load_core()

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(["--source", str(self.source), "--rank", "2", *map(str, args)])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_layer_resolution(self):
        self.assertEqual(cli.resolve_layers("last", 24), [23])
        self.assertEqual(cli.resolve_layers("23, 21", 24), [21, 23])
        for text in ("", "-1", "1,1", "all", "0,", "24", "1-3"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                cli.resolve_layers(text, 24)

    def test_dry_run_creates_no_outputs(self):
        before = set(self.root.iterdir())
        status, output, _ = self.run_cli("--dry-run")
        self.assertEqual(status, 0)
        self.assertIn('"geometry": false', output)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_floating_point_encodings_and_byte_audit(self):
        for typ in (0, 1, 30):
            with self.subTest(qtype=typ):
                write_fixture(self.source, qtype=typ)
                original = self.source.read_bytes()
                out = self.root / f"type-{typ}"
                status, _, error = self.run_cli("--out", out)
                self.assertEqual(status, 0, error)
                self.assertEqual(self.source.read_bytes(), original)
                result = out / "candidate.gguf"
                self.assertEqual(result.stat().st_size, len(original))
                src, dst = self.core.GGUF(self.source), self.core.GGUF(result)
                self.assertEqual(src.tensors, dst.tensors)
                self.assertEqual(src.metadata, dst.metadata)
                self.assertEqual(src.raw("blk.0.ffn_down.weight"), dst.raw("blk.0.ffn_down.weight"))
                self.assertEqual(src.raw("token_embd.weight"), dst.raw("token_embd.weight"))
                self.assertNotEqual(src.raw("blk.1.ffn_down.weight"), dst.raw("blk.1.ffn_down.weight"))
                report = json.loads((out / "report.json").read_text())
                self.assertLessEqual(report["tensors"][0]["encoded_relative_change"], 0.12)
                self.assertEqual(report["capability_status"], "UNVERIFIED")
                self.assertFalse(json.loads((out / "entrypoint.json").read_text())["inference_started"])

    def test_multiple_explicit_layers(self):
        status, _, error = self.run_cli("--out", self.out, "--layers", "0,1")
        self.assertEqual(status, 0, error)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["recipe"]["layers"], [0, 1])
        self.assertEqual(len(report["tensors"]), 2)

    def test_existing_output_is_never_overwritten(self):
        self.out.mkdir()
        marker = self.out / "marker.txt"
        marker.write_text("keep")
        self.assertEqual(self.run_cli("--out", self.out)[0], 1)
        self.assertEqual(marker.read_text(), "keep")

    def test_expected_hash_mismatch(self):
        self.assertEqual(self.run_cli("--out", self.out, "--expected-sha256", "0" * 64)[0], 1)
        self.assertFalse(self.out.exists())

    def test_expected_hash_accepts_uppercase(self):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest().upper()
        self.assertEqual(self.run_cli("--dry-run", "--expected-sha256", digest)[0], 0)

    def test_budget_rejects_invalid_values(self):
        for value in ("0", "-0.1", "0.31", "nan", "inf"):
            with self.subTest(value=value):
                self.assertEqual(self.run_cli("--dry-run", "--budget", value)[0], 1)

    def test_invalid_rank_is_not_silently_clamped(self):
        for value in ("0", "4", "8193"):
            with self.subTest(value=value):
                self.assertEqual(self.run_cli("--dry-run", "--rank", value)[0], 1)

    def test_supported_architecture_metadata(self):
        for arch in ("llama", "qwen2", "qwen3"):
            with self.subTest(arch=arch):
                write_fixture(self.source, arch=arch)
                self.assertEqual(self.run_cli("--dry-run")[0], 0)

    def test_unsupported_architecture_moe_and_shards(self):
        for config in ({"arch": "unknown"}, {"extra": {"llama.expert_count": 8}},
                       {"extra": {"split.count": 2}}):
            with self.subTest(config=config):
                write_fixture(self.source, **config)
                self.assertEqual(self.run_cli("--dry-run")[0], 1)

    def test_bias_vector_and_quantized_target_rejected(self):
        for config in ({"bias": True}, {"vector": True}, {"qtype": 2}):
            with self.subTest(config=config):
                write_fixture(self.source, **config)
                self.assertEqual(self.run_cli("--out", self.out)[0], 1)
                self.assertFalse(self.out.exists())

    def test_nonfinite_and_noop_numerical_edits_rejected(self):
        for index, config in enumerate(({"nonfinite": True}, {"zero": True})):
            with self.subTest(config=config):
                write_fixture(self.source, **config)
                original = self.source.read_bytes()
                out = self.root / f"invalid-{index}"
                self.assertEqual(self.run_cli("--out", out)[0], 1)
                self.assertFalse((out / "candidate.gguf").exists())
                self.assertTrue((out / "REJECTED.json").exists())
                self.assertEqual(self.source.read_bytes(), original)

    def test_pointer_file_is_not_a_gguf(self):
        self.source.write_text("version https://git-lfs.github.com/spec/v1\n")
        self.assertEqual(self.run_cli("--dry-run")[0], 1)

    def test_missing_source_and_archive(self):
        self.source.unlink()
        self.assertEqual(self.run_cli("--dry-run")[0], 1)
        write_fixture(self.source)
        with mock.patch.object(cli, "ARCHIVE", self.root / "missing"):
            self.assertEqual(self.run_cli("--dry-run")[0], 1)

    def test_required_arguments(self):
        for args in ([], ["--dry-run", "--expected-sha256", "bad"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as stopped:
                self.run_cli(*args)
            self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
