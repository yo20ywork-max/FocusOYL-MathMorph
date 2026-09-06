"""Experimental Euclidean GGUF editing using the unchanged MathMorph v0.4 core.

For the fixed published Prism recipe, use convert_prism.py instead.
No downloads, inference, training, or automatic publication are performed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

VERSION = "1.0.0"
ARCHIVE = Path(__file__).resolve().parent / "archive" / "v0.4"
NATIVE_TYPES = {0: "F32", 1: "F16", 30: "BF16"}


def load_core():
    if not (ARCHIVE / "convert_geometry.py").is_file():
        raise ValueError("Missing archive/v0.4: clone the complete repository")
    sys.path.insert(0, str(ARCHIVE))
    import convert_geometry
    return convert_geometry


def resolve_layers(text: str, blocks: int) -> list[int]:
    if text.strip().lower() == "last":
        return [blocks - 1]
    fields = text.split(",")
    if not fields or any(not re.fullmatch(r"[0-9]+", s.strip()) for s in fields):
        raise ValueError("--layers must be 'last' or comma-separated zero-based indices")
    layers = [int(s.strip()) for s in fields]
    if len(set(layers)) != len(layers):
        raise ValueError("Duplicate layer indices are not allowed")
    if any(not 0 <= layer < blocks for layer in layers):
        raise ValueError(f"Layer indices must be in 0..{blocks - 1}")
    return sorted(layers)


def preflight(source: Path, layers: str, rank: int, budget: float, core) -> dict:
    if not source.is_file() or source.suffix.lower() != ".gguf":
        raise ValueError("--source must be an existing GGUF file")
    if not math.isfinite(budget) or not 0 < budget <= 0.30:
        raise ValueError("--budget must be finite and in (0, 0.30]")
    if rank < 1 or rank > 8192:
        raise ValueError("--rank must be in 1..8192")
    gguf = core.GGUF(source)
    architecture, blocks = core.validate(gguf)
    if blocks < 1:
        raise ValueError("The GGUF must declare a positive block count")
    selected = resolve_layers(layers, blocks)
    targets = []
    workspaces = []
    for layer in selected:
        name = f"blk.{layer}.ffn_down.weight"
        tensor = gguf.tensors.get(name)
        if tensor is None or len(tensor.shape) != 2:
            raise ValueError(f"Missing or non-matrix target: {name}")
        if tensor.qtype not in NATIVE_TYPES:
            raise ValueError(f"{name}: target must be F16, F32, or BF16; no quantized writes")
        if any(f"blk.{layer}.ffn_{part}.bias" in gguf.tensors for part in ("gate", "up", "down")):
            raise ValueError(f"Block {layer}: biased FFNs are not supported")
        m, n = tensor.shape
        if rank >= min(m, n):
            raise ValueError(f"{name}: --rank must be smaller than min(shape)={min(m, n)}")
        workspace = m * m * 32 + n * n * 36 + m * n * 24
        if max(m, n) > 8192 or workspace > 3 * 1024**3:
            raise ValueError(f"{name}: exceeds the archived core's conservative workspace guard")
        workspaces.append(workspace)
        targets.append({"name": name, "shape": list(tensor.shape),
                        "storage": NATIVE_TYPES[tensor.qtype], "bytes": tensor.nbytes})
    if max(workspaces) + sum(t["bytes"] for t in targets) > 3 * 1024**3:
        raise ValueError("Selected targets exceed the conservative workspace plus patch-byte guard")
    return {
        "entrypoint": "convert_euclidean.py", "entrypoint_version": VERSION,
        "architecture": architecture, "blocks": blocks, "source_bytes": gguf.size,
        "recipe": {"label": "euclidean_experiment", "layers": selected,
                   "rank": rank, "budget": budget, "geometry": False},
        "targets": targets, "budget_scope": "relative Frobenius change per selected tensor",
        "readout_protection": False, "capability_status": "UNVERIFIED",
        "preflight_scope": "metadata, layout, and parameter checks; no tensor-value or inference validation",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="New output directory; required unless --dry-run")
    parser.add_argument("--layers", default="last", help="'last' or comma-separated indices, e.g. 21,23")
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--budget", type=float, default=0.12, help="Per-tensor relative change cap, at most 0.30")
    parser.add_argument("--expected-sha256", help="Optional expected source identity; mismatch rejects")
    parser.add_argument("--dry-run", action="store_true", help="Check metadata and print plan without writing a GGUF")
    args = parser.parse_args(argv)
    if not args.dry_run and args.out is None:
        parser.error("--out is required for conversion")
    if args.expected_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_sha256):
        parser.error("--expected-sha256 must contain exactly 64 hexadecimal characters")
    try:
        source = args.source.expanduser().resolve(strict=True)
        out = args.out.expanduser() if args.out else None
        if out is not None and os.path.lexists(out):
            raise ValueError("--out must not exist; existing files, directories, and symlinks are never replaced")
        out = out.resolve() if out else None
        core = load_core()
        plan = preflight(source, args.layers, args.rank, args.budget, core)
        expected = args.expected_sha256.lower() if args.expected_sha256 else None
        # Hashing reads the file but performs no model inference or weight modification.
        if expected and core.sha256(source) != expected:
            raise ValueError("Source SHA-256 mismatch")
        print(json.dumps(plan, indent=2, allow_nan=False))
        if args.dry_run:
            print("DRY_RUN: no output written; numerical conversion and model behavior remain untested.")
            return 0
        candidate = core.convert(source, out, plan["recipe"], expected_sha=expected)
        receipt = {**plan, "entrypoint_sha256": core.sha256(Path(__file__)),
                   "candidate": str(candidate), "training": False, "inference_started": False}
        with (out / "entrypoint.json").open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, allow_nan=False)
            f.write("\n")
        print(f"Candidate written: {candidate}")
        print("UNVERIFIED: benchmark against the original before using or publishing this candidate.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Inspect the output directory before retrying; choose a new output path.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, ImportError, ArithmeticError, MemoryError) as exc:
        print(f"Conversion rejected: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
