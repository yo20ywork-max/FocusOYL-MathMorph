"""Reproduce the published Prism recipe from its exact original GGUF.

No training, downloads, model serving, or automatic publication is performed.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

BASELINE_SHA256 = "68c40b08b1242754a107b9510af89aa75b10c75843ca7844643c70956b7f1e3d"
RECIPE = {"label": "euclid_control", "layers": [23], "rank": 512,
          "budget": 0.12, "geometry": False}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original MiniCPM5 F16 GGUF")
    parser.add_argument("--out", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    out = args.out.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".gguf":
        parser.error("--source must be an existing GGUF file")
    if out.exists():
        parser.error("--out must not already exist; originals are never overwritten")
    archived = Path(__file__).resolve().parent / "archive" / "v0.4"
    if not (archived / "convert_geometry.py").is_file():
        parser.error("Missing archive/v0.4 source; clone the complete repository")
    sys.path.insert(0, str(archived))
    from convert_geometry import convert
    try:
        candidate = convert(source, out, dict(RECIPE), expected_sha=BASELINE_SHA256)
    except Exception as exc:
        parser.exit(1, f"Conversion rejected: {exc}\n")
    print(f"Research candidate written: {candidate}")
    print("Status: UNVERIFIED. This command does not certify a capability upgrade.")

if __name__ == "__main__":
    main()
