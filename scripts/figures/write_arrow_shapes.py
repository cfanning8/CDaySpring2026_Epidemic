from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_project_assets as gpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write pipeline arrow mesh PNGs only (PyVista).")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / "results" / "figures" / "pipeline_overview" / "assets" / "arrow_shapes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if gpa.pv is None:
        raise RuntimeError("PyVista is required for arrow mesh assets.")
    gpa._write_pipeline_arrow_shape_assets(args.asset_dir)
    print(f"Wrote arrow meshes under {args.asset_dir}")


if __name__ == "__main__":
    main()
