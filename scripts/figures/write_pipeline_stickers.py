from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_project_assets import write_pipeline_overview_stickers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write pipeline overview CSV + sticker PNGs only (matplotlib).")
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=ROOT / "temp" / "pilot_infectious" / "table.csv",
    )
    parser.add_argument(
        "--features-npz",
        type=Path,
        default=ROOT / "temp" / "pilot_infectious" / "features.npz",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / "results" / "figures" / "pipeline_overview" / "assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.table_csv.is_file():
        raise RuntimeError(f"missing --table-csv: {args.table_csv}")
    if not args.features_npz.is_file():
        raise RuntimeError(f"missing --features-npz: {args.features_npz}")
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    write_pipeline_overview_stickers(args.table_csv, args.features_npz, args.asset_dir)
    print(f"Wrote pipeline stickers under {args.asset_dir}")


if __name__ == "__main__":
    main()
