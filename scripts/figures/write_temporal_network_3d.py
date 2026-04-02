from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_project_assets as gpa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write temporal_network_3d PNGs only (PyVista).")
    p.add_argument("--table-csv", type=Path, default=gpa.PILOT_INFECTIOUS_TABLE_DEFAULT)
    p.add_argument("--features-npz", type=Path, default=gpa.PILOT_INFECTIOUS_FEATURES_DEFAULT)
    p.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / "results" / "figures" / "temporal_network_3d" / "assets",
    )
    p.add_argument("--dataset-key", type=str, default=gpa.PRIMARY_SCHOOL_DATASET_KEY)
    p.add_argument("--min-events-per-window", type=int, default=20)
    p.add_argument("--preprocessed-cache-dir", type=Path, default=gpa.INFECTIOUS_PREPROCESSED_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if gpa.pv is None:
        raise RuntimeError("PyVista is required.")
    if not args.table_csv.is_file():
        raise RuntimeError(f"missing --table-csv: {args.table_csv}")
    if not args.features_npz.is_file():
        raise RuntimeError(f"missing --features-npz: {args.features_npz}")
    table = pd.read_csv(args.table_csv)
    gpa._assert_table_features_row_alignment(table, args.features_npz)
    gpa.generate_temporal_network_3d_assets(
        table,
        args.asset_dir,
        dataset_key=args.dataset_key,
        min_events_per_window=args.min_events_per_window,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
    )
    print(args.asset_dir)


if __name__ == "__main__":
    main()
