from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = str(Path(sys.executable).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import HeatRandomFeatures  # noqa: E402

DATASET_CONFIGS = {
    "Infectious": {
        "dataset_key": r"infectious\sciencegallery_infectious_contacts\listcontacts_2009_06_10.txt",
        "window_seconds": 24 * 60 * 60,
        "stride_seconds": 24 * 60 * 60,
        "min_events": 20,
    },
}


def main() -> None:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASET_CONFIGS:
            raise ValueError(f"unknown dataset in --datasets: {d}")
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    for lv in levels:
        if lv not in (1, 2):
            raise ValueError("only levels 1 and 2 are supported")

    ensure_cuda()

    for level in levels:
        run_level(level=level, datasets=datasets, args=args)

    total_cmd = [PYTHON_EXE, "scripts/figures/generate_total_level_figures.py"]
    run_command(total_cmd, "total_level_figures")


def run_level(level: int, datasets: list[str], args: argparse.Namespace) -> None:
    stage_root = PROJECT_ROOT / "results" / f"level{level}"
    stage_output_dir = stage_root / "output"
    stage_weight_dir = stage_root / "weights"
    stage_fig_dir = PROJECT_ROOT / "results" / "figures" / f"level{level}"
    stage_temp_root = PROJECT_ROOT / "temp" / f"level{level}"
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    stage_weight_dir.mkdir(parents=True, exist_ok=True)
    stage_fig_dir.mkdir(parents=True, exist_ok=True)
    stage_temp_root.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float | str]] = []
    ts_rows: list[dict[str, float | str | int]] = []

    for dataset in datasets:
        slug = dataset.lower()
        base_temp = PROJECT_ROOT / "temp" / f"pilot_{slug}"
        base_features = base_temp / "features.npz"
        base_table = base_temp / "table.csv"
        if not base_features.exists() or not base_table.exists():
            raise RuntimeError(
                f"missing poster-grade base artifacts for {dataset}. "
                f"expected {base_features} and {base_table}"
            )
        stage_temp = stage_temp_root / f"pilot_{slug}"
        stage_temp.mkdir(parents=True, exist_ok=True)
        stage_features = stage_temp / "features.npz"
        stage_table = stage_temp / "table.csv"
        build_level_features(level, base_features, base_table, stage_features, stage_table, args.rff_temperature)

        cfg = DATASET_CONFIGS[dataset]
        cache_dir = PROJECT_ROOT / "data" / "preprocessed" / dataset
        tag_tgn = f"level{level}_{slug}_tgn"
        tag_landscape = f"level{level}_{slug}_landscape"
        tag_rkhs = f"level{level}_{slug}_rkhs"

        run_command(
            [
                PYTHON_EXE,
                "scripts/train_tgn_baseline.py",
                "--dataset-key",
                str(cfg["dataset_key"]),
                "--window-seconds",
                str(cfg["window_seconds"]),
                "--stride-seconds",
                str(cfg["stride_seconds"]),
                "--min-events-per-window",
                str(cfg["min_events"]),
                "--feature-npz",
                str(stage_features),
                "--preprocessed-cache-dir",
                str(cache_dir),
                "--epochs",
                str(args.epochs),
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--early-stopping-min-delta",
                str(args.early_stopping_min_delta),
                "--output-tag",
                tag_tgn,
                "--device",
                "cuda",
            ],
            f"level{level}:{dataset}:train_tgn",
        )
        run_command(
            [
                PYTHON_EXE,
                "scripts/train_tgn_landscape_constraint.py",
                "--dataset-key",
                str(cfg["dataset_key"]),
                "--window-seconds",
                str(cfg["window_seconds"]),
                "--stride-seconds",
                str(cfg["stride_seconds"]),
                "--min-events-per-window",
                str(cfg["min_events"]),
                "--feature-npz",
                str(stage_features),
                "--preprocessed-cache-dir",
                str(cache_dir),
                "--epochs",
                str(args.epochs),
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--early-stopping-min-delta",
                str(args.early_stopping_min_delta),
                "--lambda-d",
                str(args.lambda_d),
                "--output-tag",
                tag_landscape,
                "--device",
                "cuda",
            ],
            f"level{level}:{dataset}:train_landscape",
        )
        run_command(
            [
                PYTHON_EXE,
                "scripts/train_tgn_rkhs_constraint.py",
                "--dataset-key",
                str(cfg["dataset_key"]),
                "--window-seconds",
                str(cfg["window_seconds"]),
                "--stride-seconds",
                str(cfg["stride_seconds"]),
                "--min-events-per-window",
                str(cfg["min_events"]),
                "--feature-npz",
                str(stage_features),
                "--preprocessed-cache-dir",
                str(cache_dir),
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(args.rkhs_learning_rate),
                "--weight-decay",
                str(args.rkhs_weight_decay),
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--early-stopping-min-delta",
                str(args.early_stopping_min_delta),
                "--lambda-g",
                str(args.lambda_g),
                "--output-tag",
                tag_rkhs,
                "--device",
                "cuda",
            ],
            f"level{level}:{dataset}:train_rkhs",
        )

        output_dir = PROJECT_ROOT / "results" / "output"
        weight_dir = PROJECT_ROOT / "results" / "weights"
        pred_paths = {
            "TGN": output_dir / f"tgn_baseline_predictions_{tag_tgn}.csv",
            "Landscape": output_dir / f"tgn_landscape_constraint_predictions_{tag_landscape}.csv",
            "RKHS": output_dir / f"tgn_rkhs_constraint_predictions_{tag_rkhs}.csv",
        }
        hist_paths = [
            output_dir / f"tgn_baseline_history_{tag_tgn}.csv",
            output_dir / f"tgn_landscape_constraint_history_{tag_landscape}.csv",
            output_dir / f"tgn_rkhs_constraint_history_{tag_rkhs}.csv",
        ]
        weight_paths = [
            weight_dir / f"tgn_baseline_{tag_tgn}.pt",
            weight_dir / f"tgn_landscape_constraint_{tag_landscape}.pt",
            weight_dir / f"tgn_rkhs_constraint_{tag_rkhs}.pt",
        ]
        for p in list(pred_paths.values()) + hist_paths:
            if not p.exists():
                raise RuntimeError(f"missing training artifact: {p}")
            shutil.copy2(p, stage_output_dir / p.name)
        for w in weight_paths:
            if not w.exists():
                raise RuntimeError(f"missing weight artifact: {w}")
            shutil.copy2(w, stage_weight_dir / w.name)

        for model_name, path in pred_paths.items():
            metrics = evaluate_predictions(path, risk_threshold=args.risk_threshold)
            metrics_rows.append({"dataset": dataset, "model": model_name, **metrics})

        table = pd.read_csv(stage_table)
        y = table["y_large_outbreak_prob"].to_numpy(dtype=float)
        y_ci = 1.96 * np.sqrt(np.maximum(y * (1.0 - y), 0.0) / 100.0)
        for idx in range(len(table)):
            ts_rows.append(
                {
                    "dataset": dataset,
                    "t": int(table.loc[idx, "t"]),
                    "y_t": float(table.loc[idx, "y_large_outbreak_prob"]),
                    "g_l2_norm": float(table.loc[idx, "g_l2_norm"]),
                    "y_ci": float(y_ci[idx]),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    max_rmse = float(max(metrics_df["rmse"].max(), 1e-12))
    metrics_df["rmse_norm"] = metrics_df["rmse"] / max_rmse
    metrics_df.to_csv(stage_output_dir / "collective_metrics.csv", index=False)
    ts_df = pd.DataFrame(ts_rows).sort_values(["dataset", "t"]).reset_index(drop=True)
    ts_df.to_csv(stage_output_dir / "collective_timeseries.csv", index=False)

    copy_stage_model_aliases(level, stage_output_dir)
    figure_cmd = [
        PYTHON_EXE,
        "scripts/figures/generate_project_assets.py",
        "--table-csv",
        str(stage_temp_root / "pilot_infectious" / "table.csv"),
        "--features-npz",
        str(stage_temp_root / "pilot_infectious" / "features.npz"),
        "--model-output-dir",
        str(stage_output_dir),
        "--collective-metrics-csv",
        str(stage_output_dir / "collective_metrics.csv"),
        "--collective-timeseries-csv",
        str(stage_output_dir / "collective_timeseries.csv"),
        "--figure-output-root",
        str(stage_fig_dir),
        "--dataset-key",
        r"infectious\sciencegallery_infectious_contacts\listcontacts_2009_06_10.txt",
        "--min-events-per-window",
        "20",
        "--preprocessed-cache-dir",
        str(PROJECT_ROOT / "data" / "preprocessed" / "Infectious"),
    ]
    run_command(figure_cmd, f"level{level}:generate_figures")


def copy_stage_model_aliases(level: int, stage_output_dir: Path) -> None:
    slug = "infectious"
    mapping = {
        stage_output_dir / f"tgn_baseline_predictions_level{level}_{slug}_tgn.csv": stage_output_dir / "tgn_baseline_predictions.csv",
        stage_output_dir / f"tgn_landscape_constraint_predictions_level{level}_{slug}_landscape.csv": stage_output_dir
        / "tgn_landscape_constraint_predictions.csv",
        stage_output_dir / f"tgn_rkhs_constraint_predictions_level{level}_{slug}_rkhs.csv": stage_output_dir
        / "tgn_rkhs_constraint_predictions.csv",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)


def build_level_features(
    level: int,
    base_features_path: Path,
    base_table_path: Path,
    out_features_path: Path,
    out_table_path: Path,
    rff_temperature: float,
) -> None:
    base = np.load(base_features_path)
    base_table = pd.read_csv(base_table_path)
    d_t = np.asarray(base["d_t"], dtype=np.float32)
    g_t = np.asarray(base["g_t"], dtype=np.float32)
    rkhs_g_t = np.asarray(base["rkhs_g_t"], dtype=np.float32)
    y_t = np.asarray(base["y_t"], dtype=np.float32)
    if level == 1:
        np.savez_compressed(out_features_path, d_t=d_t, g_t=g_t, rkhs_g_t=rkhs_g_t, y_t=y_t)
        base_table.to_csv(out_table_path, index=False)
        return

    if g_t.shape[0] < 2:
        raise RuntimeError("level2 requires at least two first-order drift rows")
    # Level-2 sample k uses (g_{k+1} - g_k), which depends on D_k, D_{k+1}, D_{k+2}.
    # To avoid temporal leakage, align the target with the same endpoint (y_{k+2}).
    d2_raw = g_t[1:].astype(np.float32)
    g2_raw = (g_t[1:] - g_t[:-1]).astype(np.float32)
    if rkhs_g_t.shape[1] <= 0:
        raise RuntimeError("invalid rkhs feature dimension")
    rff = HeatRandomFeatures(
        input_dim=g2_raw.shape[1],
        n_components=int(rkhs_g_t.shape[1]),
        temperature=float(rff_temperature),
        random_state=14,
    )
    rkhs2_raw = rff.transform(g2_raw).astype(np.float32)
    y2_raw = y_t[2:].astype(np.float32)
    n2 = int(min(len(d2_raw), len(g2_raw), len(rkhs2_raw), len(y2_raw), max(0, len(base_table) - 2)))
    if n2 <= 0:
        raise RuntimeError("level2 has no aligned rows after trimming")
    d2 = d2_raw[:n2]
    g2 = g2_raw[:n2]
    rkhs2 = rkhs2_raw[:n2]
    y2 = y2_raw[:n2]
    np.savez_compressed(out_features_path, d_t=d2, g_t=g2, rkhs_g_t=rkhs2, y_t=y2)

    table2 = base_table.iloc[2 : 2 + n2].copy().reset_index(drop=True)
    table2["t"] = np.arange(len(table2), dtype=int)
    table2["y_large_outbreak_prob"] = y2
    y_next = np.concatenate([y2[1:], y2[-1:]])
    table2["y_next_large_outbreak_prob"] = y_next
    table2["delta_y_large_outbreak_prob"] = y_next - y2
    table2["g_l2_norm"] = np.linalg.norm(g2, axis=1)
    table2.to_csv(out_table_path, index=False)


def evaluate_predictions(predictions_path: Path, risk_threshold: float) -> dict[str, float | str]:
    df = pd.read_csv(predictions_path)
    eval_scope = "all"
    if "split" in df.columns:
        test_df = df[df["split"] == "test"].copy()
        if len(test_df) > 0:
            df = test_df
            eval_scope = "test"
        else:
            raise RuntimeError(f"missing test split rows in predictions file: {predictions_path}")
    if len(df) == 0:
        raise RuntimeError(f"empty evaluation slice in predictions file: {predictions_path}")
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = np.clip(df["y_pred"].to_numpy(dtype=float), 0.0, 1.0)
    y_true_bin = (y_true >= risk_threshold).astype(np.int64)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    brier = float(np.mean((y_pred - y_true_bin.astype(float)) ** 2))
    ece = expected_calibration_error(y_true_bin, y_pred, num_bins=10)
    positive_count = int(np.sum(y_true_bin == 1))
    negative_count = int(np.sum(y_true_bin == 0))
    class_balance_ok = int((positive_count > 0) and (negative_count > 0))
    if class_balance_ok == 0:
        print(
            f"[warn] one-class test labels at threshold={risk_threshold} for {predictions_path.name}; "
            "Brier/ECE are class-imbalanced diagnostics."
        )
    return {
        "rmse": rmse,
        "brier": brier,
        "ece": ece,
        "eval_count": float(len(y_true_bin)),
        "positive_count": float(positive_count),
        "negative_count": float(negative_count),
        "class_balance_ok": float(class_balance_ok),
        "eval_scope": eval_scope,
    }


def expected_calibration_error(y_true_binary: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    total = max(1, y_true_binary.size)
    ece = 0.0
    for idx in range(num_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == num_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true_binary[mask]))
        ece += (float(np.sum(mask)) / float(total)) * abs(acc - conf)
    return float(ece)


def ensure_cuda() -> None:
    cmd = [PYTHON_EXE, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA is required'"]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError("CUDA check failed; training is GPU-only")


def run_command(command: list[str], name: str) -> None:
    print(f"[run] {name}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed for step: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run levelized topological jet hierarchy training and figures.")
    parser.add_argument("--datasets", type=str, default="Infectious")
    parser.add_argument("--levels", type=str, default="1,2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--lambda-d", type=float, default=0.3)
    parser.add_argument("--lambda-g", type=float, default=0.7)
    parser.add_argument("--rkhs-learning-rate", type=float, default=0.001)
    parser.add_argument("--rkhs-weight-decay", type=float, default=1e-4)
    parser.add_argument("--rff-temperature", type=float, default=0.2)
    parser.add_argument("--risk-threshold", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
