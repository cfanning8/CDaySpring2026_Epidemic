from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import pyvista as pv
except ImportError:
    pv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from persistence_diagram_3d_engine import (  # noqa: E402
    render_persistence_diagram_3d,
    render_virtual_persistence_diagram_3d,
)
from src.dataloaders import load_all_datasets  # noqa: E402
from src.edge_preparation import extract_temporal_events_for_dataset  # noqa: E402
from src.window_cache import aggregate_window_edges  # noqa: E402
from src.window_cache import load_cached_windows  # noqa: E402

COLOR_LOW = "#87CEEB"
COLOR_HIGH = "#DC143C"
COLOR_SECONDARY = "#FF8C00"
COLOR_ACCENT = "#4169E1"
FIGURE_DPI = 300
PRIMARY_SCHOOL_DATASET_KEY = r"infectious\sciencegallery_infectious_contacts\listcontacts_2009_06_10.txt"
INFECTIOUS_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed" / "Infectious"
PILOT_INFECTIOUS_TABLE_DEFAULT = PROJECT_ROOT / "temp" / "pilot_infectious" / "table.csv"
PILOT_INFECTIOUS_FEATURES_DEFAULT = PROJECT_ROOT / "temp" / "pilot_infectious" / "features.npz"


def main() -> None:
    args = parse_args()
    if not args.table_csv.is_file():
        raise RuntimeError(f"missing --table-csv: {args.table_csv}")
    if not args.features_npz.is_file():
        raise RuntimeError(f"missing --features-npz: {args.features_npz}")
    table = pd.read_csv(args.table_csv)
    features = np.load(args.features_npz)
    _assert_table_features_row_alignment(table, args.features_npz)
    output_root = Path(args.figure_output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if pv is None:
        raise RuntimeError("PyVista is required for project asset generation.")

    generate_drift_scatter_assets(
        table,
        output_root / "drift_vs_risk_scatter_3d" / "assets",
        dataset_label=_dataset_label_from_key(args.dataset_key),
    )
    generate_rkhs_trajectory_assets(table, features, output_root / "rkhs_trajectory_3d" / "assets")
    generate_persistence_regime_assets(
        table,
        features,
        output_root / "persistence_regimes_3d" / "assets",
        dataset_key=args.dataset_key,
        min_events_per_window=args.min_events_per_window,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
    )
    generate_temporal_network_3d_assets(
        table,
        output_root / "temporal_network_3d" / "assets",
        dataset_key=args.dataset_key,
        min_events_per_window=args.min_events_per_window,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
    )
    generate_model_comparison_assets(args.model_output_dir, output_root / "model_comparison_3d" / "assets")
    generate_pipeline_spec_assets(
        args.table_csv,
        args.features_npz,
        output_root / "pipeline_overview" / "assets",
        dataset_key=args.dataset_key,
        min_events_per_window=args.min_events_per_window,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
    )
    generate_collective_assets(
        output_root=output_root,
        collective_metrics_csv=args.collective_metrics_csv,
        collective_timeseries_csv=args.collective_timeseries_csv,
    )
    print(f"Generated project assets under {output_root}")


def generate_drift_timeseries_assets(table: pd.DataFrame, asset_dir: Path) -> None:
    _clear_asset_dir(asset_dir)
    t_values = table["t"].to_numpy()
    y_values = table["y_large_outbreak_prob"].to_numpy()
    g_values = table["g_l2_norm"].to_numpy()
    g_scaled = (g_values - g_values.min()) / (g_values.max() - g_values.min() + 1e-12)
    y_ci = 1.96 * np.sqrt(np.maximum(y_values * (1.0 - y_values), 0.0) / 200.0)
    g_ci = np.full_like(g_scaled, max(float(np.std(g_scaled) * 0.20), 0.03))
    ci_diagnostics = {
        "median_y_ci_width": float(np.median(2.0 * y_ci)),
        "median_g_ci_width": float(np.median(2.0 * g_ci)),
    }
    show_y_ci = ci_diagnostics["median_y_ci_width"] >= 0.01
    show_g_ci = ci_diagnostics["median_g_ci_width"] >= 0.01

    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    _add_floor_grid(plotter, float(t_values.min()), float(t_values.max()), -0.2, 1.2, z=0.0, steps=10)

    risk_points = np.column_stack([t_values, np.zeros_like(t_values), y_values])
    drift_points = np.column_stack([t_values, np.ones_like(t_values), g_scaled])

    risk_line = pv.lines_from_points(risk_points, close=False).tube(radius=0.05)
    drift_line = pv.lines_from_points(drift_points, close=False).tube(radius=0.05)
    plotter.add_mesh(risk_line, color=COLOR_HIGH, opacity=0.9)
    plotter.add_mesh(drift_line, color=COLOR_LOW, opacity=0.9)

    if show_y_ci:
        risk_ci = _create_ci_ribbon_surface(
            t_values,
            0.0,
            np.clip(y_values - y_ci, 0.0, 1.0),
            np.clip(y_values + y_ci, 0.0, 1.0),
        )
        plotter.add_mesh(risk_ci, color=COLOR_HIGH, opacity=0.75)
    if show_g_ci:
        drift_ci = _create_ci_ribbon_surface(
            t_values,
            1.0,
            np.clip(g_scaled - g_ci, 0.0, 1.0),
            np.clip(g_scaled + g_ci, 0.0, 1.0),
        )
        plotter.add_mesh(drift_ci, color=COLOR_LOW, opacity=0.75)

    risk_surface = _create_under_curve_surface(t_values, np.zeros_like(t_values), y_values)
    drift_surface = _create_under_curve_surface(t_values, np.ones_like(t_values), g_scaled)
    plotter.add_mesh(risk_surface, color=COLOR_HIGH, opacity=0.33)
    plotter.add_mesh(drift_surface, color=COLOR_LOW, opacity=0.33)

    _set_camera_head_on(plotter, pad=3.4, tilt=0.10)
    _save_plotter_image(plotter, asset_dir / "mesh_timeseries_3d.png")
    plotter.close()

    _create_latex_sticker(r"$Y_t$", asset_dir / "text_sticker_Yt.png", COLOR_HIGH, fontsize=30)
    _create_latex_sticker(r"$\Vert g_t \Vert_2$", asset_dir / "text_sticker_gt.png", COLOR_LOW, fontsize=28)
    if show_y_ci or show_g_ci:
        _create_latex_sticker(r"$95\%\ \mathrm{CI}$", asset_dir / "text_sticker_ci.png", COLOR_SECONDARY, fontsize=24)
    _create_latex_sticker(r"$\Delta t$", asset_dir / "text_sticker_t.png", COLOR_ACCENT, fontsize=24)

    pd.DataFrame(
        {
            "t": t_values,
            "y_t": y_values,
            "g_norm": g_values,
            "delta_y": table["delta_y_large_outbreak_prob"].to_numpy(),
            "y_ci": y_ci,
            "g_ci_scaled": g_ci,
            "show_y_ci": np.repeat(show_y_ci, len(t_values)),
            "show_g_ci": np.repeat(show_g_ci, len(t_values)),
        }
    ).to_csv(asset_dir / "text_timeseries.csv", index=False)
    pd.DataFrame([ci_diagnostics]).to_csv(asset_dir / "text_ci_diagnostics.csv", index=False)


def generate_drift_scatter_assets(table: pd.DataFrame, asset_dir: Path, dataset_label: str) -> None:
    _clear_asset_dir(asset_dir)
    x = table["g_l2_norm"].to_numpy()
    y = table["y_large_outbreak_prob"].to_numpy()
    z = table["delta_y_large_outbreak_prob"].to_numpy()

    _render_scatter_persistence_style(
        x=x,
        y=y,
        z=z,
        output_path=asset_dir / "mesh_scatter_3d.png",
    )

    _create_latex_sticker(r"$\Vert g_t \Vert_2$", asset_dir / "text_sticker_x.png", COLOR_LOW, fontsize=24)
    _create_latex_sticker(r"$Y_t$", asset_dir / "text_sticker_y.png", COLOR_HIGH, fontsize=24)
    _create_latex_sticker(r"$\Delta Y_t$", asset_dir / "text_sticker_z.png", COLOR_SECONDARY, fontsize=24)
    _ = dataset_label
    pd.DataFrame({"g_norm": x, "y_t": y, "delta_y": z}).to_csv(asset_dir / "text_scatter.csv", index=False)
    pd.DataFrame(
        [
            {
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "z_min": float(np.min(z)),
                "z_max": float(np.max(z)),
            }
        ]
    ).to_csv(asset_dir / "text_scatter_debug_ranges.csv", index=False)


def generate_rkhs_trajectory_assets(table: pd.DataFrame, features: np.lib.npyio.NpzFile, asset_dir: Path) -> None:
    _clear_asset_dir(asset_dir)
    rkhs = features["rkhs_g_t"]
    y_vals = table["y_large_outbreak_prob"].to_numpy()
    n = int(min(rkhs.shape[0], y_vals.shape[0]))
    if n <= 1:
        return
    rkhs = rkhs[:n]
    y_vals = y_vals[:n]
    pca = PCA(n_components=3, random_state=14, whiten=True)
    coords = pca.fit_transform(rkhs)
    coords = coords * 3.0

    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    points = pv.PolyData(coords)
    points["risk"] = y_vals
    mesh = points.glyph(scale=False, orient=False, geom=pv.Sphere(radius=0.08))
    plotter.add_mesh(mesh, scalars="risk", cmap="coolwarm", opacity=0.9, show_scalar_bar=False)
    for idx in range(coords.shape[0] - 1):
        p0 = coords[idx]
        p1 = coords[idx + 1]
        s = float(0.5 * (y_vals[idx] + y_vals[idx + 1]))
        color = _blend_color(COLOR_LOW, COLOR_HIGH, s)
        seg = pv.Line(tuple(p0), tuple(p1)).tube(radius=0.02)
        plotter.add_mesh(seg, color=color, opacity=0.88)
    _set_camera_head_on(plotter, pad=4.8, tilt=0.18)
    _save_plotter_image(plotter, asset_dir / "mesh_rkhs_trajectory.png")
    plotter.close()

    points_only = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    points_only.set_background([0, 0, 0, 0])
    points_only.add_mesh(mesh, scalars="risk", cmap="coolwarm", opacity=0.90, show_scalar_bar=False)
    _set_camera_head_on(points_only, pad=4.8, tilt=0.18)
    _save_plotter_image(points_only, asset_dir / "mesh_rkhs_points_only.png")
    points_only.close()

    _create_latex_sticker(r"$\Phi(g_t)$", asset_dir / "text_sticker_rkhs.png", COLOR_SECONDARY, fontsize=28)

    pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2], "risk": y_vals}
    ).to_csv(asset_dir / "text_rkhs_trajectory.csv", index=False)
    pd.DataFrame(
        [
            {
                "x_span": float(np.max(coords[:, 0]) - np.min(coords[:, 0])),
                "y_span": float(np.max(coords[:, 1]) - np.min(coords[:, 1])),
                "z_span": float(np.max(coords[:, 2]) - np.min(coords[:, 2])),
            }
        ]
    ).to_csv(asset_dir / "text_rkhs_debug_ranges.csv", index=False)


def generate_persistence_regime_assets(
    table: pd.DataFrame,
    features: np.lib.npyio.NpzFile,
    asset_dir: Path,
    dataset_key: str,
    min_events_per_window: int,
    preprocessed_cache_dir: Path | None = None,
) -> None:
    _clear_asset_dir(asset_dir)
    d_values = features["d_t"]
    grid_size = int(np.sqrt(d_values.shape[1] // 2))
    idx_low = int(np.argmin(table["y_large_outbreak_prob"].to_numpy()))
    idx_high = int(np.argmax(table["y_large_outbreak_prob"].to_numpy()))
    idx_transition = int(np.argmax(np.abs(table["delta_y_large_outbreak_prob"].to_numpy())))
    windows = _load_windows_aligned_with_table(table, dataset_key, min_events_per_window, preprocessed_cache_dir)
    regimes = [("low", idx_low), ("transition", idx_transition), ("high", idx_high)]
    regime_rows = []

    for label, idx in regimes:
        safe_idx = int(np.clip(idx, 0, d_values.shape[0] - 1))
        next_idx = int(np.clip(safe_idx + 1, 0, d_values.shape[0] - 1))
        if next_idx == safe_idx and safe_idx > 0:
            next_idx = safe_idx - 1
        diag_t = _vpd_vector_to_diagram(d_values[safe_idx], grid_size)
        diag_next = _vpd_vector_to_diagram(d_values[next_idx], grid_size)

        render_persistence_diagram_3d(
            diag_t,
            asset_dir / f"mesh_{label}_persistence_3d.png",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            elev=25,
            azim=-60,
            bar_scale=1.0,
        )
        render_virtual_persistence_diagram_3d(
            diag_next,
            diag_t,
            asset_dir / f"mesh_{label}_virtual_persistence_3d.png",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            elev=25,
            azim=-60,
            bar_scale=1.0,
        )
        t_w = _table_row_to_window_index(table, safe_idx)
        edges = aggregate_window_edges(windows[t_w])
        _render_network_asset(edges, asset_dir / f"mesh_{label}_network_3d.png")
        regime_rows.append(
            {
                "label": label,
                "t": safe_idx,
                "window_t": t_w,
                "num_nodes": len(set(edges["source"]).union(set(edges["target"]))),
                "num_edges": len(edges),
            }
        )
    pd.DataFrame(regime_rows).to_csv(asset_dir / "text_regime_summary.csv", index=False)


def generate_temporal_network_3d_assets(
    table: pd.DataFrame,
    asset_dir: Path,
    dataset_key: str,
    min_events_per_window: int,
    preprocessed_cache_dir: Path | None = None,
) -> None:
    if pv is None:
        raise RuntimeError("PyVista is required for temporal network meshes.")
    _clear_asset_dir(asset_dir)
    windows = _load_windows_aligned_with_table(table, dataset_key, min_events_per_window, preprocessed_cache_dir)
    mid = len(table) // 2
    t_mid = _table_row_to_window_index(table, mid)
    edges_mid = aggregate_window_edges(windows[t_mid])
    _render_network_asset(edges_mid, asset_dir / "mesh_representative_network_3d.png")
    idx_low = int(np.argmin(table["y_large_outbreak_prob"].to_numpy()))
    idx_high = int(np.argmax(table["y_large_outbreak_prob"].to_numpy()))
    idx_transition = int(np.argmax(np.abs(table["delta_y_large_outbreak_prob"].to_numpy())))
    rows: list[dict[str, object]] = [
        {
            "label": "representative",
            "table_row": mid,
            "window_t": t_mid,
            "num_nodes": len(set(edges_mid["source"]).union(set(edges_mid["target"]))),
            "num_edges": len(edges_mid),
        }
    ]
    for label, idx in (("low", idx_low), ("transition", idx_transition), ("high", idx_high)):
        safe_idx = int(np.clip(idx, 0, len(table) - 1))
        t_w = _table_row_to_window_index(table, safe_idx)
        edges = aggregate_window_edges(windows[t_w])
        _render_network_asset(edges, asset_dir / f"mesh_{label}_network_3d.png")
        rows.append(
            {
                "label": label,
                "table_row": safe_idx,
                "window_t": t_w,
                "num_nodes": len(set(edges["source"]).union(set(edges["target"]))),
                "num_edges": len(edges),
            }
        )
    pd.DataFrame(rows).to_csv(asset_dir / "text_network_summary.csv", index=False)


def generate_model_comparison_assets(model_output_dir: Path, asset_dir: Path) -> None:
    _clear_asset_dir(asset_dir)
    files = [
        ("tgn_baseline_predictions.csv", "TGN"),
        ("tgn_landscape_constraint_predictions.csv", "Landscape"),
        ("tgn_rkhs_constraint_predictions.csv", "RKHS"),
    ]
    rows = []
    for filename, model in files:
        path = model_output_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rmse = float(np.sqrt(np.mean((df["y_pred"] - df["y_true"]) ** 2)))
        rows.append({"model": model, "rmse": rmse})
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(asset_dir / "text_model_scores.csv", index=False)
    model_order = ["TGN", "Landscape", "RKHS"]
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")
    colors = {"TGN": COLOR_LOW, "Landscape": COLOR_SECONDARY, "RKHS": COLOR_HIGH}

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=FIGURE_DPI, facecolor="none")
    x = np.arange(len(summary))
    y = summary["rmse"].to_numpy(dtype=float)
    c = [colors[str(m)] for m in summary["model"]]
    bars = ax.bar(x, y, color=c, alpha=0.88, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"].astype(str).tolist())
    ax.set_ylabel("RMSE")
    ax.set_ylim(0.0, max(float(np.max(y)) * 1.15, 1e-6))
    ax.grid(axis="y", linestyle=":", alpha=0.30)
    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() * 0.5,
            float(value) + 0.01 * max(float(np.max(y)), 1.0),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )
    fig.savefig(asset_dir / "curve_model_rmse_2d.png", transparent=True, bbox_inches="tight")
    plt.close(fig)

    _create_latex_sticker(r"$\mathrm{RMSE}$", asset_dir / "text_sticker_rmse.png", COLOR_ACCENT, fontsize=22)


def _write_pipeline_loss_function_stickers(asset_dir: Path) -> None:
    loss_dir = asset_dir / "loss-functions"
    loss_dir.mkdir(parents=True, exist_ok=True)
    for path in loss_dir.glob("*.png"):
        path.unlink(missing_ok=True)
    for legacy_name in (
        "text_sticker_loss_scope_plain.png",
        "text_sticker_loss_task_mse.png",
        "text_sticker_loss_baseline.png",
        "text_sticker_loss_landscape_align.png",
        "text_sticker_loss_landscape_total.png",
        "text_sticker_loss_rkhs_align.png",
        "text_sticker_loss_rkhs_total.png",
    ):
        stale = asset_dir / legacy_name
        if stale.is_file():
            stale.unlink()

    _create_latex_sticker(
        r"$\frac{1}{\vert I_{\mathrm{tr}}\vert}\sum_{i\in I_{\mathrm{tr}}}(\hat y_i-y_i)^2"
        r"+\lambda_d\frac{1}{\vert I_{\mathrm{tr}}\vert}\sum_{i\in I_{\mathrm{tr}}}\Vert h_i-z_{d,i}\Vert_2^2"
        r"+\lambda_g\frac{1}{\vert I_{\mathrm{tr}}\vert}\sum_{i\in I_{\mathrm{tr}}}\Vert W(h_i)-r_i\Vert_2^2,\ "
        r"r_i=\mathrm{RFF}(g_t)_i$",
        loss_dir / "loss_train_full.png",
        COLOR_ACCENT,
        fontsize=10,
        fig_width=14.5,
        fig_height=1.25,
    )
    _create_latex_sticker(
        r"$\mathrm{TGN}$",
        loss_dir / "label_TGN.png",
        COLOR_LOW,
        fontsize=30,
        fig_width=2.0,
    )
    _create_latex_sticker(
        r"$\mathrm{Landscape}$",
        loss_dir / "label_Landscape.png",
        COLOR_SECONDARY,
        fontsize=22,
        fig_width=3.6,
    )
    _create_latex_sticker(
        r"$\mathrm{RKHS}$",
        loss_dir / "label_RKHS.png",
        COLOR_HIGH,
        fontsize=30,
        fig_width=2.2,
    )


def write_pipeline_overview_stickers(table_csv: Path, features_npz: Path, asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in ("text_sticker_psi_embed.png", "text_sticker_z_rkhs_def.png"):
        o_path = asset_dir / obsolete
        if o_path.is_file():
            o_path.unlink()
    steps = [
        {"name": "events_to_temporal_graph", "input": str(table_csv), "output": "G_t"},
        {"name": "temporal_graph_3d_mesh", "input": "G_t (one window)", "output": "mesh_network_3d.png"},
        {"name": "weighted_filtration", "input": "G_t", "output": "K_t^epsilon"},
        {"name": "persistence_diagram", "input": "K_t^epsilon", "output": "D_t"},
        {"name": "persistence_landscape_grid", "input": "D_t", "output": "L_t"},
        {"name": "diagram_to_vpd_vector", "input": "D_t", "output": "v_t in R^m"},
        {"name": "virtual_drift_jet", "input": "consecutive v", "output": "g_t, a_t"},
        {"name": "tgn_window_pool", "input": "G_t", "output": "h_t"},
        {"name": "landscape_mlp", "input": "L_t", "output": "z_d"},
        {"name": "rkhs_features", "input": "g_t", "output": "RFF(g_t)"},
        {"name": "predict_head", "input": "h_t and optional z_d", "output": "y_hat"},
        {"name": "train_loss", "input": "train indices", "output": "loss-functions/"},
        {"name": "notation_vpd_heads", "input": "pipeline", "output": "other_improvements/"},
        {"name": "arrow_mesh_library", "input": "PyVista", "output": "arrow_shapes/"},
        {"name": "cached_features", "input": str(table_csv), "output": str(features_npz)},
    ]
    pd.DataFrame(steps).to_csv(asset_dir / "text_pipeline_steps.csv", index=False)

    oi = asset_dir / "other_improvements"
    oi.mkdir(parents=True, exist_ok=True)
    for path in oi.glob("*.png"):
        path.unlink(missing_ok=True)
    for root_dup in (
        "text_sticker_task_target_plain.png",
        "text_sticker_label_Y_t.png",
        "text_sticker_vpd_vector.png",
        "text_sticker_gt_eq.png",
        "text_sticker_second_order_jet.png",
        "text_sticker_temporal_jet.png",
        "text_sticker_Dt_space.png",
        "text_sticker_bigraded_jet.png",
        "text_sticker_jet_metric.png",
        "text_sticker_RKHS_label.png",
        "text_sticker_rkhs_kernel.png",
        "text_sticker_rkhs_measure.png",
        "text_sticker_rkhs_feature.png",
        "text_sticker_z_top.png",
        "text_sticker_predict_head_baseline.png",
        "text_sticker_predict_head_landscape.png",
        "text_sticker_predict_sigma.png",
        "text_sticker_target.png",
        "text_sticker_baseline_mlp.png",
        "text_sticker_model_landscape.png",
        "text_sticker_model_rkhs_name.png",
    ):
        stale_root = asset_dir / root_dup
        if stale_root.is_file():
            stale_root.unlink()

    # Data and window object
    _create_text_sticker("Infectious: 24h windows", asset_dir / "text_sticker_data_windows.png", "#333333")
    _create_text_sticker(
        "Supervised target y_i: table y_large_outbreak_prob (window risk in [0,1]).",
        oi / "text_sticker_task_target_plain.png",
        "#222222",
        fig_width=9.2,
        fig_height=1.05,
        fontsize=11,
    )
    _create_latex_sticker(r"$G_t=(V_t,E_t,\mathcal{T}_t)$", asset_dir / "text_sticker_Gt.png", COLOR_HIGH, fontsize=20, fig_width=4.6)
    _create_latex_sticker(r"$(A_t)_{uv}=w_t(u,v)$", asset_dir / "text_sticker_At_weights.png", COLOR_HIGH, fontsize=18, fig_width=3.8)
    _create_latex_sticker(
        r"$\mathcal{X}_t=(G_t,D_t,g_t,Y_t)$", asset_dir / "text_sticker_window_tuple.png", COLOR_ACCENT, fontsize=17, fig_width=5.4
    )
    _create_latex_sticker(
        r"$Y_t=\mathbb{P}(A_t\geq\tau),\ \tau=0.20$", oi / "text_sticker_label_Y_t.png", COLOR_HIGH, fontsize=17, fig_width=6.2
    )
    # Weighted clique filtration and homology
    _create_latex_sticker(
        r"$f_t(e)=1-\frac{w_t(e)}{\max_{e'}w_t(e')}$", asset_dir / "text_sticker_filtration_f.png", COLOR_SECONDARY, fontsize=15, fig_width=6.8
    )
    _create_latex_sticker(
        r"$K_t^{\epsilon}=\mathrm{Clique}(\{e:f_t(e)\leq\epsilon\})$",
        asset_dir / "text_sticker_Kepsilon.png",
        COLOR_SECONDARY,
        fontsize=14,
        fig_width=8.2,
    )
    _create_latex_sticker(r"$D_t^{(r)}=\{(b_i,d_i)\}_i$", asset_dir / "text_sticker_diagram_Dt.png", COLOR_HIGH, fontsize=18, fig_width=4.2)
    _create_latex_sticker(r"$D_t=\bigcup_r D_t^{(r)}$", asset_dir / "text_sticker_diagram_union.png", COLOR_HIGH, fontsize=17, fig_width=4.0)
    # Bubenik persistence landscapes (replace PersLay-style summaries)
    _create_latex_sticker(
        r"$\lambda_i(x)=\max(0,\min(x-b_i,d_i-x))$", asset_dir / "text_sticker_landscape_tent.png", COLOR_ACCENT, fontsize=16, fig_width=7.0
    )
    _create_latex_sticker(
        r"$\Lambda_k(x)=\mathrm{order\ stat}_k(\lambda_i(x))$",
        asset_dir / "text_sticker_landscape_Lambda_k.png",
        COLOR_ACCENT,
        fontsize=15,
        fig_width=7.2,
    )
    _create_latex_sticker(
        r"$L_t\in\mathbb{R}^{K\times m},\ (L_t)_{k,j}=\Lambda_k(x_j)$", asset_dir / "text_sticker_Lt_grid.png", COLOR_ACCENT, fontsize=14, fig_width=8.8
    )
    _create_latex_sticker(
        r"$v_t=\mathrm{VPD}(D_t)\in\mathbb{R}^m$", oi / "text_sticker_vpd_vector.png", COLOR_LOW, fontsize=16, fig_width=5.2
    )
    _create_latex_sticker(
        r"$g_t=v_{t+\Delta}-v_t$", oi / "text_sticker_gt_eq.png", COLOR_LOW, fontsize=18, fig_width=4.2
    )
    _create_latex_sticker(
        r"$a_t=v_{t+2\Delta}-2v_{t+\Delta}+v_t$",
        oi / "text_sticker_second_order_jet.png",
        COLOR_LOW,
        fontsize=15,
        fig_width=5.6,
    )
    _create_latex_sticker(
        r"$\Delta_{\tau}^{k}v_t=\Delta_{\tau}(\Delta_{\tau}^{k-1}v_t)$",
        oi / "text_sticker_temporal_jet.png",
        COLOR_SECONDARY,
        fontsize=15,
        fig_width=7.2,
    )
    _create_latex_sticker(r"$D_t\in\mathcal{H}$", oi / "text_sticker_Dt_space.png", COLOR_HIGH, fontsize=22, fig_width=3.2)
    _create_latex_sticker(
        r"$J_t=\left(v_t,\Delta_\tau v_t,\ldots,\Delta_\tau^k v_t\right)$",
        oi / "text_sticker_bigraded_jet.png",
        COLOR_HIGH,
        fontsize=14,
        fig_width=9.5,
        fig_height=1.0,
    )
    _create_latex_sticker(
        r"$d_J^{(k)}(t,s)=\sum_{j=0}^{k}\omega_j\Vert\Delta_\tau^j v_t-\Delta_\tau^j v_s\Vert_{\mathcal{H}}$",
        oi / "text_sticker_jet_metric.png",
        COLOR_ACCENT,
        fontsize=12,
        fig_width=11.5,
        fig_height=1.05,
    )
    # TGN (event-driven TGNN)
    _create_latex_sticker(r"$\mathrm{TGN}$", asset_dir / "text_sticker_TGN.png", COLOR_LOW, fontsize=26, fig_width=2.2)
    _create_latex_sticker(
        r"$m=\mathrm{msg}(s_u,s_v,\Delta\tau,e_{uv})$", asset_dir / "text_sticker_tgn_message.png", COLOR_LOW, fontsize=15, fig_width=6.6
    )
    _create_latex_sticker(
        r"$\bar{m}_i=\mathrm{Agg}\{m_i(\tau_k)\colon\tau_k\leq t\}$", asset_dir / "text_sticker_tgn_agg.png", COLOR_LOW, fontsize=13, fig_width=7.6
    )
    _create_latex_sticker(r"$s_i=\mathrm{GRU}(\bar{m}_i,s_i^-)$", asset_dir / "text_sticker_tgn_gru.png", COLOR_LOW, fontsize=16, fig_width=5.0)
    _create_latex_sticker(
        r"$h_t=\frac{1}{\vert V_t\vert}\sum_{i\in V_t}z_i(t)$", asset_dir / "text_sticker_tgn_pool.png", COLOR_LOW, fontsize=15, fig_width=6.2
    )
    # RKHS features (random Fourier features of drift g_t in code; not the same map as VPD(D))
    _create_latex_sticker(r"$\mathrm{RKHS}$", oi / "text_sticker_RKHS_label.png", COLOR_SECONDARY, fontsize=22, fig_width=2.8)
    _create_latex_sticker(
        r"$k_{\nu_t}(\alpha,\beta)=\langle\Phi_{\nu_t}(\alpha),\Phi_{\nu_t}(\beta)\rangle_{\mathcal{H}_{\nu_t}}$",
        oi / "text_sticker_rkhs_kernel.png",
        COLOR_SECONDARY,
        fontsize=12,
        fig_width=11.0,
        fig_height=1.05,
    )
    _create_latex_sticker(
        r"$d\nu_t(\theta)\propto e^{-t\lambda(\theta)}\,d\mu(\theta)$", oi / "text_sticker_rkhs_measure.png", COLOR_SECONDARY, fontsize=13, fig_width=7.8
    )
    _create_latex_sticker(r"$\mathrm{RFF}(g_t)$", oi / "text_sticker_rkhs_feature.png", COLOR_SECONDARY, fontsize=20, fig_width=3.4)
    # Heads (pair with Baseline / Landscape stickers; no model names in the math)
    _create_latex_sticker(
        r"$z_{d,i}=\mathrm{MLP}(L_{t,i})$", oi / "text_sticker_z_top.png", COLOR_HIGH, fontsize=15, fig_width=4.8
    )
    _create_latex_sticker(
        r"$\hat y_i=\sigma(f(h_i))$", oi / "text_sticker_predict_head_baseline.png", COLOR_ACCENT, fontsize=16, fig_width=4.6
    )
    _create_latex_sticker(
        r"$\hat y_i=\sigma(f([h_i,z_{d,i}]))$", oi / "text_sticker_predict_head_landscape.png", COLOR_ACCENT, fontsize=15, fig_width=6.2
    )
    _create_latex_sticker(r"$\hat y_i\in[0,1]$", oi / "text_sticker_predict_sigma.png", COLOR_ACCENT, fontsize=17, fig_width=3.2)
    _create_latex_sticker(r"$y_i=\mathbb{P}(A_i\geq\tau),\ \tau=0.20$", oi / "text_sticker_target.png", COLOR_ACCENT, fontsize=16, fig_width=6.4)
    _write_pipeline_loss_function_stickers(asset_dir)
    # Evaluation metrics
    _create_latex_sticker(
        r"$\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_t(\hat{Y}_t-Y_t)^2}$", asset_dir / "text_sticker_metric_rmse.png", "#333333", fontsize=13, fig_width=8.2
    )
    _create_latex_sticker(
        r"$\mathrm{Brier}=\frac{1}{n}\sum_t(\hat{Y}_t-Y_t^{\mathrm{bin}})^2$",
        asset_dir / "text_sticker_metric_brier.png",
        "#333333",
        fontsize=12,
        fig_width=8.8,
    )
    _create_latex_sticker(
        r"$\mathrm{ECE}=\sum_b\frac{\vert I_b\vert}{n}\vert\mathrm{acc}_b-\mathrm{conf}_b\vert$",
        asset_dir / "text_sticker_metric_ece.png",
        "#333333",
        fontsize=11,
        fig_width=9.6,
        fig_height=1.05,
    )
    # Model labels (diagram functor stack)
    _create_latex_sticker(r"$\mathrm{Baseline}$", oi / "text_sticker_baseline_mlp.png", COLOR_LOW, fontsize=20, fig_width=3.0)
    _create_latex_sticker(r"$\mathrm{Landscape}$", oi / "text_sticker_model_landscape.png", COLOR_SECONDARY, fontsize=20, fig_width=3.4)
    _create_latex_sticker(r"$\mathrm{RKHS}$", oi / "text_sticker_model_rkhs_name.png", COLOR_HIGH, fontsize=20, fig_width=2.6)
    _create_latex_sticker(
        r"$D^{(n+1)}(X_t)=D(D^{(n)}(X_t))$", asset_dir / "text_sticker_structural_jet.png", COLOR_SECONDARY, fontsize=17, fig_width=6.4
    )


def generate_pipeline_spec_assets(
    table_csv: Path,
    features_npz: Path,
    asset_dir: Path,
    *,
    dataset_key: str = PRIMARY_SCHOOL_DATASET_KEY,
    min_events_per_window: int = 20,
    preprocessed_cache_dir: Path | None = None,
) -> None:
    if pv is None:
        raise RuntimeError("PyVista is required for pipeline mesh assets.")
    _clear_asset_dir(asset_dir)
    write_pipeline_overview_stickers(table_csv, features_npz, asset_dir)
    _render_single_box_asset(asset_dir / "mesh_box_blue.png", COLOR_LOW)
    _render_single_box_asset(asset_dir / "mesh_box_red.png", COLOR_HIGH)
    _render_gradient_arc_arrow(asset_dir / "mesh_arrow_blue_to_red.png", COLOR_LOW, COLOR_HIGH)
    _render_gradient_arc_arrow(asset_dir / "mesh_arrow_red_to_blue.png", COLOR_HIGH, COLOR_LOW)
    _render_gradient_arc_arrow(asset_dir / "mesh_arrow_blue_to_red_shallow.png", COLOR_LOW, COLOR_HIGH, curvature=0.14, height=0.10)
    _render_gradient_arc_arrow(asset_dir / "mesh_arrow_blue_to_red_deep.png", COLOR_LOW, COLOR_HIGH, curvature=0.42, height=0.28)
    _render_gradient_s_curve_arrow(asset_dir / "mesh_arrow_blue_to_red_scurve.png", COLOR_LOW, COLOR_HIGH)
    _render_gradient_s_curve_arrow(asset_dir / "mesh_arrow_red_to_blue_scurve.png", COLOR_HIGH, COLOR_LOW)
    _render_gradient_spiral_arrow(asset_dir / "mesh_arrow_blue_to_red_spiral.png", COLOR_LOW, COLOR_HIGH)
    _render_gradient_spiral_arrow(asset_dir / "mesh_arrow_red_to_blue_spiral.png", COLOR_HIGH, COLOR_LOW)
    _write_pipeline_arrow_shape_assets(asset_dir / "arrow_shapes")
    table_pipe = pd.read_csv(table_csv)
    _assert_table_features_row_alignment(table_pipe, features_npz)
    pipe_edges = _pick_pipeline_network_window_edges(
        table_pipe, dataset_key, min_events_per_window, preprocessed_cache_dir
    )
    _render_network_asset(pipe_edges, asset_dir / "mesh_network_3d.png")


def generate_collective_assets(output_root: Path, collective_metrics_csv: Path, collective_timeseries_csv: Path) -> None:
    if collective_metrics_csv.exists():
        metrics = pd.read_csv(collective_metrics_csv)
        _generate_collective_model_comparison(output_root / "model_comparison_3d" / "assets", metrics)
        _generate_collective_results_table(output_root / "results_table_figure" / "assets", metrics)
    if collective_timeseries_csv.exists():
        ts = pd.read_csv(collective_timeseries_csv)
        _generate_collective_timeseries(output_root / "drift_risk_timeseries_3d" / "assets", ts)


def total_level_collective_metric_paths(project_root: Path) -> list[tuple[str, Path]]:
    return [
        ("Base", project_root / "results" / "output" / "collective_metrics.csv"),
        ("Level 1", project_root / "results" / "level1" / "output" / "collective_metrics.csv"),
        ("Level 2", project_root / "results" / "level2" / "output" / "collective_metrics.csv"),
    ]


def load_total_level_metrics_frame(project_root: Path) -> pd.DataFrame:
    specs = total_level_collective_metric_paths(project_root)
    frames: list[pd.DataFrame] = []
    for level_label, path in specs:
        if not path.is_file():
            raise RuntimeError(f"missing collective metrics for {level_label}: {path}")
        part = pd.read_csv(path)
        part = part.copy()
        part.insert(0, "level", level_label)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def generate_total_level_figure_assets(project_root: Path | None = None) -> None:
    root = project_root if project_root is not None else PROJECT_ROOT
    metrics = load_total_level_metrics_frame(root)
    out_root = root / "results" / "figures" / "total"
    _generate_total_collective_model_comparison(out_root / "model_comparison_3d" / "assets", metrics)
    _generate_total_collective_results_table(out_root / "results_table_figure" / "assets", metrics)
    print(f"Generated total (Base + Level 1 + Level 2) figures under {out_root}")


def _generate_collective_timeseries(asset_dir: Path, collective_ts: pd.DataFrame) -> None:
    _clear_asset_dir(asset_dir)
    if collective_ts.empty:
        return
    ds = collective_ts.copy()
    if "dataset" in ds.columns:
        ds = ds[ds["dataset"].astype(str).str.lower() == "infectious"].copy()
    ds = ds.sort_values("t")
    if ds.empty:
        return
    if len(ds) <= 1:
        x = np.zeros(len(ds), dtype=float)
    else:
        x = np.linspace(0.0, 1.0, len(ds))
    y = ds["y_t"].to_numpy(dtype=float)
    if "y_ci" in ds.columns:
        y_ci = ds["y_ci"].to_numpy(dtype=float)
    else:
        y_ci = 1.96 * np.sqrt(np.maximum(y * (1.0 - y), 0.0) / 100.0)
    y_lo = np.clip(y - y_ci, 0.0, 1.0)
    y_hi = np.clip(y + y_ci, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=FIGURE_DPI, facecolor="none")
    ax.fill_between(x, y_lo, y_hi, color=COLOR_HIGH, alpha=0.22, linewidth=0.0, zorder=1)
    ax.plot(x, y, color=COLOR_HIGH, linewidth=4.0, alpha=0.96, zorder=3)
    ax.plot(x, y_lo, color=COLOR_HIGH, linewidth=1.2, alpha=0.65, zorder=2)
    ax.plot(x, y_hi, color=COLOR_HIGH, linewidth=1.2, alpha=0.65, zorder=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel(r"$Y_t$")
    ax.grid(alpha=0.22, linestyle=":", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(asset_dir / "curve_infectious_timeseries_2d.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    _create_latex_sticker(r"$Y_t\ \pm\ 95\%\ \mathrm{CI}$", asset_dir / "text_sticker_Yt.png", COLOR_HIGH, fontsize=20, fig_width=3.8)


def _generate_collective_model_comparison(asset_dir: Path, metrics: pd.DataFrame) -> None:
    _clear_asset_dir(asset_dir)
    if metrics.empty:
        return
    model_order = ["TGN", "Landscape", "RKHS"]
    metrics = metrics[metrics["dataset"].astype(str).str.lower() == "infectious"].copy()
    datasets = list(dict.fromkeys(metrics["dataset"].tolist()))
    colors = {"TGN": COLOR_LOW, "Landscape": COLOR_SECONDARY, "RKHS": COLOR_HIGH}

    pivot = metrics.pivot(index="dataset", columns="model", values="rmse").reindex(index=datasets, columns=model_order)
    if pivot.isna().any().any():
        missing = pivot.isna()
        missing_pairs = [(datasets[i], model_order[j]) for i, j in np.argwhere(missing.to_numpy())]
        raise ValueError(f"missing collective model metrics rows: {missing_pairs}")

    x = np.arange(len(datasets), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIGURE_DPI, facecolor="none")
    for idx, model in enumerate(model_order):
        y = pivot[model].to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, y, width=width, label=model, color=colors[model], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(asset_dir / "curve_infectious_model_rmse_2d.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    metrics.to_csv(asset_dir / "text_infectious_model_metrics.csv", index=False)


def _generate_collective_results_table(asset_dir: Path, metrics: pd.DataFrame) -> None:
    _clear_asset_dir(asset_dir)
    if metrics.empty:
        return
    metrics = metrics[metrics["dataset"].astype(str).str.lower() == "infectious"].copy()
    rows = []
    model_order = ["TGN", "Landscape", "RKHS"]
    for model in model_order:
        row = metrics[metrics["model"] == model]
        if row.empty:
            continue
        if "brier" not in row.columns:
            raise ValueError("collective metrics is missing required column 'brier'")
        if "ece" not in row.columns:
            raise ValueError("collective metrics is missing required column 'ece'")
        if "eval_count" not in row.columns:
            raise ValueError("collective metrics is missing required column 'eval_count'")
        rows.append(
            {
                "Model": model,
                "RMSE": float(row.iloc[0]["rmse"]),
                "Brier": float(row.iloc[0]["brier"]),
                "ECE": float(row.iloc[0]["ece"]),
                "EvalCount": float(row.iloc[0]["eval_count"]),
            }
        )
    table_df = pd.DataFrame(rows)
    table_df.to_csv(asset_dir / "text_infectious_results_table.csv", index=False)
    _render_collective_results_table_strict(table_df, asset_dir / "table_infectious_results_main.png")


def _render_collective_results_table_strict(table_df: pd.DataFrame, output_path: Path) -> None:
    required_cols = {"Model", "RMSE", "Brier", "ECE"}
    missing = required_cols.difference(set(table_df.columns))
    if missing:
        raise ValueError(f"missing collective table columns: {sorted(missing)}")

    model_order = ["TGN", "Landscape", "RKHS"]
    table_df["Model"] = pd.Categorical(table_df["Model"], categories=model_order, ordered=True)
    table_df = table_df.sort_values("Model").reset_index(drop=True)
    if len(table_df) != 3:
        raise ValueError(f"expected exactly 3 model rows, found {len(table_df)}")

    col_widths = [1.8, 1.5, 1.5, 1.5]
    row_h = 0.60
    header_h = 0.70
    n_rows = len(table_df)
    fig_w = sum(col_widths)
    fig_h = header_h + n_rows * row_h + 0.30
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIGURE_DPI, facecolor="none")
    ax.set_xlim(0.0, fig_w)
    ax.set_ylim(0.0, fig_h)
    ax.axis("off")

    x0 = 0.0
    y_top = fig_h - 0.15
    headers = ["Model", "RMSE", "Brier", "ECE"]
    for col_idx, (w, label) in enumerate(zip(col_widths, headers)):
        rect = plt.Rectangle((x0, y_top - header_h), w, header_h, facecolor=(0.85, 0.85, 0.85, 0.98), edgecolor="#222222", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x0 + w * 0.5, y_top - header_h * 0.5, label, ha="center", va="center", fontsize=11, fontweight="bold")
        x0 += w

    current_y = y_top - header_h
    best_rmse = float(table_df["RMSE"].min())
    for _, row in table_df.iterrows():
        entries = [str(row["Model"]), f"{float(row['RMSE']):.3f}", f"{float(row['Brier']):.3f}", f"{float(row['ECE']):.3f}"]
        x = 0.0
        is_best = abs(float(row["RMSE"]) - best_rmse) < 1e-12
        for idx, txt in enumerate(entries):
            w = col_widths[idx]
            fill = (0.93, 0.97, 0.93, 0.95) if is_best else (1.0, 1.0, 1.0, 0.90)
            rect = plt.Rectangle((x, current_y - row_h), w, row_h, facecolor=fill, edgecolor="#222222", linewidth=1.0)
            ax.add_patch(rect)
            fw = "bold" if is_best and idx in (0, 1) else "normal"
            ax.text(x + w * 0.5, current_y - row_h * 0.5, txt, ha="center", va="center", fontsize=10, fontweight=fw)
            x += w
        current_y -= row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _generate_total_collective_model_comparison(asset_dir: Path, metrics: pd.DataFrame) -> None:
    _clear_asset_dir(asset_dir)
    if metrics.empty:
        return
    model_order = ["TGN", "Landscape", "RKHS"]
    level_order = ["Base", "Level 1", "Level 2"]
    colors = {"TGN": COLOR_LOW, "Landscape": COLOR_SECONDARY, "RKHS": COLOR_HIGH}
    sub = metrics[metrics["dataset"].astype(str).str.lower() == "infectious"].copy()
    pivot = sub.pivot(index="level", columns="model", values="rmse").reindex(index=level_order, columns=model_order)
    if pivot.isna().any().any():
        missing = pivot.isna()
        missing_pairs = [(level_order[i], model_order[j]) for i, j in np.argwhere(missing.to_numpy())]
        raise ValueError(f"missing total collective model metrics rows: {missing_pairs}")

    x = np.arange(len(level_order), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI, facecolor="none")
    for idx, model in enumerate(model_order):
        y = pivot[model].to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, y, width=width, label=model, color=colors[model], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(level_order, rotation=0, ha="center")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(asset_dir / "curve_infectious_model_rmse_2d.png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    sub.to_csv(asset_dir / "text_infectious_model_metrics.csv", index=False)


def _generate_total_collective_results_table(asset_dir: Path, metrics: pd.DataFrame) -> None:
    _clear_asset_dir(asset_dir)
    if metrics.empty:
        return
    sub = metrics[metrics["dataset"].astype(str).str.lower() == "infectious"].copy()
    rows: list[dict[str, str | float]] = []
    level_order = ["Base", "Level 1", "Level 2"]
    model_order = ["TGN", "Landscape", "RKHS"]
    for level_label in level_order:
        lev = sub[sub["level"] == level_label]
        if lev.empty:
            raise ValueError(f"total collective metrics has no rows for level={level_label}")
        for model in model_order:
            row = lev[lev["model"] == model]
            if row.empty:
                raise ValueError(f"missing row for level={level_label} model={model}")
            if "brier" not in row.columns:
                raise ValueError("collective metrics is missing required column 'brier'")
            if "ece" not in row.columns:
                raise ValueError("collective metrics is missing required column 'ece'")
            if "eval_count" not in row.columns:
                raise ValueError("collective metrics is missing required column 'eval_count'")
            rows.append(
                {
                    "Level": level_label,
                    "Model": model,
                    "RMSE": float(row.iloc[0]["rmse"]),
                    "Brier": float(row.iloc[0]["brier"]),
                    "ECE": float(row.iloc[0]["ece"]),
                    "EvalCount": float(row.iloc[0]["eval_count"]),
                }
            )
    table_df = pd.DataFrame(rows)
    table_df.to_csv(asset_dir / "text_infectious_results_table.csv", index=False)
    _render_total_collective_results_table_strict(table_df, asset_dir / "table_infectious_results_main.png")


def _render_total_collective_results_table_strict(table_df: pd.DataFrame, output_path: Path) -> None:
    required = {"Level", "Model", "RMSE", "Brier", "ECE"}
    missing = required.difference(set(table_df.columns))
    if missing:
        raise ValueError(f"missing total collective table columns: {sorted(missing)}")

    level_order = ["Base", "Level 1", "Level 2"]
    model_order = ["TGN", "Landscape", "RKHS"]
    table_df["Level"] = pd.Categorical(table_df["Level"], categories=level_order, ordered=True)
    table_df["Model"] = pd.Categorical(table_df["Model"], categories=model_order, ordered=True)
    table_df = table_df.sort_values(["Level", "Model"]).reset_index(drop=True)
    if len(table_df) != 9:
        raise ValueError(f"expected exactly 9 rows (3 levels x 3 models), found {len(table_df)}")

    col_widths = [1.2, 1.65, 1.45, 1.45, 1.45]
    row_h = 0.60
    header_h = 0.70
    n_rows = len(table_df)
    fig_w = sum(col_widths)
    fig_h = header_h + n_rows * row_h + 0.30
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIGURE_DPI, facecolor="none")
    ax.set_xlim(0.0, fig_w)
    ax.set_ylim(0.0, fig_h)
    ax.axis("off")

    y_top = fig_h - 0.15
    headers = ["Level", "Model", "RMSE", "Brier", "ECE"]
    x0 = 0.0
    for w, label in zip(col_widths, headers):
        rect = plt.Rectangle((x0, y_top - header_h), w, header_h, facecolor=(0.85, 0.85, 0.85, 0.98), edgecolor="#222222", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x0 + w * 0.5, y_top - header_h * 0.5, label, ha="center", va="center", fontsize=11, fontweight="bold")
        x0 += w

    w_level = col_widths[0]
    w_rest = col_widths[1:]
    current_y = y_top - header_h
    row_idx = 0
    for _level_name in level_order:
        block_top = current_y
        block_h = 3 * row_h
        rect_lv = plt.Rectangle((0.0, block_top - block_h), w_level, block_h, facecolor=(0.92, 0.92, 0.96, 0.95), edgecolor="#222222", linewidth=1.0)
        ax.add_patch(rect_lv)
        ax.text(w_level * 0.5, block_top - 1.5 * row_h, _level_name, ha="center", va="center", fontsize=10, fontweight="bold")

        slice_df = table_df.iloc[row_idx : row_idx + 3]
        best_rmse = float(slice_df["RMSE"].min())
        row_idx += 3

        row_y = block_top
        for _, row in slice_df.iterrows():
            entries = [str(row["Model"]), f"{float(row['RMSE']):.3f}", f"{float(row['Brier']):.3f}", f"{float(row['ECE']):.3f}"]
            x = w_level
            is_best = abs(float(row["RMSE"]) - best_rmse) < 1e-12
            for idx, txt in enumerate(entries):
                w = w_rest[idx]
                fill = (0.93, 0.97, 0.93, 0.95) if is_best else (1.0, 1.0, 1.0, 0.90)
                rect = plt.Rectangle((x, row_y - row_h), w, row_h, facecolor=fill, edgecolor="#222222", linewidth=1.0)
                ax.add_patch(rect)
                fw = "bold" if is_best and idx in (0, 1) else "normal"
                ax.text(x + w * 0.5, row_y - row_h * 0.5, txt, ha="center", va="center", fontsize=10, fontweight=fw)
                x += w
            row_y -= row_h

        current_y -= block_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project figure assets from selected results.")
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=PILOT_INFECTIOUS_TABLE_DEFAULT,
        help="Training table CSV (default: temp/pilot_infectious/table.csv).",
    )
    parser.add_argument(
        "--features-npz",
        type=Path,
        default=PILOT_INFECTIOUS_FEATURES_DEFAULT,
        help="Features NPZ aligned with the table (default: temp/pilot_infectious/features.npz).",
    )
    parser.add_argument(
        "--preprocessed-cache-dir",
        type=Path,
        default=INFECTIOUS_PREPROCESSED_DIR,
        help="Directory with windows.npz for Infectious (multi-day merge; default data/preprocessed/Infectious).",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "output",
        help="Directory containing model prediction CSV outputs.",
    )
    parser.add_argument(
        "--figure-output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
        help="Root directory for generated figure asset folders.",
    )
    parser.add_argument(
        "--collective-metrics-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "output" / "collective_metrics.csv",
    )
    parser.add_argument(
        "--collective-timeseries-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "output" / "collective_timeseries.csv",
    )
    parser.add_argument("--dataset-key", type=str, default=PRIMARY_SCHOOL_DATASET_KEY)
    parser.add_argument("--min-events-per-window", type=int, default=20)
    return parser.parse_args()


def _create_under_curve_surface(x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray) -> pv.PolyData:
    points = []
    faces = []
    for idx in range(len(x_values) - 1):
        p0 = [x_values[idx], y_values[idx], 0.0]
        p1 = [x_values[idx + 1], y_values[idx + 1], 0.0]
        p2 = [x_values[idx + 1], y_values[idx + 1], z_values[idx + 1]]
        p3 = [x_values[idx], y_values[idx], z_values[idx]]
        base = len(points)
        points.extend([p0, p1, p2, p3])
        faces.extend([4, base, base + 1, base + 2, base + 3])
    poly = pv.PolyData(np.array(points), np.array(faces))
    return poly.triangulate()


def _create_ci_ribbon_surface(x_values: np.ndarray, y_center: float, z_low: np.ndarray, z_high: np.ndarray) -> pv.PolyData:
    points = []
    faces = []
    for idx in range(len(x_values) - 1):
        p0 = [x_values[idx], y_center - 0.06, z_low[idx]]
        p1 = [x_values[idx + 1], y_center - 0.06, z_low[idx + 1]]
        p2 = [x_values[idx + 1], y_center + 0.06, z_high[idx + 1]]
        p3 = [x_values[idx], y_center + 0.06, z_high[idx]]
        base = len(points)
        points.extend([p0, p1, p2, p3])
        faces.extend([4, base, base + 1, base + 2, base + 3])
    poly = pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64))
    return poly.triangulate()


def _add_floor_grid(plotter: pv.Plotter, x_min: float, x_max: float, y_min: float, y_max: float, z: float, steps: int) -> None:
    for x in np.linspace(x_min, x_max, steps):
        line = pv.Line((float(x), y_min, z), (float(x), y_max, z))
        plotter.add_mesh(line.tube(radius=0.004), color="#999999", opacity=0.55)
    for y in np.linspace(y_min, y_max, steps):
        line = pv.Line((x_min, float(y), z), (x_max, float(y), z))
        plotter.add_mesh(line.tube(radius=0.004), color="#999999", opacity=0.55)
    plane = pv.Plane(center=((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, z), i_size=max(x_max - x_min, 1e-6), j_size=max(y_max - y_min, 1e-6))
    plotter.add_mesh(plane, color="white", opacity=0.20)


def _render_scatter_persistence_style(x: np.ndarray, y: np.ndarray, z: np.ndarray, output_path: Path) -> None:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))

    x_n = (x - x_min) / (x_max - x_min + 1e-12)
    y_n = (y - y_min) / (y_max - y_min + 1e-12)
    z_n = (z - z_min) / (z_max - z_min + 1e-12)
    heights = 0.08 + 0.92 * z_n

    fig = plt.figure(figsize=(10, 10), dpi=FIGURE_DPI, facecolor="none")
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    z_grid, z_plane, z_pts = 1, 2, 4
    for val in np.linspace(0.0, 1.0, 6):
        ax.plot([val, val], [0.0, 1.0], [0.0, 0.0], color="#999999", linewidth=2, alpha=0.6, zorder=z_grid)
        ax.plot([0.0, 1.0], [val, val], [0.0, 0.0], color="#999999", linewidth=2, alpha=0.6, zorder=z_grid)
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 2), np.linspace(0.0, 1.0, 2))
    zz = np.zeros_like(xx)
    surf = ax.plot_surface(xx, yy, zz, color="white", alpha=0.5, shade=False)
    surf.set_zorder(z_plane)

    for i in range(len(x_n)):
        col = _blend_color(COLOR_LOW, COLOR_HIGH, float(y_n[i]))
        ax.plot([x_n[i], x_n[i]], [y_n[i], y_n[i]], [0.0, heights[i]], color=col, linewidth=3.2, alpha=0.88, zorder=z_pts)
        ax.scatter([x_n[i]], [y_n[i]], [heights[i]], c=[col], s=90, alpha=0.92, edgecolors="black", linewidths=0.7, zorder=z_pts)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.view_init(elev=25, azim=-60)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.2, facecolor="none", edgecolor="none", transparent=True)
    plt.close(fig)


def _create_text_sticker(
    text: str,
    output_path: Path,
    color: str,
    fig_width: float = 3.4,
    fig_height: float = 0.9,
    fontsize: int = 12,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=FIGURE_DPI, facecolor="none")
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color, linewidth=1.0, alpha=0.9),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True, pad_inches=0.1)
    plt.close(fig)


def _dataset_label_from_key(dataset_key: str) -> str:
    token = dataset_key.split("\\")[0].strip().lower()
    mapping = {"infectious": "Infectious"}
    return mapping.get(token, token)


def _create_latex_sticker(
    latex_text: str,
    output_path: Path,
    color: str,
    fontsize: int,
    fig_width: float = 2.8,
    fig_height: float = 0.9,
) -> None:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=FIGURE_DPI, facecolor="none")
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        latex_text,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=(1.0, 1.0, 1.0, 0.55), edgecolor=color, linewidth=0.8),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True, pad_inches=0.04)
    plt.close(fig)


def _clear_asset_dir(asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for path in asset_dir.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir() and path.name in ("loss-functions", "other_improvements", "arrow_shapes"):
            for sub in path.iterdir():
                if sub.is_file():
                    sub.unlink(missing_ok=True)


def _is_infectious_sciencegallery_dataset_key(dataset_key: str) -> bool:
    return dataset_key.startswith("infectious\\sciencegallery_infectious_contacts\\")


def _assert_table_features_row_alignment(table: pd.DataFrame, features_npz: Path) -> None:
    if not features_npz.is_file():
        raise RuntimeError(f"missing features npz: {features_npz}")
    data = np.load(features_npz)
    if "d_t" not in data:
        raise RuntimeError(f"features npz missing array d_t: {features_npz}")
    n_feat = int(data["d_t"].shape[0])
    n_tbl = len(table)
    if n_tbl != n_feat:
        raise RuntimeError(
            f"table has {n_tbl} rows but features d_t has {n_feat} for {features_npz}; "
            "use the table and npz from the same build_infectious_pilot_table or level run."
        )


def _load_windows_aligned_with_table(
    table: pd.DataFrame,
    dataset_key: str,
    min_events: int,
    preprocessed_cache_dir: Path | None,
) -> list[pd.DataFrame]:
    if _is_infectious_sciencegallery_dataset_key(dataset_key):
        cache_root = preprocessed_cache_dir if preprocessed_cache_dir is not None else INFECTIOUS_PREPROCESSED_DIR
        npz_path = Path(cache_root) / "windows.npz"
        if not npz_path.is_file():
            raise RuntimeError(
                f"Infectious figure geometry requires cached windows.npz at {npz_path} "
                "(same multi-day merge as run_collective_benchmark / build_infectious_pilot_table). "
                "Run: python scripts/cache_dataset_windows.py --dataset Infectious"
            )
        windows = load_cached_windows(Path(cache_root))
        if "t" not in table.columns:
            raise RuntimeError("table must include column t (window index) matching windows.npz")
        t_series = pd.to_numeric(table["t"], errors="coerce")
        if bool(t_series.isna().any()):
            raise RuntimeError("table column t has non-numeric entries")
        t_max = int(t_series.max())
        if t_max >= len(windows):
            raise RuntimeError(
                f"table column t reaches {t_max} but cached windows count is {len(windows)}; "
                f"table does not match windows under {cache_root}"
            )
        return windows
    return _reconstruct_windows_from_table(table, dataset_key=dataset_key, min_events=min_events)


def _table_row_to_window_index(table: pd.DataFrame, row_index: int) -> int:
    idx = int(np.clip(row_index, 0, len(table) - 1))
    val = float(pd.to_numeric(table.iloc[idx]["t"], errors="raise"))
    return int(val)


def _reconstruct_windows_from_table(table: pd.DataFrame, dataset_key: str, min_events: int) -> list[pd.DataFrame]:
    datasets = load_all_datasets(PROJECT_ROOT / "data")
    if dataset_key not in datasets:
        return []
    temporal_result = extract_temporal_events_for_dataset(dataset_key, datasets[dataset_key])
    if temporal_result is None:
        return []
    events = temporal_result.events.copy()
    window_seconds = float(np.median(table["window_end"].to_numpy() - table["window_start"].to_numpy()))
    starts = np.sort(table["window_start"].to_numpy())
    if len(starts) > 1:
        stride_seconds = float(np.median(np.diff(starts)))
    else:
        stride_seconds = max(window_seconds * 0.5, 1.0)
    t_min = float(events["t_start"].min())
    t_max = float(events["t_start"].max())
    windows = []
    cursor = t_min
    while cursor + window_seconds <= t_max + 1e-9:
        mask = (events["t_start"] >= cursor) & (events["t_start"] < cursor + window_seconds)
        window_events = events.loc[mask, ["source", "target", "t_start", "duration_seconds"]].copy()
        if len(window_events) >= min_events:
            windows.append(window_events)
        cursor += stride_seconds
    return windows


def _pick_pipeline_network_window_edges(
    table: pd.DataFrame,
    dataset_key: str,
    min_events: int,
    preprocessed_cache_dir: Path | None,
) -> pd.DataFrame:
    windows = _load_windows_aligned_with_table(table, dataset_key, min_events, preprocessed_cache_dir)
    if not windows:
        raise RuntimeError(
            f"no temporal windows for dataset {dataset_key!r}; cannot render pipeline mesh_network_3d.png"
        )
    mid = len(table) // 2
    t_idx = _table_row_to_window_index(table, mid)
    return aggregate_window_edges(windows[t_idx])


def _render_network_asset(edges: pd.DataFrame, output_path: Path) -> None:
    if pv is None:
        raise RuntimeError("PyVista is required for network rendering.")
    if edges.empty:
        return
    nodes = sorted(set(edges["source"]).union(set(edges["target"])))
    angles = np.linspace(0.0, 2.0 * np.pi, len(nodes), endpoint=False)
    coords = {node: np.array([np.cos(ang), np.sin(ang), 0.0]) for node, ang in zip(nodes, angles)}
    weights = edges["duration_seconds"].to_numpy(dtype=float)
    w_min = float(np.min(weights))
    w_max = float(np.max(weights))
    if w_max <= w_min:
        level = np.full(len(edges), 0.5, dtype=float)
    else:
        level = (weights - w_min) / (w_max - w_min)

    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    r_lo, r_hi = 0.0024, 0.026
    op_lo, op_hi = 0.18, 0.92
    gamma = 1.35
    for lev, (_, row) in zip(level, edges.iterrows()):
        p0 = coords[str(row["source"])]
        p1 = coords[str(row["target"])]
        t = float(np.clip(lev, 0.0, 1.0))
        tube_r = r_lo + (r_hi - r_lo) * (t**gamma)
        line = pv.Line(tuple(p0), tuple(p1)).tube(radius=tube_r)
        edge_color = _blend_color(COLOR_LOW, COLOR_HIGH, t)
        op = op_lo + (op_hi - op_lo) * t
        plotter.add_mesh(line, color=edge_color, opacity=float(op))
    for node in nodes:
        sphere = pv.Sphere(radius=0.022, center=tuple(coords[node]), theta_resolution=14, phi_resolution=14)
        plotter.add_mesh(sphere, color="#A8A8A8", opacity=0.88)
    _set_camera_overhead(plotter, pad=2.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _add_floor_plane(plotter: pv.Plotter, x_min: float, x_max: float, y_min: float, y_max: float, z: float) -> None:
    plane = pv.Plane(
        center=((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, z),
        i_size=max(x_max - x_min, 1e-6),
        j_size=max(y_max - y_min, 1e-6),
    )
    plotter.add_mesh(plane, color="white", opacity=0.20)


def _vpd_vector_to_diagram(vector: np.ndarray, grid_size: int) -> np.ndarray:
    half = grid_size * grid_size
    h0 = np.asarray(vector[:half], dtype=float).reshape(grid_size, grid_size)
    h1 = np.asarray(vector[half:], dtype=float).reshape(grid_size, grid_size)
    rows: list[list[float]] = []
    for dim, matrix in ((0, h0), (1, h1)):
        for i in range(grid_size):
            for j in range(grid_size):
                multiplicity = int(round(matrix[i, j]))
                if multiplicity <= 0:
                    continue
                birth = (i + 0.5) / grid_size
                death = (j + 0.5) / grid_size
                if death <= birth:
                    continue
                rows.append([float(dim), float(birth), float(death), float(multiplicity)])
    if not rows:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def _blend_color(hex_a: str, hex_b: str, t: float) -> tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    a = tuple(int(hex_a.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    b = tuple(int(hex_b.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))


def _set_camera_head_on(plotter: pv.Plotter, pad: float, tilt: float) -> None:
    x_min, x_max, y_min, y_max, z_min, z_max = plotter.bounds
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    dz = max(z_max - z_min, 1e-6)
    dist = pad * max(dx, dy, dz)
    plotter.camera.position = (cx, cy - dist, cz + tilt * dist)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up = (0, 0, 1)


def _set_camera_overhead(plotter: pv.Plotter, pad: float) -> None:
    x_min, x_max, y_min, y_max, z_min, z_max = plotter.bounds
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)
    d = pad * max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6)
    plotter.camera.position = (cx, cy, cz + d)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up = (0, 1, 0)


def _render_single_box_asset(output_path: Path, color: str) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    box = pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=0.6)
    plotter.add_mesh(box, color=color, opacity=0.84)
    _set_camera_head_on(plotter, pad=2.6, tilt=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _render_gradient_straight_arrow(output_path: Path, color_start: str, color_end: str) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    n = 80
    points = [(-0.8 + 1.6 * t, 0.0, 0.0) for t in np.linspace(0.0, 1.0, n)]
    for idx in range(n - 1):
        p0, p1 = points[idx], points[idx + 1]
        color = _blend_color(color_start, color_end, idx / max(n - 2, 1))
        plotter.add_mesh(pv.Line(p0, p1).tube(radius=0.018), color=color, opacity=0.9)
    d = np.array(points[-1]) - np.array(points[-2])
    cone = pv.Cone(center=np.array(points[-1]) + 0.05 * d, direction=d, height=0.18, radius=0.06, resolution=18)
    plotter.add_mesh(cone, color=color_end, opacity=0.9)
    _set_camera_head_on(plotter, pad=3.0, tilt=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _render_gradient_planar_sem_arc_arrow(output_path: Path, color_start: str, color_end: str) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    radius = 0.55
    n = 88
    angles = np.linspace(0.0, np.pi, n)
    points = []
    for ang in angles:
        x = float(radius * np.cos(np.pi - ang))
        y = float(radius * np.sin(np.pi - ang))
        points.append((x, y, 0.0))
    for idx in range(n - 1):
        p0, p1 = points[idx], points[idx + 1]
        color = _blend_color(color_start, color_end, idx / max(n - 2, 1))
        plotter.add_mesh(pv.Line(p0, p1).tube(radius=0.017), color=color, opacity=0.9)
    d = np.array(points[-1]) - np.array(points[-2])
    cone = pv.Cone(center=np.array(points[-1]) + 0.05 * d, direction=d, height=0.16, radius=0.055, resolution=18)
    plotter.add_mesh(cone, color=color_end, opacity=0.9)
    _set_camera_head_on(plotter, pad=3.1, tilt=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _render_gradient_polyline_arrow(
    output_path: Path,
    color_start: str,
    color_end: str,
    waypoints: list[tuple[float, float, float]],
    samples_per_edge: int = 45,
    tube_radius: float = 0.017,
) -> None:
    if len(waypoints) < 2:
        raise ValueError("polyline arrow needs at least two waypoints")
    pts: list[tuple[float, float, float]] = []
    for i in range(len(waypoints) - 1):
        a = np.asarray(waypoints[i], dtype=float)
        b = np.asarray(waypoints[i + 1], dtype=float)
        ts = np.linspace(0.0, 1.0, samples_per_edge, endpoint=True)
        if i > 0:
            ts = ts[1:]
        for t in ts:
            pts.append(tuple(float(x) for x in (a + (b - a) * t)))
    n = len(pts)
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    for idx in range(n - 1):
        p0, p1 = pts[idx], pts[idx + 1]
        color = _blend_color(color_start, color_end, idx / max(n - 2, 1))
        plotter.add_mesh(pv.Line(p0, p1).tube(radius=tube_radius), color=color, opacity=0.9)
    d = np.array(pts[-1]) - np.array(pts[-2])
    cone = pv.Cone(center=np.array(pts[-1]) + 0.05 * d, direction=d, height=0.16, radius=0.055, resolution=18)
    plotter.add_mesh(cone, color=color_end, opacity=0.9)
    _set_camera_head_on(plotter, pad=3.2, tilt=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _write_pipeline_arrow_shape_assets(arrow_dir: Path) -> None:
    if pv is None:
        return
    arrow_dir.mkdir(parents=True, exist_ok=True)
    for path in arrow_dir.glob("*.png"):
        path.unlink(missing_ok=True)
    b, r = COLOR_LOW, COLOR_HIGH
    _render_gradient_straight_arrow(arrow_dir / "mesh_straight_blue_red.png", b, r)
    _render_gradient_straight_arrow(arrow_dir / "mesh_straight_red_blue.png", r, b)
    _render_gradient_planar_sem_arc_arrow(arrow_dir / "mesh_arc_planar_blue_red.png", b, r)
    _render_gradient_planar_sem_arc_arrow(arrow_dir / "mesh_arc_planar_red_blue.png", r, b)
    # Planar L in xy (z=0): equal leg length, horizontal (+x) then vertical (+y), head at top.
    L = 0.72
    h = 0.36
    bent_L = [
        (-h, -h, 0.0),
        (h, -h, 0.0),
        (h, h, 0.0),
    ]
    _render_gradient_polyline_arrow(
        arrow_dir / "mesh_bent_90_blue_red.png",
        b,
        r,
        bent_L,
    )
    _render_gradient_polyline_arrow(
        arrow_dir / "mesh_bent_90_red_blue.png",
        r,
        b,
        bent_L,
    )
    for name, curv, h in (
        ("shallow", 0.14, 0.10),
        ("medium", 0.28, 0.14),
        ("default", 0.35, 0.18),
        ("deep", 0.42, 0.28),
        ("steep", 0.55, 0.32),
    ):
        _render_gradient_arc_arrow(arrow_dir / f"mesh_arc_{name}_blue_red.png", b, r, curvature=curv, height=h)
        _render_gradient_arc_arrow(arrow_dir / f"mesh_arc_{name}_red_blue.png", r, b, curvature=curv, height=h)
    _render_gradient_s_curve_arrow(arrow_dir / "mesh_scurve_blue_red.png", b, r)
    _render_gradient_s_curve_arrow(arrow_dir / "mesh_scurve_red_blue.png", r, b)
    _render_gradient_spiral_arrow(arrow_dir / "mesh_spiral_blue_red.png", b, r)
    _render_gradient_spiral_arrow(arrow_dir / "mesh_spiral_red_blue.png", r, b)


def _render_gradient_arc_arrow(
    output_path: Path,
    color_start: str,
    color_end: str,
    curvature: float = 0.35,
    height: float = 0.18,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    n = 80
    t_values = np.linspace(0.0, 1.0, n)
    points = []
    for t in t_values:
        x = -0.8 + 1.6 * t
        y = float(curvature) * np.sin(np.pi * t)
        z = float(height) * (1.0 - np.cos(np.pi * t))
        points.append((x, y, z))
    for idx in range(n - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        color = _blend_color(color_start, color_end, idx / max(n - 2, 1))
        seg = pv.Line(p0, p1).tube(radius=0.018)
        plotter.add_mesh(seg, color=color, opacity=0.9)
    d = np.array(points[-1]) - np.array(points[-2])
    cone = pv.Cone(center=np.array(points[-1]) + 0.05 * d, direction=d, height=0.18, radius=0.06, resolution=18)
    plotter.add_mesh(cone, color=color_end, opacity=0.9)
    _set_camera_head_on(plotter, pad=3.0, tilt=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _render_gradient_s_curve_arrow(output_path: Path, color_start: str, color_end: str) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    n = 90
    t_values = np.linspace(0.0, 1.0, n)
    points = []
    for t in t_values:
        x = -0.9 + 1.8 * t
        y = 0.30 * np.sin(2.0 * np.pi * (t - 0.5))
        z = 0.12 + 0.12 * np.sin(np.pi * t)
        points.append((x, y, z))
    for idx in range(n - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        color = _blend_color(color_start, color_end, idx / max(n - 2, 1))
        seg = pv.Line(p0, p1).tube(radius=0.016)
        plotter.add_mesh(seg, color=color, opacity=0.9)
    d = np.array(points[-1]) - np.array(points[-2])
    cone = pv.Cone(center=np.array(points[-1]) + 0.05 * d, direction=d, height=0.16, radius=0.055, resolution=20)
    plotter.add_mesh(cone, color=color_end, opacity=0.9)
    _set_camera_head_on(plotter, pad=3.2, tilt=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _render_gradient_spiral_arrow(output_path: Path, color_start: str, color_end: str) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=[3000, 3000])
    plotter.set_background([0, 0, 0, 0])
    n = 120
    t_values = np.linspace(0.0, 1.0, n)
    points = []
    for t in t_values:
        theta = 3.2 * np.pi * t
        r = 0.15 + 0.35 * t
        x = -0.7 + 1.4 * t
        y = r * np.cos(theta)
        z = r * np.sin(theta) * 0.6
        points.append((x, y, z))
    for idx in range(n - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        color = _blend_color(color_start, color_end, idx / max(n - 2, 1))
        seg = pv.Line(p0, p1).tube(radius=0.013)
        plotter.add_mesh(seg, color=color, opacity=0.9)
    d = np.array(points[-1]) - np.array(points[-2])
    cone = pv.Cone(center=np.array(points[-1]) + 0.05 * d, direction=d, height=0.14, radius=0.05, resolution=18)
    plotter.add_mesh(cone, color=color_end, opacity=0.9)
    _set_camera_head_on(plotter, pad=3.3, tilt=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_plotter_image(plotter, output_path)
    plotter.close()


def _save_plotter_image(plotter: pv.Plotter, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = plotter.screenshot(transparent_background=True, return_img=True)
    try:
        from PIL import Image

        Image.fromarray(image).save(output_path, dpi=(FIGURE_DPI, FIGURE_DPI))
    except Exception:
        plt.imsave(output_path, image)


if __name__ == "__main__":
    main()
