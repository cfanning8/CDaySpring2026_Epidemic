# C-DAY Epidemiology Micro-Project

## Current status
- Data loading and sanity checks are in place for active datasets in `data/`.
- Active mature benchmark set: `Infectious` only.
- Paper 1 topology tools are now in `src/topology/`.
- A primary-school pilot builder produces `D_t`, `g_t`, RKHS features, and epidemic labels.
- Figure generation is being simplified to 2D-first interpretable assets with strict sticker-only labeling.
- Tiny/immature datasets have been removed from the benchmark pipeline (`LH10`, `SCC2034`, `Malawi`).

## Frozen research claim
- Main claim: virtual topological drift predicts epidemic instability.
- Supporting claim: topology-aware models improve prediction over graph-only temporal baselines.

## Frozen model set
- Model 1: TGN only.
- Model 2: TGN + persistence landscapes + constraint loss.
- Model 3: TGN + RKHS + constraint loss.

### Core equations in use
- TGN event messaging and memory update:
  - `m_i(t) = msg_s(s_i(t^-), s_j(t^-), Delta t, e_ij(t))`
  - `m_j(t) = msg_d(s_j(t^-), s_i(t^-), Delta t, e_ij(t))`
  - `mbar_i(t) = agg({m_i(t_k)}_{t_k <= t})`
  - `s_i(t) = mem(mbar_i(t), s_i(t^-))`
  - `z_i(t) = emb(i,t) = sum_{j in N_i^t([0,t])} h(s_i(t), s_j(t), e_ij(t), v_i(t), v_j(t))`
- Graph-topology objective:
  - `g_t = D_(t+Delta) - D_t`
  - `L = (Yhat_t - Y_t)^2 + lambda_D ||z_t - phi(D_t)||_2^2 + lambda_g ||z_t - phi(g_t)||_2^2`
- Landscape branch:
  - `z_D = psi_L(D_t)`
  - `psi_L` is a fixed vectorization based on persistence landscapes.
- RKHS branch:
  - `k_nu_t(alpha,beta) = <Phi_nu_t(alpha), Phi_nu_t(beta)>_(L2(nu_t))`
  - `H_nu_t = span{Phi_nu_t(alpha): alpha in K(X,A)}`

## Frozen target and simulator policy
- Single training target: `Y_t = P(attack_rate >= tau)`.
- Global threshold: `tau = 0.20`.
- Simulator for first official run: SIR only.
- For short windows, contact sequences are cycled until simulation horizon.
- Infectious handling: each day is independent, no pooled pseudo-sequence.

## Frozen topology policy
- Common-basis contract is mandatory before drift subtraction:
  - for each dataset and homology dimension, all vectorized `D_t` share one fixed birth/death grid.
  - `g_t = D_(t+delta) - D_t` is computed only in that shared vector space.
- Topology objects are separated by type:
  - ordinary diagrams (raw point multisets) for persistence landscapes.
  - common-basis vectors `D_t`, signed drift `g_t`, and RKHS features `rkhs_g_t` for the RKHS branch.
- Constraint policy:
  - Topology enters as both:
    - an auxiliary alignment loss term, and
    - a fused predictor input through concatenation with the TGN window embedding.
  - Landscape branch uses persistence-landscape vectorization on raw diagram points:
    - `z_D = psi_L(D_t)` where `psi_L` concatenates top-k landscape layers over a fixed grid.
    - fused prediction head uses `[h_t || z_D]`.
    - applies `L = L_pred + lambda_D * L_D`.
  - RKHS branch:
    - maps `h_t` to RKHS space for alignment (`z_g`),
    - fused prediction head uses `[h_t || z_g]`,
    - applies `L = L_pred + lambda_g * L_g`.
  - RKHS branch uses virtual drift only and applies `L = L_pred + lambda_g * L_g`.
  - RKHS alignment term uses the explicit virtual-drift feature map and signed drift semantics.
  - `h_t` is the window-level TGN latent:
    - TGN memory is updated by all events in window `t`,
    - active-node embeddings are pooled by mean to form one graph/window representation `h_t`.
  - RKHS dimension matching:
    - explicit random-feature map `Phi(g_t)` is cached as `rkhs_g_t`,
    - a learned projection head maps `h_t` into RKHS feature dimension before alignment loss.
  - Filtration implementation detail:
    - edge filtration is computed from cumulative duration with a monotone superlevel-equivalent transform
      `f(e)=1-(w_e / max_w)` so larger duration enters earlier in the filtration.

## Window defaults
- `infectious`: per-day independent windows.

## Label calibration targets
- median(`Y_t`) in `[0.15, 0.35]`
- std(`Y_t`) `>= 0.15`
- at least 20 percent of windows with `Y_t < 0.10`
- at least 20 percent of windows with `Y_t > 0.50`

## Frozen SIR calibration policy (one-and-done)
- Per-dataset SIR calibration is frozen in `results/output/sir_calibration_by_dataset.csv`.
- Full calibration search traces are stored in `results/output/sir_calibration_search.csv`.
- Post-calibration label spread summary is stored in `results/output/sir_post_calibration_summary.csv`.
- Calibration script: `scripts/calibrate_sir_all_datasets.py`.
- Recompute only if explicitly changing epidemiological assumptions (`tau`, horizon, window policy, or simulator model).
- `scripts/run_collective_benchmark.py` reads calibrated `beta_per_second` and `gamma_per_second` from the calibration CSV by default.
- Cached calibrated labels are persisted in `data/preprocessed/<dataset>/sir_labels.csv`.

### Frozen calibration run command
- `.\.venv\Scripts\python -u scripts\calibrate_sir_all_datasets.py --sample-windows 6 --calibration-num-simulations 10 --search-rounds 1 --beta-scales 0.7,1.0 --gamma-scales 1.0,1.4 --apply-full-cache --full-num-simulations 120 --full-workers 8 --full-flush-every 2`

### Frozen beta/gamma by dataset
- `Infectious`: beta=`0.006999999999999999`, gamma=`0.0000648144`

## Simulation counts
- smoke tests: small counts are allowed
- official pilot: at least 200 simulations
- poster-grade run: 500 simulations

## Project structure
- `src/dataloaders.py`: dataset loading
- `src/edge_preparation.py`: canonical edges and temporal event extraction
- `src/episim.py`: cycled-window SIR labels
- `src/topology/`: Paper 1 topology modules
- `scripts/smoke_test_loaders.py`: loader smoke test
- `scripts/smoke_test_paper1_tools.py`: topology smoke test
- `scripts/sanity_check_inputs.py`: structural sanity checks
- `scripts/pipeline_readiness_audit.py`: temporal readiness audit
- `scripts/build_infectious_pilot_table.py`: pilot feature/label table
- `scripts/cache_persistence_features.py`: two-pass persistence cache with fixed-range vectorization and raw diagram cache
- `scripts/train_tgn_baseline.py`: Model 1 training script
- `scripts/train_tgn_landscape_constraint.py`: Model 2 training script
- `scripts/train_tgn_rkhs_constraint.py`: Model 3 training script
- `scripts/tune_topology_loss_terms.py`: gentle offline tuning for `lambda_d` and `lambda_g`
- `scripts/figures/generate_project_assets.py`: canonical figure asset generator
- `scripts/figures/write_pipeline_stickers.py`: pipeline overview stickers + `text_pipeline_steps.csv` only (no PyVista; keeps existing mesh PNGs in that folder)
- `scripts/figures/write_arrow_shapes.py`: gradient arrow meshes (straight, planar arc, bent, raised arcs, S-curve, spiral), blue-red and red-blue each; output `results/figures/pipeline_overview/assets/arrow_shapes/`
- `scripts/figures/write_temporal_network_3d.py`: Infectious window graphs (representative + low/transition/high risk); output `results/figures/temporal_network_3d/assets/`
- `scripts/figures/persistence_diagram_3d_engine.py`: persistence and virtual persistence rendering backend
- `results/figures/`: figure outputs as per-figure asset folders
- `results/output/`: structured outputs such as CSV logs and metrics
- `results/weights/`: model checkpoint weights

## Current critical caution
- Label degeneracy must be avoided before scaling. Always check `Y_t` distribution against calibration targets before model training.
- Temporal leakage caution:
  - training uses chronological splits (train/val/test by time order), not random window shuffling.
  - split logic enforces minimum validation/test support on active datasets.
- Pilot table stores transition diagnostics for Claim B:
  - `y_next_large_outbreak_prob`
  - `delta_y_large_outbreak_prob`
  - `g_l2_norm`

## Next actions
- Run model training with frozen calibrated SIR labels.
- Regenerate metrics/figures from calibrated Infectious caches.
- Refresh pipeline equation stickers: `python scripts/figures/write_pipeline_stickers.py` (requires `temp/pilot_infectious/table.csv` and `features.npz` by default). Training loss: `results/figures/pipeline_overview/assets/loss-functions/`. VPD/drift/jet, RKHS/RFF, prediction heads, model labels, targets: `results/figures/pipeline_overview/assets/other_improvements/`. Full asset rebuild (defaults: `temp/pilot_infectious/table.csv`, `features.npz`, `data/preprocessed/Infectious/windows.npz` for Infectious mesh): `python scripts/figures/generate_project_assets.py` (PyVista). Pipeline graph `mesh_network_3d.png` uses the cached multi-day Infectious windows, not a single raw contact file.
- Refresh arrow mesh library: `python scripts/figures/write_arrow_shapes.py` (PyVista).
- Temporal network PNGs only: `python scripts/figures/write_temporal_network_3d.py`
- Rebuild Infectious from scratch with:
  - `python -u scripts/run_collective_benchmark.py --dataset Infectious --reset-cache --force-rebuild --force-train`
- Early stopping is enabled in all three training scripts with patience-based stopping.
- Gentle loss-term tuning is done outside the main pipeline:
  - `.\.venv\Scripts\python -u scripts\tune_topology_loss_terms.py --dataset <DATASET>`

## Evaluation protocol
- Primary error metric: RMSE on chronological test split.
- Calibration metrics: Brier score and ECE.
- Metrics are computed per dataset, on chronological test split.
- No fallback scopes are allowed during official evaluation.
- Cache generation and training are executed one dataset at a time by design.

## Figure asset policy
- Complex figures must be produced as assets only.
- For each figure, use `results/figures/<figure_name>/assets/`.
- Do not synthesize final poster composition in code.
- Text, arrows, and overlays should be separate assets.
- Preferred asset naming:
  - `mesh_*.png` for 3D meshes and geometry layers (only where explicitly needed)
  - `curve_*.png` for 2D curve/scatter layers
  - `overlay_*.png` for transparent compositing layers
  - `text_*.csv` for textual/structured payloads
  - `arrow_*.png` for arrow-specific geometry layers
- `drift_risk_timeseries_3d/assets` keep only `curve_infectious_timeseries_2d.png` plus essential stickers for assembly.
- `rkhs_trajectory_3d` is currently frozen and should not be altered during figure cleanup passes.

## Figure generation safety
- Always specify exact input artifacts when generating figures.
- Use:
  - `--table-csv <path>`
  - `--features-npz <path>`
  - `--model-output-dir <path>`
- This prevents accidental reuse of stale outputs.

## Topological jet hierarchy formalism (full math)
- The methodology is modeled as a jet-theoretic hierarchy over persistence state space.
- Temporal jets and structural jets are separated, then combined into a bi-graded object.

### State space
- Discrete time index:
  - `T = {t_0, t_1, ..., t_N}`
- Persistence state map:
  - `D: T -> H`
- Common-basis requirement:
  - `D_t in H` with linear subtraction defined.
- Valid choices:
  - `H = R^m` (common-basis vector space), or
  - `H = H_k` (RKHS).

### First-order temporal jet
- Discrete differential operator:
  - `Delta_tau D_t = D_{t+tau} - D_t`
- Existing drift object:
  - `g_t = D_{t+Delta} - D_t`
- First jet:
  - `J_t^(1) = (D_t, Delta_tau D_t)`

### Higher-order temporal jets
- Recursive operator:
  - `Delta_tau^0 D_t = D_t`
  - `Delta_tau^k D_t = Delta_tau(Delta_tau^(k-1) D_t)`
- Second order:
  - `Delta_tau^2 D_t = D_{t+2tau} - 2 D_{t+tau} + D_t`
- Third order:
  - `Delta_tau^3 D_t = D_{t+3tau} - 3 D_{t+2tau} + 3 D_{t+tau} - D_t`
- General order:
  - `Delta_tau^k D_t = sum_{j=0}^k (-1)^(k-j) * C(k,j) * D_{t+j tau}`
- Full temporal jet:
  - `J_t^(k) = (D_t, Delta D_t, Delta^2 D_t, ..., Delta^k D_t)`
  - `J_t^(k) in H^(k+1)`

### Structural recursive jet hierarchy
- Recursive persistence interaction hierarchy:
  - `D^(n+1)(X) = D(X^(n), A^(n))`
- Interaction-lift interpretation:
  - Let `D = sum_i m_i I_i` in interval basis.
  - First lift uses pairwise multiplicity interactions:
    - `Phi_2(D) ~= Phi(D) tensor Phi(D)`
  - Higher lifts follow recursive interaction expansion.

### Bi-graded hierarchy
- Combined temporal + structural jet:
  - `J_t^(k,n) = ( D_t^(n), Delta D_t^(n), ..., Delta^k D_t^(n) )`
- Interpretation:
  - `n` indexes within-diagram structural order.
  - `k` indexes across-time dynamical order.

### Jet metric
- Sobolev-style jet metric on order `k`:
  - `d_J^(k)(J_t^(k), J_s^(k)) = ( sum_{r=0}^k w_r * ||Delta^r D_t - Delta^r D_s||^p )^(1/p )`
- RKHS form:
  - `d_J^(k) = ( sum_{r=0}^k w_r * ||Delta^r D_t - Delta^r D_s||_(H_k)^2 )^(1/2)`
- Weights satisfy:
  - `w_r > 0`.

### Statistical interpretation
- Zeroth order:
  - `D_t` is topology state.
- First order:
  - `Delta D_t` is instability direction.
- Second order:
  - `Delta^2 D_t` is transition acceleration (curvature proxy).
- Third order:
  - `Delta^3 D_t` is jerk-like abrupt regime-change signature.

### Learning integration
- Feature fusion with TGN latent:
  - `x_t = [h_t || flatten(J_t^(k,n))]`
- Prediction head:
  - `yhat_t = f_theta(x_t)`
- Jet regularization:
  - `L_jet = sum_{r=1}^k lambda_r * ||Delta^r D_t||^2`
- Total objective:
  - `L = L_pred + L_align + L_jet`

### Complexity
- Temporal jet construction for `T` windows, state dimension `m`, order `k`:
  - `O(T * m * k)`
- This is lower-order relative to sequence model training cost.

### Implemented stage definitions
- Stage 1 (level1):
  - first-order temporal jet features (current drift hierarchy).
  - outputs under:
    - `results/level1/output`
    - `results/level1/weights`
    - `results/figures/level1`
- Stage 2 (level2):
  - second-order temporal jet features using:
    - `g_t^(2) = Delta g_t = g_{t+1} - g_t`
    - equivalent:
      - `g_t^(2) = D_{t+2Delta} - 2 D_{t+Delta} + D_t`
  - outputs under:
    - `results/level2/output`
    - `results/level2/weights`
    - `results/figures/level2`
- Aggregated infectious table + RMSE bar chart (Base, Level 1, Level 2): `results/figures/total`
  - `run_jet_hierarchy_levels.py` runs `scripts/figures/generate_total_level_figures.py` at the end (needs `results/output/collective_metrics.csv` for Base).

### Levelized execution entry point
- End-to-end levelized training + metrics + figures:
  - `python -u scripts/run_jet_hierarchy_levels.py --levels 1,2 --datasets Infectious --epochs 50`
