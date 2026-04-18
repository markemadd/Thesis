"""
Task 2 — Sensitivity Analysis: α, β Grid Search + π Threshold Sweep

Part A — α×β grid (25 combinations), at fixed π=0.60:
    α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
    β ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
    Metric: standard LOOCV accuracy + #stable pathways
    Output: heatmap (output/sensitivity_heatmap_alpha_beta.png)

Part B — π sweep (9 values) at best (α, β) from Part A:
    π ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
    Metric: standard LOOCV accuracy + #stable pathways
    Output: dual-axis line plot (output/sensitivity_lineplot_pi.png)

Compute strategy:
  - Jaccard matrix computed ONCE (most expensive step), cached in memory
  - Standard LOOCV used for α×β grid (not nested) — fast and sufficient for
    comparing relative sensitivity; declared clearly in method notes
  - n_jobs=-1 for all sklearn models
  - Scores recomputed per (α,β) pair (Phase 3 is cheap: ~10s/run)

Usage:
    python -m analysis.sensitivity_analysis
"""

import os
import json
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

from pipeline.config import (
    OUTPUT_DIR, SAMPLE_COLS, PATIENT_COLS, CONTROL_COLS,
    RANDOM_STATE, STABILITY_N_BOOTSTRAP, STABILITY_THRESHOLD,
    ELASTIC_NET_L1_RATIOS, ELASTIC_NET_CV_FOLDS,
    JACCARD_THRESHOLD, SCORE_CORR_WEIGHT, IQR_PERCENTILE_CUTOFF,
)
from pipeline.phase3_pathway_scoring import compute_pathway_scores
from pipeline.phase5_dim_reduction import (
    compute_jaccard_matrix,
    compute_combined_similarity,
    cluster_and_select_representatives,
    stage2_variance_filter,
    build_feature_matrix,
    loocv_evaluate,
)

# Import curated weights from ablation_study to avoid duplication
from analysis.ablation_study import (
    DISGENET_CURATED,
    STRING_CURATED,
    build_protein_weights,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
IN_CANONICAL = os.path.join(OUTPUT_DIR, "canonical_log2.csv")
IN_PATHWAYS  = os.path.join(OUTPUT_DIR, "pathway_gene_sets.json")
OUT_HEATMAP  = os.path.join(OUTPUT_DIR, "sensitivity_heatmap_alpha_beta.png")
OUT_LINEPLOT = os.path.join(OUTPUT_DIR, "sensitivity_lineplot_pi.png")
OUT_CSV_GRID = os.path.join(OUTPUT_DIR, "sensitivity_grid_results.csv")

# ── Thesis colour palette ──────────────────────────────────────────────────────
NAVY  = '#1B2A4A'
CORAL = '#E74C3C'
TEAL  = '#1ABC9C'
GOLD  = '#F39C12'
BLUE  = '#2980B9'
GREY  = '#95A5A6'
WHITE = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': WHITE,
    'axes.facecolor': WHITE,
    'axes.edgecolor': '#DDDDDD',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ── Grid definitions ───────────────────────────────────────────────────────────
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
BETA_GRID  = [0.0, 0.25, 0.5, 0.75, 1.0]
PI_GRID    = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
PI_FIXED   = 0.60   # current pipeline default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FAST SINGLE-CONDITION PIPELINE (no nested LOOCV for grid speed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_single_condition(
    canonical: pd.DataFrame,
    pathway_sets: dict,
    all_genes: list[str],
    alpha: float,
    beta: float,
    pi: float,
    cached_pw_ids: list[str],
    cached_jaccard: np.ndarray,
) -> dict:
    """
    Run Phase3 + Phase5 Stages 1-3 + std LOOCV for one (α,β,π) combination.

    Returns:
        dict with keys: alpha, beta, pi, n_stable, loocv_acc, loocv_auc, loocv_kappa
    """
    # Build weights
    protein_weights = build_protein_weights(
        all_genes, alpha=alpha, beta=beta,
        use_disgenet=(alpha > 0), use_string=(beta > 0),
    )

    # Phase 3: score
    scores_df = compute_pathway_scores(canonical, pathway_sets, protein_weights)
    valid_ids = [pid for pid in cached_pw_ids if pid in scores_df.index]

    # Stage 1: de-redundancy (use cached Jaccard, recompute correlation)
    combined = compute_combined_similarity(
        cached_jaccard, scores_df, valid_ids, use_corr=SCORE_CORR_WEIGHT
    )
    representatives = cluster_and_select_representatives(
        combined, scores_df, valid_ids, pathway_sets, JACCARD_THRESHOLD
    )

    # Stage 2: variance filter
    filtered = stage2_variance_filter(scores_df, representatives)

    # Stage 3: stability selection with custom π threshold
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    X_full, y_full, valid_filtered = build_feature_matrix(scores_df, filtered)
    n_samples, n_features = X_full.shape

    if n_features == 0:
        return {'alpha': alpha, 'beta': beta, 'pi': pi,
                'n_stable': 0, 'loocv_acc': np.nan,
                'loocv_auc': np.nan, 'loocv_kappa': np.nan}

    # Mini stability selection with π threshold
    selection_counts = np.zeros(n_features)
    successful = 0
    rng = np.random.RandomState(RANDOM_STATE)

    # Use fewer bootstraps for the grid (100 vs 200) for speed;
    # sufficient for comparing relative sensitivity
    n_boot = min(100, STABILITY_N_BOOTSTRAP)

    for b in range(n_boot):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        Xb, yb = X_full[boot_idx], y_full[boot_idx]

        if len(np.unique(yb)) < 2:
            continue

        scaler = StandardScaler()
        Xb_s = scaler.fit_transform(Xb)

        try:
            model = LogisticRegressionCV(
                penalty='elasticnet', solver='saga',
                l1_ratios=ELASTIC_NET_L1_RATIOS,
                Cs=10,
                cv=min(ELASTIC_NET_CV_FOLDS, int(np.bincount(yb.astype(int)).min())),
                max_iter=3000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                scoring='roc_auc',
                n_jobs=-1,
            )
            model.fit(Xb_s, yb)
            selected = np.abs(model.coef_.flatten()) > 1e-8
            selection_counts += selected.astype(float)
            successful += 1
        except Exception:
            continue

    if successful == 0:
        return {'alpha': alpha, 'beta': beta, 'pi': pi,
                'n_stable': 0, 'loocv_acc': np.nan,
                'loocv_auc': np.nan, 'loocv_kappa': np.nan}

    probs = selection_counts / successful
    stable_mask = probs >= pi
    stable_ids = [valid_filtered[i] for i, flag in enumerate(stable_mask) if flag]
    n_stable = len(stable_ids)

    if n_stable < 2:
        return {'alpha': alpha, 'beta': beta, 'pi': pi,
                'n_stable': n_stable, 'loocv_acc': np.nan,
                'loocv_auc': np.nan, 'loocv_kappa': np.nan}

    # Standard LOOCV on stable pathways
    X_stable, y_stable, _ = build_feature_matrix(scores_df, stable_ids)
    metrics = loocv_evaluate(X_stable, y_stable, 'logistic')

    return {
        'alpha': alpha, 'beta': beta, 'pi': pi,
        'n_stable': n_stable,
        'loocv_acc': metrics['accuracy'],
        'loocv_auc': metrics['auc'],
        'loocv_kappa': metrics['kappa'],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PART A: α×β HEATMAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_alpha_beta_grid(
    canonical, pathway_sets, all_genes, cached_pw_ids, cached_jaccard
) -> pd.DataFrame:
    """Run 25 (α,β) combinations at fixed π=PI_FIXED."""
    rows = []
    total = len(ALPHA_GRID) * len(BETA_GRID)
    done = 0
    t0 = time.perf_counter()

    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            done += 1
            elapsed = time.perf_counter() - t0
            eta = (elapsed / done) * (total - done) if done > 1 else 0
            print(f"    α={alpha:.2f}, β={beta:.2f}  "
                  f"[{done}/{total}, ETA: {eta:.0f}s]")

            result = run_single_condition(
                canonical, pathway_sets, all_genes,
                alpha, beta, PI_FIXED,
                cached_pw_ids, cached_jaccard,
            )
            rows.append(result)

    return pd.DataFrame(rows)


def plot_heatmap(grid_df: pd.DataFrame, out_path: str) -> tuple[float, float]:
    """
    Heatmap of LOOCV accuracy over α×β grid.
    Returns the (best_alpha, best_beta) for Part B.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(WHITE)

    for ax_idx, (metric, title, fmt) in enumerate([
        ('loocv_acc',  'Standard LOOCV Accuracy', '.3f'),
        ('n_stable',   'Number of Stable Pathways', '.0f'),
    ]):
        ax = axes[ax_idx]
        pivot = grid_df.pivot(index='beta', columns='alpha', values=metric)
        pivot.index = [f'β={v}' for v in pivot.index]
        pivot.columns = [f'α={v}' for v in pivot.columns]

        # Choose colormap
        cmap = 'RdYlGn' if ax_idx == 0 else 'Blues'
        vmin = 0.5 if ax_idx == 0 else 0
        vmax = 1.0 if ax_idx == 0 else pivot.values.max()

        sns.heatmap(
            pivot, ax=ax,
            annot=True, fmt=fmt,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor='#EEE',
            cbar_kws={'shrink': 0.8},
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
        )
        ax.set_title(f'{title}\n(fixed π={PI_FIXED})', fontsize=12, color=NAVY)
        ax.set_xlabel('α (DisGeNET weight multiplier)', fontsize=11)
        ax.set_ylabel('β (STRING weight multiplier)', fontsize=11)

        # Highlight best cell (max accuracy)
        if ax_idx == 0:
            best_idx = grid_df['loocv_acc'].idxmax()
            best_alpha = grid_df.loc[best_idx, 'alpha']
            best_beta  = grid_df.loc[best_idx, 'beta']
            best_acc   = grid_df.loc[best_idx, 'loocv_acc']
            # Draw star on best cell
            col_pos = list(pivot.columns).index(f'α={best_alpha}')
            row_pos = list(pivot.index).index(f'β={best_beta}')
            ax.add_patch(plt.Rectangle(
                (col_pos, row_pos), 1, 1,
                fill=False, edgecolor=NAVY, lw=3, zorder=10
            ))

    # Annotation about method
    axes[0].text(-0.02, -0.18,
        f'Method: std LOOCV (logistic regression); '
        f'100 bootstrap iterations; π={PI_FIXED}',
        transform=axes[0].transAxes, fontsize=8.5,
        color=GREY, style='italic')

    fig.suptitle(
        'Sensitivity Analysis: Grid Search over α × β\n'
        'AD vs Control Pathway Classification',
        fontsize=14, color=NAVY
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print(f"  ✓ Heatmap → {out_path}")

    return best_alpha, best_beta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PART B: π SWEEP LINE PLOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pi_sweep(
    canonical, pathway_sets, all_genes,
    best_alpha: float, best_beta: float,
    cached_pw_ids, cached_jaccard,
) -> pd.DataFrame:
    """Run π sweep at best (α,β) found in Part A."""
    rows = []
    # Pre-score once: scores are the same for all π values
    protein_weights = build_protein_weights(
        all_genes, alpha=best_alpha, beta=best_beta,
        use_disgenet=(best_alpha > 0), use_string=(best_beta > 0),
    )
    scores_df = compute_pathway_scores(canonical, pathway_sets, protein_weights)
    valid_ids = [pid for pid in cached_pw_ids if pid in scores_df.index]

    combined = compute_combined_similarity(
        cached_jaccard, scores_df, valid_ids, use_corr=SCORE_CORR_WEIGHT
    )
    representatives = cluster_and_select_representatives(
        combined, scores_df, valid_ids, pathway_sets, JACCARD_THRESHOLD
    )
    filtered = stage2_variance_filter(scores_df, representatives)

    X_full, y_full, valid_filtered = build_feature_matrix(scores_df, filtered)
    n_samples, n_features = X_full.shape

    # Run stability selection ONCE (100 boots) and record per-feature probabilities
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    selection_counts = np.zeros(n_features)
    successful = 0
    rng = np.random.RandomState(RANDOM_STATE)
    n_boot = 100

    print(f"\n  Pre-running {n_boot} bootstraps at (α={best_alpha}, β={best_beta})...")
    for b in range(n_boot):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        Xb, yb = X_full[boot_idx], y_full[boot_idx]
        if len(np.unique(yb)) < 2:
            continue
        scaler = StandardScaler()
        Xb_s = scaler.fit_transform(Xb)
        try:
            model = LogisticRegressionCV(
                penalty='elasticnet', solver='saga',
                l1_ratios=ELASTIC_NET_L1_RATIOS, Cs=10,
                cv=min(ELASTIC_NET_CV_FOLDS, int(np.bincount(yb.astype(int)).min())),
                max_iter=3000, random_state=RANDOM_STATE,
                class_weight='balanced', scoring='roc_auc', n_jobs=-1,
            )
            model.fit(Xb_s, yb)
            selected = np.abs(model.coef_.flatten()) > 1e-8
            selection_counts += selected.astype(float)
            successful += 1
        except Exception:
            continue

    if successful == 0:
        print("  ⚠ No stable bootstraps — returning empty results")
        return pd.DataFrame()

    probs = selection_counts / successful
    print(f"  Bootstrap done ({successful}/{n_boot} successful)")

    # Now sweep π thresholds
    for pi in PI_GRID:
        stable_mask = probs >= pi
        stable_ids = [valid_filtered[i] for i, flag in enumerate(stable_mask) if flag]
        n_stable = len(stable_ids)

        if n_stable < 2:
            rows.append({'pi': pi, 'n_stable': n_stable,
                         'loocv_acc': np.nan, 'loocv_auc': np.nan,
                         'loocv_kappa': np.nan})
            continue

        X_stable, y_stable, _ = build_feature_matrix(scores_df, stable_ids)
        metrics = loocv_evaluate(X_stable, y_stable, 'logistic')
        rows.append({
            'pi': pi,
            'n_stable': n_stable,
            'loocv_acc': metrics['accuracy'],
            'loocv_auc': metrics['auc'],
            'loocv_kappa': metrics['kappa'],
        })
        print(f"    π={pi:.2f}: n_stable={n_stable}, "
              f"acc={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")

    return pd.DataFrame(rows)


def plot_pi_lineplot(pi_df: pd.DataFrame, best_alpha: float,
                     best_beta: float, out_path: str):
    """Dual-axis line plot: LOOCV accuracy + #biomarkers vs π."""
    if pi_df.empty:
        print("  ⚠ No π sweep data to plot")
        return

    fig, ax1 = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(WHITE)

    pis  = pi_df['pi'].values
    accs = pi_df['loocv_acc'].values
    ns   = pi_df['n_stable'].values

    # ── Accuracy (left axis) ───────────────────────────────────────────────
    color_acc = CORAL
    ax1.plot(pis, accs, 'o-', color=color_acc, linewidth=2.5,
             markersize=8, label='LOOCV Accuracy', zorder=4)
    ax1.fill_between(pis, accs, alpha=0.08, color=color_acc)

    ax1.set_xlabel('Stability Threshold π', fontsize=13)
    ax1.set_ylabel('Standard LOOCV Accuracy', fontsize=13, color=color_acc)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0.3, 1.05)

    # Mark π=0.60 (current pipeline)
    ax1.axvline(PI_FIXED, color=GREY, linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'Current π={PI_FIXED}')

    # ── #Biomarkers (right axis) ───────────────────────────────────────────
    ax2 = ax1.twinx()
    color_n = BLUE
    ax2.bar(pis, ns, width=0.03, alpha=0.35, color=color_n,
            label='#Stable Pathways', zorder=3)
    ax2.plot(pis, ns, 's--', color=color_n, linewidth=1.5,
             markersize=7, alpha=0.85)
    ax2.set_ylabel('Number of Stable Pathways', fontsize=13, color=color_n)
    ax2.tick_params(axis='y', labelcolor=color_n)
    ax2.set_ylim(0, max(ns) * 1.4 + 1)

    # Annotate each point
    for pi_val, acc, n in zip(pis, accs, ns):
        if not np.isnan(acc):
            ax1.annotate(f'{acc:.2f}',
                         xy=(pi_val, acc),
                         xytext=(0, 8), textcoords='offset points',
                         ha='center', fontsize=8.5, color=color_acc)
        ax2.annotate(f'n={n}',
                     xy=(pi_val, n),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', fontsize=8.5, color=color_n)

    # ── Legend ─────────────────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=10, framealpha=0.9)

    ax1.set_xticks(pis)
    ax1.set_xticklabels([f'{p:.2f}' for p in pis], fontsize=10)

    ax1.set_title(
        f'Sensitivity to Stability Threshold π\n'
        f'(Best α={best_alpha}, β={best_beta} from grid search)',
        fontsize=14, color=NAVY
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print(f"  ✓ π line plot → {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "=" * 75)
    print("  TASK 2 — Sensitivity Analysis: α, β Grid + π Sweep")
    print("=" * 75)
    print(f"  Part A: {len(ALPHA_GRID)}×{len(BETA_GRID)}=25 (α,β) combos at π={PI_FIXED}")
    print(f"  Part B: {len(PI_GRID)} π values at best (α,β)")
    print("  Std LOOCV (not nested) used for grid speed; "
          "100 bootstrap iterations per cell\n")

    # ── Load shared data ───────────────────────────────────────────────────
    print("[0/3] Loading shared data...")
    canonical = pd.read_csv(IN_CANONICAL)
    with open(IN_PATHWAYS) as f:
        pathway_sets = json.load(f)
    all_genes = [g for g in canonical['gene_symbol'].dropna().unique()
                 if isinstance(g, str) and len(g) >= 2]
    print(f"  Canonical proteins: {len(canonical)}, genes: {len(all_genes)}, "
          f"pathways: {len(pathway_sets)}")

    # ── Pre-compute Jaccard matrix once ───────────────────────────────────
    print("\n[1/3] Pre-computing Jaccard similarity matrix...")
    t0 = time.perf_counter()
    jaccard, pw_ids_all = compute_jaccard_matrix(pathway_sets)
    cached_pw_ids = list(pathway_sets.keys())
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # ── Part A: α×β grid ──────────────────────────────────────────────────
    print("\n[2/3] Part A — α×β grid search (25 conditions)...")
    grid_df = run_alpha_beta_grid(
        canonical, pathway_sets, all_genes, cached_pw_ids, jaccard
    )
    grid_df.to_csv(OUT_CSV_GRID, index=False)
    print(f"  ✓ Grid results CSV → {OUT_CSV_GRID}")

    # Print top 5 by accuracy
    print("\n  Top 5 (α,β) combinations by LOOCV accuracy:")
    print(f"  {'α':>5} {'β':>5} {'#Stable':>9} {'LOOCV Acc':>11}")
    print("  " + "─" * 35)
    top5 = grid_df.nlargest(5, 'loocv_acc')
    for _, r in top5.iterrows():
        print(f"  {r['alpha']:>5.2f} {r['beta']:>5.2f} "
              f"{r['n_stable']:>9.0f} {r['loocv_acc']:>10.3f}")

    # Find best (α, β)
    best_row = grid_df.loc[grid_df['loocv_acc'].idxmax()]
    best_alpha = float(best_row['alpha'])
    best_beta  = float(best_row['beta'])
    print(f"\n  ✓ Best: α={best_alpha}, β={best_beta}, acc={best_row['loocv_acc']:.3f}")

    # Plot heatmap
    best_alpha, best_beta = plot_heatmap(grid_df, OUT_HEATMAP)

    # ── Part B: π sweep ────────────────────────────────────────────────────
    print(f"\n[3/3] Part B — π sweep at α={best_alpha}, β={best_beta}...")
    pi_df = run_pi_sweep(
        canonical, pathway_sets, all_genes,
        best_alpha, best_beta,
        cached_pw_ids, jaccard,
    )
    plot_pi_lineplot(pi_df, best_alpha, best_beta, OUT_LINEPLOT)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 75)
    print(f"  Grid search: {len(grid_df)} conditions;  "
          f"acc range [{grid_df['loocv_acc'].min():.3f}, "
          f"{grid_df['loocv_acc'].max():.3f}]")
    if not pi_df.empty:
        valid_pi = pi_df.dropna(subset=['loocv_acc'])
        print(f"  π sweep: {len(pi_df)} thresholds; "
              f"#stable range [{pi_df['n_stable'].min():.0f}, "
              f"{pi_df['n_stable'].max():.0f}]")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    main()
