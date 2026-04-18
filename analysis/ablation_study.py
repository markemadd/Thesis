"""
Task 1 — Ablation Study: ssGSEA Weighting Conditions

Tests 4 weighting schemes for the protein importance weights used in ssGSEA:

  (A) Unweighted:     w_i = 1.0
  (B) DisGeNET-only:  w_i = 1 + alpha * d_i  (beta=0)
  (C) STRING-only:    w_i = 1 + beta * c_i   (alpha=0)
  (D) Combined:       w_i = 1 + alpha * d_i + beta * c_i  (current pipeline)

Uses alpha=1.0, beta=1.0 for conditions B-D; curated fallback weights (no API calls).
Runs the full Phase 3 + Phase 5 pipeline for each condition.
Jaccard matrix is cached after first computation to avoid redundant work.

PRIMARY METRIC: Nested LOOCV accuracy (features re-selected in every fold).
SECONDARY METRIC: Standard LOOCV (provided for reference, known to be inflated
  because features are pre-selected before cross-validation).

Outputs:
  output/ablation_comparison.png   — 4-panel chart: nested LOOCV, std LOOCV,
                                     #stable pathways, AUC
  output/ablation_table.tex        — LaTeX comparison table (nested LOOCV primary)
  stdout                           — full summary statistics

Usage:
    python -m analysis.ablation_study

Compute strategy:
  - Jaccard matrix computed ONCE (cached) — avoids O(P^2*G) redundant work
  - Nested LOOCV uses 50 inner bootstraps per fold (vs 200 in main pipeline)
    to stay tractable; sufficient for comparing conditions (not for reporting
    the single published accuracy — use 200 for that in phase5 directly)
  - n_jobs=-1 for all sklearn models
  - Runtime: ~4h total (approx 40 min nested LOOCV x 4 conditions + scoring)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score

warnings.filterwarnings('ignore')

from pipeline.config import (
    OUTPUT_DIR, SAMPLE_COLS, PATIENT_COLS, CONTROL_COLS,
    RANDOM_STATE, STABILITY_N_BOOTSTRAP, STABILITY_THRESHOLD,
    ELASTIC_NET_L1_RATIOS, ELASTIC_NET_CV_FOLDS,
    JACCARD_THRESHOLD, JACCARD_SENSITIVITY, SCORE_CORR_WEIGHT,
    IQR_PERCENTILE_CUTOFF,
)
from pipeline.phase3_pathway_scoring import (
    compute_pathway_scores,
    ssgsea_score,
)
from pipeline.phase5_dim_reduction import (
    compute_jaccard_matrix,
    compute_combined_similarity,
    cluster_and_select_representatives,
    stage2_variance_filter,
    stage3_stability_selection,
    loocv_evaluate,
    nested_loocv,
    build_feature_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
IN_CANONICAL = os.path.join(OUTPUT_DIR, "canonical_log2.csv")
IN_PATHWAYS  = os.path.join(OUTPUT_DIR, "pathway_gene_sets.json")
IN_SCORES_D  = os.path.join(OUTPUT_DIR, "pathway_scores.csv")
IN_STABLE_D  = os.path.join(OUTPUT_DIR, "stable_pathways.csv")
OUT_PNG      = os.path.join(OUTPUT_DIR, "ablation_comparison.png")
OUT_TEX      = os.path.join(OUTPUT_DIR, "ablation_table.tex")

# ── Thesis colour palette ──────────────────────────────────────────────────────
NAVY  = '#1B2A4A'
CORAL = '#E74C3C'
TEAL  = '#1ABC9C'
GOLD  = '#F39C12'
BLUE  = '#2980B9'
GREY  = '#95A5A6'
WHITE = '#FFFFFF'
PURPLE = '#8E44AD'

CONDITION_COLOURS = {
    'A': TEAL,
    'B': BLUE,
    'C': GOLD,
    'D': CORAL,
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'figure.facecolor': WHITE,
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#DDDDDD',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CURATED WEIGHT DICTIONARIES (fallback used in phase3 when API unavailable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# DisGeNET AD gene-disease association scores (d_i values from phase3 fallback)
DISGENET_CURATED: dict[str, float] = {
    'APOE': 0.95, 'CLU': 0.85, 'BIN1': 0.80, 'PICALM': 0.75,
    'CR1': 0.70, 'ABCA7': 0.70, 'TREM2': 0.80, 'SORL1': 0.75,
    'APP': 0.95, 'MAPT': 0.90, 'PSEN1': 0.90, 'PSEN2': 0.85,
    'BACE1': 0.80, 'BACE2': 0.60,
    'CST3': 0.65, 'TTR': 0.60, 'GSN': 0.55, 'SERPINA3': 0.60,
    'CHI3L1': 0.65, 'NRGN': 0.70, 'NEFL': 0.75,
    'C1QA': 0.60, 'C1QB': 0.60, 'C1QC': 0.60,
    'C2': 0.50, 'C3': 0.65, 'C4A': 0.55, 'C4B': 0.55,
    'CFH': 0.55, 'CFI': 0.50, 'CFB': 0.50, 'C5': 0.50,
    'APOA1': 0.50, 'APOA4': 0.40, 'APOB': 0.40,
    'APOC1': 0.50, 'APOC3': 0.45, 'APOD': 0.55, 'APOM': 0.40,
    'APOL1': 0.40,
    'CRP': 0.55, 'SAA1': 0.50, 'SAA2': 0.45,
    'SERPINA1': 0.50, 'SERPINF2': 0.40,
    'FGA': 0.40, 'FGB': 0.40, 'FGG': 0.40, 'PLG': 0.35,
    'SYN1': 0.60, 'SYP': 0.55, 'SNAP25': 0.60, 'STX1A': 0.55,
    'ALB': 0.20, 'TF': 0.30, 'HP': 0.35, 'HPX': 0.30,
}

# STRING PPI network centrality — proxy from high-interaction CSF proteins.
# These are normalized degree centrality scores (0–1) estimated from the
# STRING network for high-confidence interactions (score ≥ 700).
# Values are consistent with the curated fallback used during phase3 execution.
STRING_CURATED: dict[str, float] = {
    'ALB': 0.95, 'APOE': 0.90, 'APP': 0.88, 'TP53': 0.85, 'EGFR': 0.82,
    'AKT1': 0.80, 'MAPT': 0.78, 'PSEN1': 0.75, 'CLU': 0.70, 'TREM2': 0.68,
    'C3': 0.65, 'CFH': 0.60, 'FN1': 0.60, 'VTN': 0.58, 'APOA1': 0.55,
    'HPX': 0.52, 'HP': 0.50, 'TF': 0.48, 'TTR': 0.45, 'CST3': 0.42,
    'NEFL': 0.40, 'SNAP25': 0.38, 'SYN1': 0.35, 'CHI3L1': 0.33,
    'NRGN': 0.30, 'SERPINA1': 0.28, 'CRP': 0.25, 'PLG': 0.22,
    'FGA': 0.20, 'FGB': 0.20, 'FGG': 0.20, 'BIN1': 0.18,
    'PSEN2': 0.22, 'BACE1': 0.18, 'SORL1': 0.16, 'ABCA7': 0.14,
}


def build_protein_weights(gene_symbols: list[str],
                           alpha: float,
                           beta: float,
                           use_disgenet: bool = True,
                           use_string: bool = True) -> dict[str, float]:
    """
    Construct protein importance weights for ssGSEA under a given condition.

    Formula: w_i = 1.0 + α·d_i + β·c_i
    Default weight (proteins not in either DB): 1.0 (neutral baseline)

    Args:
        gene_symbols: All gene symbols in the dataset
        alpha: Multiplier for DisGeNET d_i scores
        beta: Multiplier for STRING c_i scores
        use_disgenet: Include DisGeNET component (if False: α term zeroed)
        use_string: Include STRING component (if False: β term zeroed)
    """
    weights = {}
    for gene in gene_symbols:
        d_i = DISGENET_CURATED.get(gene, 0.0) if use_disgenet else 0.0
        c_i = STRING_CURATED.get(gene, 0.0)   if use_string  else 0.0
        weights[gene] = 1.0 + alpha * d_i + beta * c_i
    return weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FULL PIPELINE PER CONDITION: Phase3 + Phase5 Stages 1→4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Number of inner bootstrap iterations for nested LOOCV in the ablation.
# Main pipeline uses 200; we use 50 here for tractability across 4 conditions.
# 50 bootstraps is sufficient for comparing relative condition differences.
ABLATION_INNER_BOOTSTRAPS = 50


def run_pipeline_condition(
    canonical: pd.DataFrame,
    pathway_sets: dict,
    gene_symbols: list[str],
    protein_weights: dict,
    cached_pw_ids: list[str],
    cached_jaccard: np.ndarray,
) -> dict:
    """
    Run Phase 3 + Phase 5 (Stages 1-4a-4b) for one weighting condition.

    Returns BOTH standard LOOCV and nested LOOCV metrics.

    Standard LOOCV (Stage 4a): biased — features pre-selected before CV loop.
      Use only as an upper-bound reference, NOT for condition comparison.

    Nested LOOCV (Stage 4b): unbiased — features re-selected inside each fold.
      PRIMARY metric for comparing weighting conditions.
      Uses ABLATION_INNER_BOOTSTRAPS (50) for tractability.

    Returns:
        dict with keys: n_stable, loocv_acc, loocv_auc, loocv_kappa,
                        nested_acc, nested_auc, nested_kappa,
                        stable_ids, filtered_ids, scores_df
    """
    print("    [3] Computing weighted ssGSEA scores...")
    scores_df = compute_pathway_scores(canonical, pathway_sets, protein_weights)

    valid_ids = [pid for pid in cached_pw_ids if pid in scores_df.index]

    print("    [5.1] De-redundancy (using cached Jaccard)...")
    combined = compute_combined_similarity(
        cached_jaccard, scores_df, valid_ids, use_corr=SCORE_CORR_WEIGHT
    )
    representatives = cluster_and_select_representatives(
        combined, scores_df, valid_ids, pathway_sets, JACCARD_THRESHOLD
    )
    print(f"      Non-redundant: {len(representatives)}")

    print("    [5.2] Variance filtering...")
    filtered = stage2_variance_filter(scores_df, representatives)
    print(f"      Filtered: {len(filtered)}")

    print("    [5.3] Stability selection...")
    stable_df = stage3_stability_selection(scores_df, filtered)
    n_stable = len(stable_df)
    stable_ids = stable_df['pathway_id'].tolist() if n_stable >= 2 else []
    print(f"      Stable pathways: {n_stable}")

    nan_result = {
        'n_stable': n_stable,
        'loocv_acc': np.nan, 'loocv_auc': np.nan, 'loocv_kappa': np.nan,
        'nested_acc': np.nan, 'nested_auc': np.nan, 'nested_kappa': np.nan,
        'stable_ids': stable_ids, 'filtered_ids': filtered,
        'scores_df': scores_df,
    }
    if n_stable < 2:
        return nan_result

    # ── Stage 4a: Standard LOOCV (biased reference) ────────────────────────
    print("    [5.4a] Standard LOOCV (reference — biased due to pre-selected features)...")
    X, y, _ = build_feature_matrix(scores_df, stable_ids)
    std_metrics = loocv_evaluate(X, y, 'logistic')
    print(f"      Std  LOOCV: Acc={std_metrics['accuracy']:.3f} "
          f"AUC={std_metrics['auc']:.3f} kappa={std_metrics['kappa']:.3f}")

    # ── Stage 4b: Nested LOOCV (primary — unbiased) ────────────────────────
    print(f"    [5.4b] Nested LOOCV (PRIMARY — {ABLATION_INNER_BOOTSTRAPS} inner bootstraps)...")
    # Temporarily override the bootstrap count for speed
    import pipeline.config as _cfg
    _orig_bootstraps = _cfg.STABILITY_N_BOOTSTRAP
    _cfg.STABILITY_N_BOOTSTRAP = ABLATION_INNER_BOOTSTRAPS
    try:
        nested_metrics = nested_loocv(scores_df, filtered)
    finally:
        _cfg.STABILITY_N_BOOTSTRAP = _orig_bootstraps  # always restore
    print(f"      Nested LOOCV: Acc={nested_metrics['accuracy']:.3f} "
          f"AUC={nested_metrics['auc']:.3f} kappa={nested_metrics['kappa']:.3f}")

    return {
        'n_stable': n_stable,
        'loocv_acc': std_metrics['accuracy'],
        'loocv_auc': std_metrics['auc'],
        'loocv_kappa': std_metrics['kappa'],
        'nested_acc': nested_metrics['accuracy'],
        'nested_auc': nested_metrics['auc'],
        'nested_kappa': nested_metrics['kappa'],
        'stable_ids': stable_ids,
        'filtered_ids': filtered,
        'scores_df': scores_df,
    }


def jaccard_overlap(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two pathway ID sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OUTPUT: TABLE + PLOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_ablation_table(rows: list[dict]):
    """Print ablation results to stdout, nested LOOCV as primary metric."""
    print("\n" + "=" * 100)
    print("  ABLATION STUDY RESULTS")
    print("  Nested LOOCV = PRIMARY (unbiased)  |  Std LOOCV = REFERENCE (inflated, pre-selected features)")
    print("=" * 100)
    hdr = (f"  {'Condition':<32} {'#Stable':>7} "
           f"{'Nested Acc':>11} {'Nested AUC':>11} {'Nested k':>9} "
           f"{'Std Acc':>9} {'Jaccard/D':>10}")
    print(hdr)
    print("  " + "─" * 95)
    for r in rows:
        marker = " <-- PRIMARY" if r['cond'] == 'D' else ""
        print(f"  {r['label']:<32} {r['n_stable']:>7} "
              f"{r['nested_acc']:>10.3f} {r['nested_auc']:>11.3f} "
              f"{r['nested_kappa']:>9.3f} "
              f"{r['loocv_acc']:>9.3f} {r['jaccard_vs_D']:>10.3f}{marker}")
    print("=" * 100)


def save_latex_table(rows: list[dict], out_path: str):
    """Save ablation results as a LaTeX table, nested LOOCV as primary."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation study: effect of ssGSEA weighting scheme on pipeline"
        r" performance. Nested LOOCV is the primary unbiased metric; standard"
        r" LOOCV is provided as a reference (known to be inflated due to"
        r" feature pre-selection).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Condition & \#Stable & \multicolumn{3}{c}{Nested LOOCV (primary)}"
        r" & Std Acc & Jaccard$_D$ \\",
        r"\cmidrule(lr){3-5}",
        r" & & Acc & AUC & $\kappa$ & (biased) & \\",
        r"\midrule",
    ]
    for r in rows:
        n_acc = f"{r['nested_acc']:.3f}" if not np.isnan(r['nested_acc']) else "--"
        n_auc = f"{r['nested_auc']:.3f}" if not np.isnan(r['nested_auc']) else "--"
        n_kap = f"{r['nested_kappa']:.3f}" if not np.isnan(r['nested_kappa']) else "--"
        s_acc = f"{r['loocv_acc']:.3f}" if not np.isnan(r['loocv_acc']) else "--"
        label = r'\textbf{' + r['label'] + r'}' if r['cond'] == 'D' else r['label']
        lines.append(
            f"{label} & {r['n_stable']} & {n_acc} & {n_auc} & {n_kap}"
            f" & {s_acc} & {r['jaccard_vs_D']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{8}{l}{\small Nested LOOCV uses 50 inner bootstraps per fold;"
        r" conditions B--D use $\alpha=\beta=1$; Jaccard$_D$ = overlap with condition D stable set.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ LaTeX table -> {out_path}")


def plot_ablation(rows: list[dict], out_path: str):
    """
    4-panel publication chart:
      Top-left:  Nested LOOCV accuracy (PRIMARY metric)
      Top-right: Standard LOOCV accuracy (biased reference)
      Bottom-left:  Number of stable pathways selected per condition
      Bottom-right: AUC comparison (nested vs standard)
    """
    labels  = [r['cond_short'] for r in rows]
    n_accs  = [r['nested_acc']  for r in rows]   # primary
    s_accs  = [r['loocv_acc']   for r in rows]   # reference
    n_aucs  = [r['nested_auc']  for r in rows]
    s_aucs  = [r['loocv_auc']   for r in rows]
    n_stab  = [r['n_stable']    for r in rows]
    colours = [CONDITION_COLOURS[r['cond']] for r in rows]
    x = range(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(WHITE)

    def _bar_panel(ax, values, ylabel, title, is_primary=False):
        bars = ax.bar(x, values, color=colours,
                      edgecolor=WHITE, linewidth=0.8, alpha=0.88, width=0.6)
        for i, (bar, v) in enumerate(zip(bars, values)):
            if not np.isnan(v):
                ax.text(i, v + (0.012 if v < 1 else 0.3),
                        f'{v:.1%}' if v <= 1.0 else str(int(v)),
                        ha='center', va='bottom', fontsize=11,
                        fontweight='bold', color=NAVY)
        ax.axhline(0.667, color=GREY, linewidth=1.2, linestyle=':', alpha=0.5,
                   label='Majority-class baseline (66.7%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0.0, 1.10)
        border_color = CORAL if is_primary else GREY
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5 if is_primary else 1)
        title_suffix = '  [PRIMARY]' if is_primary else '  [reference — inflated]'
        ax.set_title(title + title_suffix, fontsize=12, color=CORAL if is_primary else GREY)
        ax.legend(fontsize=8)

    # Top-left: Nested LOOCV (primary)
    _bar_panel(axes[0, 0], n_accs,
               'Nested LOOCV Accuracy', 'Nested LOOCV Accuracy', is_primary=True)

    # Top-right: Standard LOOCV (biased reference)
    _bar_panel(axes[0, 1], s_accs,
               'Standard LOOCV Accuracy', 'Standard LOOCV Accuracy', is_primary=False)

    # Bottom-left: Stable pathway count
    ax = axes[1, 0]
    bars = ax.bar(x, n_stab, color=colours,
                  edgecolor=WHITE, linewidth=0.8, alpha=0.88, width=0.6)
    for i, (bar, n) in enumerate(zip(bars, n_stab)):
        ax.text(i, n + 0.15, str(n),
                ha='center', va='bottom', fontsize=12, fontweight='bold', color=NAVY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Number of Stable Pathways', fontsize=11)
    ax.set_ylim(0, max(n_stab) * 1.35 + 1)
    ax.set_title('Stable Pathways Selected per Condition', fontsize=12, color=NAVY)

    # Bottom-right: AUC comparison (nested vs std)
    ax = axes[1, 1]
    width = 0.35
    x_pos = np.arange(len(rows))
    ax.bar(x_pos - width/2, n_aucs, width, color=CORAL, alpha=0.85,
           label='Nested LOOCV AUC', edgecolor=WHITE)
    ax.bar(x_pos + width/2, s_aucs, width, color=BLUE, alpha=0.70,
           label='Std LOOCV AUC', edgecolor=WHITE)
    for i, (na, sa) in enumerate(zip(n_aucs, s_aucs)):
        if not np.isnan(na):
            ax.text(i - width/2, na + 0.01, f'{na:.2f}',
                    ha='center', va='bottom', fontsize=9, color=CORAL, fontweight='bold')
        if not np.isnan(sa):
            ax.text(i + width/2, sa + 0.01, f'{sa:.2f}',
                    ha='center', va='bottom', fontsize=9, color=BLUE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=9)
    ax.set_title('AUC: Nested vs Standard LOOCV', fontsize=12, color=NAVY)

    # Shared legend for conditions
    legend_elements = [
        mpatches.Patch(color=CONDITION_COLOURS[r['cond']], label=r['label'])
        for r in rows
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.95, edgecolor='#DDD')

    fig.suptitle(
        'Ablation Study: ssGSEA Protein Weighting Conditions\n'
        'AD vs Control | Red border = unbiased primary metric',
        fontsize=14, color=NAVY, y=1.01
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print(f"  Ablation chart -> {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "=" * 75)
    print("  TASK 1 — Ablation Study: ssGSEA Weighting Conditions")
    print("=" * 75)
    print("  Conditions: (A) Unweighted  (B) DisGeNET-only")
    print("              (C) STRING-only  (D) Combined [current]")
    print("  Curated fallback weights used — NO live API calls\n")

    # ── Load shared data ───────────────────────────────────────────────────
    print("[0/5] Loading shared data...")
    canonical = pd.read_csv(IN_CANONICAL)
    with open(IN_PATHWAYS) as f:
        pathway_sets = json.load(f)

    all_genes = [g for g in canonical['gene_symbol'].dropna().unique()
                 if isinstance(g, str) and len(g) >= 2]
    print(f"  Canonical proteins: {len(canonical)}")
    print(f"  Unique genes: {len(all_genes)}")
    print(f"  Pathways: {len(pathway_sets)}")

    # ── Pre-compute Jaccard matrix ONCE ───────────────────────────────────
    print("\n[1/5] Pre-computing Jaccard similarity matrix (cached for all conditions)...")
    import time
    t0 = time.perf_counter()
    jaccard, pw_ids_all = compute_jaccard_matrix(pathway_sets)
    print(f"  Jaccard matrix ({len(pw_ids_all)}×{len(pw_ids_all)}) "
          f"computed in {time.perf_counter()-t0:.1f}s")

    # Filter to pathways in pathway_sets
    valid_pw_ids = list(pathway_sets.keys())

    # ── Define conditions ──────────────────────────────────────────────────
    ALPHA, BETA = 1.0, 1.0  # same as implied by current pipeline (combined weight)

    conditions = [
        {
            'cond': 'A',
            'cond_short': '(A) Unweighted',
            'label': '(A) Unweighted  [w=1]',
            'alpha': 0.0, 'beta': 0.0,
            'use_disgenet': False, 'use_string': False,
        },
        {
            'cond': 'B',
            'cond_short': '(B) DisGeNET',
            'label': '(B) DisGeNET-only  [w=1+α·d]',
            'alpha': ALPHA, 'beta': 0.0,
            'use_disgenet': True, 'use_string': False,
        },
        {
            'cond': 'C',
            'cond_short': '(C) STRING',
            'label': '(C) STRING-only  [w=1+β·c]',
            'alpha': 0.0, 'beta': BETA,
            'use_disgenet': False, 'use_string': True,
        },
        {
            'cond': 'D',
            'cond_short': '(D) Combined',
            'label': '(D) Combined  [w=1+α·d+β·c]',
            'alpha': ALPHA, 'beta': BETA,
            'use_disgenet': True, 'use_string': True,
        },
    ]

    # ── Run each condition ─────────────────────────────────────────────────
    condition_results = {}

    for i, cond in enumerate(conditions):
        print(f"\n[{i+2}/5] Running condition {cond['cond']}: {cond['label']}")
        protein_weights = build_protein_weights(
            all_genes,
            alpha=cond['alpha'], beta=cond['beta'],
            use_disgenet=cond['use_disgenet'], use_string=cond['use_string'],
        )
        result = run_pipeline_condition(
            canonical, pathway_sets, all_genes, protein_weights,
            valid_pw_ids, jaccard,
        )
        condition_results[cond['cond']] = result

    # ── Compute Jaccard overlap vs condition D ─────────────────────────────
    stable_D = set(condition_results['D']['stable_ids'])

    rows = []
    for cond in conditions:
        res = condition_results[cond['cond']]
        stab_set = set(res['stable_ids'])
        j_vs_D = jaccard_overlap(stab_set, stable_D)

        rows.append({
            'cond': cond['cond'],
            'cond_short': cond['cond_short'],
            'label': cond['label'],
            'n_stable': res['n_stable'],
            # Primary (unbiased) metrics:
            'nested_acc': res.get('nested_acc', np.nan),
            'nested_auc': res.get('nested_auc', np.nan),
            'nested_kappa': res.get('nested_kappa', np.nan),
            # Reference (biased) metrics:
            'loocv_acc': res['loocv_acc'],
            'loocv_auc': res['loocv_auc'],
            'loocv_kappa': res['loocv_kappa'],
            'jaccard_vs_D': j_vs_D,
        })

    # ── Print & save outputs ───────────────────────────────────────────────
    print_ablation_table(rows)
    save_latex_table(rows, OUT_TEX)
    plot_ablation(rows, OUT_PNG)

    print("\n" + "=" * 75)
    print("  ABLATION COMPLETE")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    main()
