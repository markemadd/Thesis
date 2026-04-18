"""
Task 3 — Binomial Confidence Intervals + Permutation Null Distribution

Computes:
  1. Exact Clopper-Pearson 95% CIs for three LOOCV accuracy values
  2. 1,000-shuffle permutation test null distribution (regenerated)
  3. Publication-quality histogram/KDE of null distribution with annotations

Usage:
    python -m analysis.statistical_inference

Outputs:
    output/permutation_null_distribution.png
    output/confidence_intervals.tex
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
from scipy.stats import gaussian_kde
from scipy.stats import beta as beta_dist

# Attempt statsmodels; fall back to scipy.stats.beta
try:
    from statsmodels.stats.proportion import proportion_confint
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

warnings.filterwarnings('ignore')

from pipeline.config import (
    OUTPUT_DIR, SAMPLE_COLS, PATIENT_COLS, CONTROL_COLS,
    RANDOM_STATE, PERMUTATION_N,
)
from pipeline.phase5_dim_reduction import (
    build_feature_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
IN_STABLE   = os.path.join(OUTPUT_DIR, "stable_pathways.csv")
IN_SCORES   = os.path.join(OUTPUT_DIR, "pathway_scores.csv")
IN_REPORT   = os.path.join(OUTPUT_DIR, "reduction_report.json")
OUT_PLOT    = os.path.join(OUTPUT_DIR, "permutation_null_distribution.png")
OUT_CI_TEX  = os.path.join(OUTPUT_DIR, "confidence_intervals.tex")

# ── Thesis colour palette (from differential_analysis.py) ─────────────────────
NAVY    = '#1B2A4A'
CORAL   = '#E74C3C'
TEAL    = '#1ABC9C'
GOLD    = '#F39C12'
BLUE    = '#2980B9'
GREY    = '#95A5A6'
WHITE   = '#FFFFFF'

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. CLOPPER-PEARSON CONFIDENCE INTERVALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Exact Clopper-Pearson 95% confidence interval for a binomial proportion.

    Uses scipy.stats.beta quantiles (exact method, no normal approximation).
    For k=0: lower bound is 0; for k=n: upper bound is 1.

    Args:
        k: Number of successes
        n: Total trials
        alpha: Significance level (0.05 → 95% CI)

    Returns:
        (lower, upper) bounds in [0, 1]
    """
    if _HAS_STATSMODELS:
        lo, hi = proportion_confint(k, n, alpha=alpha, method='beta')
        return float(lo), float(hi)
    else:
        # Pure scipy implementation
        lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
        return float(lo), float(hi)


def compute_all_cis(n: int = 30) -> list[dict]:
    """
    Compute Clopper-Pearson CIs for all three LOOCV results.

    Sources:
        Standard LOOCV:       29/30 (96.7%) — from validation_results.csv
        Nested LOOCV:         21/30 (70.0%) — from validation_results.csv
        Imputed nested LOOCV: 18/30 (60.0%) — reported in thesis context
    """
    cases = [
        {
            'label': 'Standard LOOCV (Logistic Regression)',
            'k': 29, 'n': n,
            'note': 'Pre-selected features — optimistic estimate',
        },
        {
            'label': 'Nested LOOCV (Logistic Regression)',
            'k': 21, 'n': n,
            'note': 'Features re-selected per fold — unbiased estimate',
        },
        {
            'label': 'Imputed Pipeline — Nested LOOCV',
            'k': 18, 'n': n,
            'note': 'Run on KNN-imputed data matrix',
        },
    ]

    for case in cases:
        k, n_trials = case['k'], case['n']
        lo, hi = clopper_pearson_ci(k, n_trials)
        case['accuracy'] = k / n_trials
        case['ci_lo'] = lo
        case['ci_hi'] = hi
        case['ci_width'] = hi - lo

    return cases


def print_ci_table(cases: list[dict]):
    """Print CIs to stdout in a readable table."""
    print("\n" + "=" * 75)
    print("  CLOPPER-PEARSON 95% CONFIDENCE INTERVALS (n=30)")
    print("=" * 75)
    print(f"  {'Metric':<45} {'k/n':<8} {'Acc':<8} {'95% CI':<22} {'Width'}")
    print("  " + "─" * 70)
    for c in cases:
        print(f"  {c['label']:<45} "
              f"{c['k']}/{c['n']:<6} "
              f"{c['accuracy']:.1%}   "
              f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]    "
              f"{c['ci_width']:.3f}")
    print("  " + "─" * 70)
    print(f"  Method: Clopper-Pearson exact (Beta distribution quantiles)")
    print("=" * 75)


def save_ci_latex(cases: list[dict], out_path: str):
    """Save CIs as a LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Clopper-Pearson 95\% confidence intervals for LOOCV accuracy (n=30)}",
        r"\label{tab:ci}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & $k$/$n$ & Accuracy & 95\% CI & Width \\",
        r"\midrule",
    ]
    for c in cases:
        lo_fmt = f"{c['ci_lo']:.3f}"
        hi_fmt = f"{c['ci_hi']:.3f}"
        lines.append(
            f"{c['label']} & {c['k']}/{c['n']} & "
            f"{c['accuracy']:.1%} & "
            f"[{lo_fmt},\\;{hi_fmt}] & "
            f"{c['ci_width']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{l}{\small Method: Clopper-Pearson exact (Beta distribution quantiles)} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ LaTeX CI table → {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. PERMUTATION NULL DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def regenerate_permutation_null(n_permutations: int = 1000) -> np.ndarray:
    """
    Regenerate the permutation null distribution used in phase5.

    Uses the 13 stable pathways (pre-selected) and exactly the same logistic
    regression LOOCV procedure as `permutation_test()` in phase5_dim_reduction.
    RANDOM_STATE=42 ensures 100% reproducible results.

    Big-O: O(N * n * p) where N=permutations, n=30 samples, p=13 pathways.
    Runtime: ~2-5 min depending on hardware.
    """
    # Load stable pathways and scores
    stable = pd.read_csv(IN_STABLE)
    scores = pd.read_csv(IN_SCORES, index_col=0)

    stable_ids = stable['pathway_id'].tolist()
    X, y, valid_ids = build_feature_matrix(scores, stable_ids)

    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} pathways")
    print(f"  Running {n_permutations} permutations...")

    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score

    rng = np.random.RandomState(RANDOM_STATE)
    perm_accuracies = []
    real_acc = 0.967  # from validation_results.csv

    loo = LeaveOneOut()

    for p in range(n_permutations):
        y_perm = rng.permutation(y)
        y_pred = np.zeros(len(y_perm))

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y_perm[train_idx]

            if len(np.unique(y_train)) < 2:
                y_pred[test_idx] = rng.choice([0, 1])
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            try:
                model = LogisticRegressionCV(
                    penalty='l2', solver='lbfgs',
                    cv=min(5, int(np.bincount(y_train.astype(int)).min())),
                    max_iter=2000,
                    class_weight='balanced',
                    random_state=RANDOM_STATE,
                )
                model.fit(X_tr_s, y_train)
                y_pred[test_idx] = model.predict(X_te_s)
            except Exception:
                y_pred[test_idx] = rng.choice([0, 1])

        perm_acc = accuracy_score(y_perm, y_pred)
        perm_accuracies.append(perm_acc)

        if (p + 1) % 100 == 0:
            n_better = sum(a >= real_acc for a in perm_accuracies)
            print(f"    {p + 1}/{n_permutations}  "
                  f"(mean={np.mean(perm_accuracies):.3f}, "
                  f"n≥{real_acc:.3f}: {n_better})")

    return np.array(perm_accuracies)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. PLOT: PERMUTATION NULL DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_permutation_null(perm_accuracies: np.ndarray,
                          real_acc: float = 0.967,
                          nested_acc: float = 0.700,
                          out_path: str = OUT_PLOT):
    """
    Publication-quality plot of the permutation null distribution.

    Shows:
      - Histogram of 1,000 permuted LOOCV accuracies
      - KDE overlay
      - Vertical line at observed std LOOCV accuracy (test statistic)
      - Dashed vertical line at nested LOOCV accuracy (for context)
      - Shaded p-value region (area to the right of real_acc)
      - Annotated empirical p-value
    """
    n_perm = len(perm_accuracies)
    n_better = (perm_accuracies >= real_acc).sum()
    p_emp = (n_better + 1) / (n_perm + 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(WHITE)

    # ── Histogram ──────────────────────────────────────────────────────────
    bins = np.arange(0.30, 1.01, 0.033)
    counts, edges, patches = ax.hist(
        perm_accuracies, bins=bins,
        color=BLUE, alpha=0.55, edgecolor=WHITE, linewidth=0.5,
        label=f'Null distribution (N={n_perm} permutations)',
        zorder=2
    )

    # Shade p-value region (bins at or above real_acc)
    for patch, left in zip(patches, edges[:-1]):
        if left >= real_acc - 0.001:
            patch.set_facecolor(CORAL)
            patch.set_alpha(0.85)
            patch.set_zorder(3)

    # ── KDE overlay ────────────────────────────────────────────────────────
    kde_x = np.linspace(perm_accuracies.min() - 0.05,
                         perm_accuracies.max() + 0.05, 400)
    try:
        kde = gaussian_kde(perm_accuracies, bw_method='silverman')
        kde_y = kde(kde_x)
        # Scale KDE to match histogram height
        bin_width = bins[1] - bins[0]
        kde_y_scaled = kde_y * n_perm * bin_width
        ax.plot(kde_x, kde_y_scaled, color=NAVY, linewidth=2.2,
                label='KDE (Silverman bandwidth)', zorder=4)
    except Exception:
        pass  # KDE rarely fails but skip gracefully

    # ── Vertical lines ─────────────────────────────────────────────────────
    ax.axvline(real_acc, color=CORAL, linewidth=2.5, linestyle='-', zorder=5,
               label=f'Std LOOCV accuracy = {real_acc:.1%} (test statistic)')
    ax.axvline(nested_acc, color=GOLD, linewidth=2.0, linestyle='--', zorder=5,
               label=f'Nested LOOCV accuracy = {nested_acc:.1%} (reference)')

    # ── Annotation box ─────────────────────────────────────────────────────
    ax.annotate(
        f'p = {p_emp:.4f}\n({n_better}/{n_perm} permutations\n≥ observed accuracy)',
        xy=(real_acc, max(counts) * 0.60),
        xytext=(real_acc - 0.22, max(counts) * 0.75),
        fontsize=11, color=CORAL, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=CORAL, lw=2),
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3F3',
                  edgecolor=CORAL, alpha=0.92)
    )

    # ── Chance level ───────────────────────────────────────────────────────
    ax.axvline(0.667, color=GREY, linewidth=1.2, linestyle=':', zorder=3,
               alpha=0.6, label='Majority-class baseline (66.7%)')

    # ── Axis labels & title ────────────────────────────────────────────────
    ax.set_xlabel('LOOCV Accuracy (Shuffled Labels)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(
        'Permutation Null Distribution of LOOCV Accuracy\n'
        'AD vs Control Classification — 13 Stable Pathway Biomarkers',
        fontsize=14, color=NAVY, pad=12
    )

    # ── Legend ─────────────────────────────────────────────────────────────
    p_patch = mpatches.Patch(color=CORAL, alpha=0.85,
                              label=f'p-value region (p={p_emp:.4f})')
    handles, labels_l = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [p_patch],
              loc='upper left', fontsize=9.5, framealpha=0.90)

    # ── Summary stats text ─────────────────────────────────────────────────
    stats_txt = (
        f'Null: mean={np.mean(perm_accuracies):.3f}, '
        f'SD={np.std(perm_accuracies):.3f}, '
        f'max={np.max(perm_accuracies):.3f}\n'
        f'Empirical p-value: {p_emp:.4f}'
    )
    ax.text(0.98, 0.97, stats_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=9.5, color=NAVY,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA',
                      edgecolor='#DDD', alpha=0.92))

    ax.set_xlim(0.25, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print(f"  ✓ Permutation plot → {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "=" * 75)
    print("  TASK 3 — Binomial CIs + Permutation Null Distribution")
    print("=" * 75)

    # ── 1. Confidence intervals ───────────────────────────────────────────
    print("\n[1/3] Computing Clopper-Pearson 95% confidence intervals...")
    ci_cases = compute_all_cis(n=30)
    print_ci_table(ci_cases)
    save_ci_latex(ci_cases, OUT_CI_TEX)

    # ── 2. Permutation null distribution ──────────────────────────────────
    print("\n[2/3] Regenerating permutation null distribution...")
    print(f"  Using RANDOM_STATE={RANDOM_STATE}, N={PERMUTATION_N} permutations")
    print("  (This matches the original pipeline exactly → reproducible results)")

    perm_accuracies = regenerate_permutation_null(n_permutations=PERMUTATION_N)

    # Verify against stored report
    if os.path.exists(IN_REPORT):
        with open(IN_REPORT) as f:
            report = json.load(f)
        stored = report.get('stage4_validation', {}).get('permutation', {})
        stored_mean = stored.get('mean_perm_accuracy', None)
        if stored_mean is not None:
            diff = abs(np.mean(perm_accuracies) - stored_mean)
            status = "✓ matches" if diff < 0.01 else f"⚠ differs by {diff:.3f}"
            print(f"  Stored mean: {stored_mean:.3f} | "
                  f"Regenerated: {np.mean(perm_accuracies):.3f} — {status}")

    # Load empirical values from report
    real_acc    = 0.967   # std LOOCV (the test statistic used in the perm test)
    nested_acc  = 0.700   # nested LOOCV (shown for context)

    # ── 3. Plot ───────────────────────────────────────────────────────────
    print("\n[3/3] Generating permutation null distribution plot...")
    plot_permutation_null(perm_accuracies, real_acc=real_acc,
                          nested_acc=nested_acc, out_path=OUT_PLOT)

    n_better = (perm_accuracies >= real_acc).sum()
    p_emp = (n_better + 1) / (PERMUTATION_N + 1)

    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    print(f"  Permutation test: {n_better}/{PERMUTATION_N} ≥ {real_acc:.3f}  →  p = {p_emp:.4f}")
    print(f"  Null distribution: mean={np.mean(perm_accuracies):.3f}, "
          f"SD={np.std(perm_accuracies):.3f}, "
          f"max={np.max(perm_accuracies):.3f}")
    print(f"\n  Clopper-Pearson 95% CIs:")
    for c in ci_cases:
        print(f"    {c['label']}")
        print(f"      {c['k']}/{c['n']} = {c['accuracy']:.1%}  "
              f"→ [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]")
    print("=" * 75 + "\n")


if __name__ == '__main__':
    main()
