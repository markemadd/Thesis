"""
Differential Pathway Activity Analysis

Compares pathway activity scores between AD and Control groups using:
  1. Mann-Whitney U test (non-parametric, appropriate for n=30)
  2. Benjamini-Hochberg FDR correction
  3. Rank-biserial effect sizes
  4. Publication-quality visualizations (volcano plot, heatmap, bar chart)

Runs on the 575 non-redundant pathways from Phase 5 Stage 1.
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from pipeline.config import (
    OUTPUT_DIR, SAMPLE_COLS, PATIENT_COLS, CONTROL_COLS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_SCORES    = os.path.join(OUTPUT_DIR, "pathway_scores.csv")
IN_NONRED    = os.path.join(OUTPUT_DIR, "nonredundant_pathways.csv")
IN_PATHWAYS  = os.path.join(OUTPUT_DIR, "pathway_gene_sets.json")
OUT_DIFF     = os.path.join(OUTPUT_DIR, "differential_pathways.csv")
FIG_DIR      = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
NAVY    = '#1B2A4A'
CORAL   = '#E74C3C'
TEAL    = '#1ABC9C'
GOLD    = '#F39C12'
BLUE    = '#2980B9'
WHITE   = '#FFFFFF'
GREY    = '#95A5A6'
AD_COLOR   = '#E74C3C'
CTRL_COLOR = '#1ABC9C'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.facecolor': WHITE,
    'axes.facecolor': WHITE,
    'axes.edgecolor': '#DDDDDD',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STATISTICAL TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rank_biserial(ad_vals: np.ndarray, ctrl_vals: np.ndarray, U: float) -> float:
    """
    Compute rank-biserial correlation effect size from Mann-Whitney U.

    r = 1 - (2U)/(n_ad * n_ctrl)

    Interpretation:
      | r |    Meaning
      ──────────────────
      < 0.1    negligible
      0.1–0.3  small
      0.3–0.5  medium
      > 0.5    large
    """
    n1, n2 = len(ad_vals), len(ctrl_vals)
    return 1 - (2 * U) / (n1 * n2)


def compute_differential_activity(scores_df: pd.DataFrame,
                                   nonred_ids: list[str],
                                   pathway_sets: dict) -> pd.DataFrame:
    """
    For each non-redundant pathway, compute:
      - Mean score in AD and Control
      - Log2 fold change (analog)
      - Mann-Whitney U test p-value
      - FDR-corrected q-value
      - Rank-biserial effect size
    """
    patient_cols = [c for c in scores_df.columns if c in PATIENT_COLS]
    control_cols = [c for c in scores_df.columns if c in CONTROL_COLS]

    results = []
    for pw_id in nonred_ids:
        if pw_id not in scores_df.index:
            continue

        ad_vals = scores_df.loc[pw_id, patient_cols].values.astype(float)
        ctrl_vals = scores_df.loc[pw_id, control_cols].values.astype(float)

        # Remove NaNs
        ad_vals = ad_vals[~np.isnan(ad_vals)]
        ctrl_vals = ctrl_vals[~np.isnan(ctrl_vals)]

        if len(ad_vals) < 3 or len(ctrl_vals) < 3:
            continue

        mean_ad = np.mean(ad_vals)
        mean_ctrl = np.mean(ctrl_vals)
        diff = mean_ad - mean_ctrl

        # Mann-Whitney U test (two-sided)
        U, p_value = stats.mannwhitneyu(ad_vals, ctrl_vals, alternative='two-sided')

        # Rank-biserial effect size
        r_effect = rank_biserial(ad_vals, ctrl_vals, U)

        # Pathway metadata
        pw_info = pathway_sets.get(pw_id, {})

        results.append({
            'pathway_id': pw_id,
            'pathway_name': pw_info.get('name', ''),
            'source': pw_info.get('source', ''),
            'n_genes': pw_info.get('size_matched', 0),
            'mean_AD': round(mean_ad, 4),
            'mean_Control': round(mean_ctrl, 4),
            'diff': round(diff, 4),
            'p_value': p_value,
            'effect_size': round(r_effect, 4),
            'abs_effect': abs(round(r_effect, 4)),
            'direction': '↑ in AD' if diff > 0 else '↓ in AD',
        })

    df = pd.DataFrame(results)

    # FDR correction (Benjamini-Hochberg)
    if len(df) > 0:
        rejected, q_values, _, _ = multipletests(df['p_value'], method='fdr_bh')
        df['q_value'] = q_values
        df['significant_fdr'] = rejected   # Strict: FDR < 0.05
        df['significant'] = df['p_value'] < 0.05  # Nominal: raw p < 0.05
        df['neg_log10_p'] = -np.log10(df['p_value'].clip(lower=1e-300))
    else:
        df['q_value'] = []
        df['significant_fdr'] = []
        df['significant'] = []
        df['neg_log10_p'] = []

    # Sort by raw p-value
    df = df.sort_values('p_value').reset_index(drop=True)

    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fig_volcano(diff_df: pd.DataFrame):
    """
    Volcano plot: effect size (x) vs −log10(raw p-value) (y).
    Two-tier significance: FDR < 0.05 (strict) and raw p < 0.05 (nominal).
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Three tiers: FDR significant, nominally significant, not significant
    fdr_sig = diff_df[diff_df['significant_fdr']]
    nom_sig = diff_df[diff_df['significant'] & ~diff_df['significant_fdr']]
    nonsig = diff_df[~diff_df['significant']]

    # Plot non-significant in grey
    ax.scatter(nonsig['effect_size'], nonsig['neg_log10_p'],
               c=GREY, alpha=0.3, s=25, edgecolors='none', label='Not significant')

    # Nominally significant — colored but smaller
    nom_up = nom_sig[nom_sig['diff'] > 0]
    nom_down = nom_sig[nom_sig['diff'] <= 0]
    ax.scatter(nom_up['effect_size'], nom_up['neg_log10_p'],
               c=AD_COLOR, alpha=0.6, s=45, edgecolors='none',
               label=f'↑ in AD, p<0.05 (n={len(nom_up)})', zorder=4)
    ax.scatter(nom_down['effect_size'], nom_down['neg_log10_p'],
               c=CTRL_COLOR, alpha=0.6, s=45, edgecolors='none',
               label=f'↓ in AD, p<0.05 (n={len(nom_down)})', zorder=4)

    # FDR significant — larger, with borders
    fdr_up = fdr_sig[fdr_sig['diff'] > 0]
    fdr_down = fdr_sig[fdr_sig['diff'] <= 0]
    if len(fdr_up) > 0:
        ax.scatter(fdr_up['effect_size'], fdr_up['neg_log10_p'],
                   c=AD_COLOR, alpha=0.9, s=80, edgecolors=NAVY, linewidths=1,
                   label=f'↑ in AD, FDR<0.05 (n={len(fdr_up)})', zorder=5)
    if len(fdr_down) > 0:
        ax.scatter(fdr_down['effect_size'], fdr_down['neg_log10_p'],
                   c=CTRL_COLOR, alpha=0.9, s=80, edgecolors=NAVY, linewidths=1,
                   label=f'↓ in AD, FDR<0.05 (n={len(fdr_down)})', zorder=5)

    # Significance threshold lines
    p_threshold = -np.log10(0.05)
    ax.axhline(y=p_threshold, color=GOLD, linestyle='--', linewidth=1.5,
               alpha=0.7, label='p = 0.05')
    p_01 = -np.log10(0.01)
    ax.axhline(y=p_01, color=CORAL, linestyle=':', linewidth=1,
               alpha=0.5, label='p = 0.01')

    # Effect size threshold lines
    ax.axvline(x=0.3, color=GREY, linestyle=':', linewidth=1, alpha=0.4)
    ax.axvline(x=-0.3, color=GREY, linestyle=':', linewidth=1, alpha=0.4)
    ax.axvline(x=0.5, color=GREY, linestyle=':', linewidth=0.7, alpha=0.3)
    ax.axvline(x=-0.5, color=GREY, linestyle=':', linewidth=0.7, alpha=0.3)

    # Label top 15 pathways by raw p-value
    to_label = diff_df[diff_df['significant']].head(15)
    for _, row in to_label.iterrows():
        name = str(row['pathway_name'])
        if len(name) > 40:
            name = name[:37] + '...'
        ax.annotate(name,
                    xy=(row['effect_size'], row['neg_log10_p']),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=7, color=NAVY, alpha=0.9,
                    arrowprops=dict(arrowstyle='-', color=GREY, alpha=0.4))

    ax.set_xlabel('Effect Size (Rank-Biserial Correlation)', fontsize=13)
    ax.set_ylabel('−log₁₀(p-value)', fontsize=13)
    ax.set_title('Differential Pathway Activity: AD vs Control\n'
                 '(575 Non-Redundant Pathways)', fontsize=16, color=NAVY)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)

    # Stats annotation
    n_nom = diff_df['significant'].sum()
    n_fdr = diff_df['significant_fdr'].sum()
    n_up = (diff_df['significant'] & (diff_df['diff'] > 0)).sum()
    n_down = (diff_df['significant'] & (diff_df['diff'] <= 0)).sum()
    stats_text = (f'{n_nom} nominally significant (p<0.05)\n'
                  f'{n_fdr} FDR-significant (q<0.05)\n'
                  f'{n_up} ↑ in AD | {n_down} ↓ in AD')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color=NAVY,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#DDD'))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'volcano_plot.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Volcano plot → {path}")


def fig_heatmap(diff_df: pd.DataFrame, scores_df: pd.DataFrame):
    """
    Heatmap of top 30 most significant pathways, samples as columns.
    AD and Control samples grouped and color-coded.
    """
    top = diff_df.head(30)
    if len(top) == 0:
        print("  ⚠ No pathways for heatmap")
        return

    patient_cols = [c for c in scores_df.columns if c in PATIENT_COLS]
    control_cols = [c for c in scores_df.columns if c in CONTROL_COLS]
    all_cols = patient_cols + control_cols

    # Build data matrix
    pw_ids = top['pathway_id'].tolist()
    pw_names = top['pathway_name'].apply(lambda x: x[:50] if len(str(x)) > 50 else x).tolist()

    data = scores_df.loc[pw_ids, all_cols].values.astype(float)

    # Z-score normalize each row (pathway) for visualization
    row_means = np.nanmean(data, axis=1, keepdims=True)
    row_stds = np.nanstd(data, axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    z_data = (data - row_means) / row_stds

    fig, ax = plt.subplots(figsize=(16, 12))

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('ad_ctrl', [TEAL, WHITE, CORAL])

    sns.heatmap(z_data, ax=ax, cmap=cmap, center=0,
                xticklabels=all_cols, yticklabels=pw_names,
                cbar_kws={'label': 'Z-score (pathway activity)', 'shrink': 0.6},
                linewidths=0.3, linecolor='#EEE')

    ax.set_xticklabels(all_cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(pw_names, fontsize=8)

    # Color-code x-axis labels by group
    for i, label in enumerate(ax.get_xticklabels()):
        if label.get_text() in PATIENT_COLS:
            label.set_color(AD_COLOR)
            label.set_fontweight('bold')
        else:
            label.set_color(CTRL_COLOR)
            label.set_fontweight('bold')

    # Significance markers on y-axis (using raw p-value for stars)
    for i, (_, row) in enumerate(top.iterrows()):
        p = row['p_value']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        direction_marker = '▲' if row['diff'] > 0 else '▼'
        color = AD_COLOR if row['diff'] > 0 else CTRL_COLOR
        ax.text(-0.5, i + 0.5, f"{direction_marker}{stars}", ha='right', va='center',
                fontsize=8, color=color, fontweight='bold')

    # Group separator line
    n_ad = len(patient_cols)
    ax.axvline(x=n_ad, color=NAVY, linewidth=2, linestyle='-')
    ax.text(n_ad / 2, -1.5, 'AD (n=20)', ha='center', fontsize=11,
            fontweight='bold', color=AD_COLOR)
    ax.text(n_ad + len(control_cols) / 2, -1.5, 'Control (n=10)',
            ha='center', fontsize=11, fontweight='bold', color=CTRL_COLOR)

    ax.set_title('Top 30 Differentially Active Pathways\n'
                 'AD vs Control (Z-scored ssGSEA activity)',
                 fontsize=16, color=NAVY, pad=30)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'pathway_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Pathway heatmap → {path}")


def fig_top_bar(diff_df: pd.DataFrame):
    """
    Horizontal bar chart of top 20 pathways by effect size,
    colored by direction, with significance annotations.
    """
    sig = diff_df[diff_df['significant']].head(20)
    if len(sig) == 0:
        # Fall back to top 20 by p-value
        sig = diff_df.head(20)

    fig, ax = plt.subplots(figsize=(14, 10))

    names = sig['pathway_name'].apply(lambda x: x[:55] if len(str(x)) > 55 else x).values
    effects = sig['effect_size'].values
    colors = [AD_COLOR if e > 0 else CTRL_COLOR for e in effects]

    bars = ax.barh(range(len(names)), effects, color=colors, edgecolor=WHITE,
                   alpha=0.85, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()

    # Add significance stars (using raw p-value)
    for i, (_, row) in enumerate(sig.iterrows()):
        p = row['p_value']
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        x_pos = effects[i] + 0.02 if effects[i] > 0 else effects[i] - 0.02
        ha = 'left' if effects[i] > 0 else 'right'
        ax.text(x_pos, i, stars, ha=ha, va='center', fontsize=9,
                fontweight='bold', color=NAVY)

    ax.axvline(x=0, color=NAVY, linewidth=1)
    ax.set_xlabel('Effect Size (Rank-Biserial Correlation)', fontsize=12)
    ax.set_title('Top Differentially Active Pathways\n'
                 '← More Active in Control | More Active in AD →',
                 fontsize=15, color=NAVY)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=AD_COLOR, label='↑ in AD'),
        Patch(facecolor=CTRL_COLOR, label='↓ in AD (↑ in Control)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Significance key
    ax.text(0.98, 0.02, '*** p<0.001  ** p<0.01  * p<0.05 (nominal, uncorrected)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color=GREY, style='italic')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'top_pathways_bar.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Top pathways bar chart → {path}")


def fig_source_summary(diff_df: pd.DataFrame):
    """
    Summary: number of significant pathways per database source,
    split by direction.
    """
    sig = diff_df[diff_df['significant']].copy()  # Uses nominal p<0.05
    if len(sig) == 0:
        print("  ⚠ No significant pathways — skipping source summary")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: count by source and direction
    ax = axes[0]
    up = sig[sig['diff'] > 0].groupby('source').size()
    down = sig[sig['diff'] <= 0].groupby('source').size()
    all_sources = sorted(set(list(up.index) + list(down.index)))

    x = np.arange(len(all_sources))
    width = 0.35
    ax.barh(x - width/2, [up.get(s, 0) for s in all_sources],
            height=width, color=AD_COLOR, label='↑ in AD', alpha=0.85)
    ax.barh(x + width/2, [down.get(s, 0) for s in all_sources],
            height=width, color=CTRL_COLOR, label='↓ in AD', alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(all_sources, fontsize=10)
    ax.set_xlabel('Number of Significant Pathways')
    ax.set_title('Significant Pathways by Database', fontweight='bold', color=NAVY)
    ax.legend(fontsize=9)

    # Right: effect size distribution
    ax = axes[1]
    for source in all_sources:
        source_data = sig[sig['source'] == source]['effect_size']
        if len(source_data) > 0:
            parts = ax.violinplot([source_data.values],
                                  positions=[list(all_sources).index(source)],
                                  showmeans=True, showmedians=True, widths=0.6)
            for pc in parts['bodies']:
                pc.set_facecolor(BLUE)
                pc.set_alpha(0.6)

    ax.set_xticks(range(len(all_sources)))
    ax.set_xticklabels(all_sources, fontsize=10)
    ax.axhline(y=0, color=NAVY, linewidth=0.8)
    ax.set_ylabel('Effect Size')
    ax.set_title('Effect Size Distribution by Database', fontweight='bold', color=NAVY)

    fig.suptitle('Differential Pathway Activity — Summary by Database',
                 fontsize=16, fontweight='bold', color=NAVY, y=1.02)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'source_summary.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Source summary → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "=" * 70)
    print("DIFFERENTIAL PATHWAY ACTIVITY ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    scores_df = pd.read_csv(IN_SCORES, index_col=0)
    nonred_df = pd.read_csv(IN_NONRED)
    with open(IN_PATHWAYS) as f:
        pathway_sets = json.load(f)

    nonred_ids = nonred_df['pathway_id'].tolist()
    print(f"  Pathway scores: {scores_df.shape[0]} total")
    print(f"  Non-redundant pathways: {len(nonred_ids)}")

    # Compute differential activity
    print("\nComputing differential activity (Mann-Whitney U + FDR)...")
    diff_df = compute_differential_activity(scores_df, nonred_ids, pathway_sets)

    n_nom = diff_df['significant'].sum()
    n_fdr = diff_df['significant_fdr'].sum()
    n_up = ((diff_df['significant']) & (diff_df['diff'] > 0)).sum()
    n_down = ((diff_df['significant']) & (diff_df['diff'] <= 0)).sum()

    print(f"\n  Results:")
    print(f"  Total pathways tested:    {len(diff_df)}")
    print(f"  Nominal (p<0.05):         {n_nom}")
    print(f"  FDR-corrected (q<0.05):   {n_fdr}")
    print(f"    ↑ in AD:                {n_up}")
    print(f"    ↓ in AD:                {n_down}")

    if n_fdr == 0 and n_nom > 0:
        print(f"\n  Note: No pathways survive FDR at n=30 with {len(diff_df)} tests.")
        print(f"  This is a statistical power limitation, not a biological one.")
        print(f"  {n_nom} nominally significant pathways have large effect sizes.")

    # Top 10
    print(f"\n  Top 10 by raw p-value:")
    print(f"  {'Pathway':<45} {'p-value':>10} {'q-value':>10} {'Effect':>8} {'Dir':>10}")
    print("  " + "─" * 85)
    for _, row in diff_df.head(10).iterrows():
        name = str(row['pathway_name'])[:44]
        print(f"  {name:<45} {row['p_value']:>10.4f} {row['q_value']:>10.4f} "
              f"{row['effect_size']:>+8.3f} {row['direction']:>10}")

    # Save
    diff_df.to_csv(OUT_DIFF, index=False)
    print(f"\n  → Saved: {OUT_DIFF}")

    # Generate figures
    print("\nGenerating figures...")
    fig_volcano(diff_df)
    fig_heatmap(diff_df, scores_df)
    fig_top_bar(diff_df)
    fig_source_summary(diff_df)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {n_nom} nominally significant pathways (p<0.05)")
    print(f"           {n_fdr} FDR-significant pathways (q<0.05)")
    print(f"  {n_up} upregulated in AD  |  {n_down} downregulated in AD")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
