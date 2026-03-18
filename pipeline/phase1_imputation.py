"""
Phase 1.5 – Missing Data Imputation (Tiered)

Strategy (literature-grounded):
  Tier 1: ≤50% missing → MissForest (IterativeImputer + RandomForest)
  Tier 2: 50–80% missing → QRILC (left-censored, MNAR-specific)
  Tier 3: >80% missing → Keep as-is (too sparse to impute)

References:
  - Webb-Robertson et al. (2023) J Proteome Res — MissForest top-ranked
  - Lazar et al. (2016) J Proteome Res — QRILC + mixed MNAR/MAR framework
  - Stekhoven & Bühlmann (2012) Bioinformatics — Original MissForest
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from pipeline.config import OUTPUT_DIR, SAMPLE_COLS

# ── Output directory ──────────────────────────────────────────────────────
IMPUTED_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "output_imputed")
os.makedirs(IMPUTED_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(IMPUTED_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Tier thresholds ───────────────────────────────────────────────────────
TIER1_MAX_MISS = 0.50   # ≤50% → MissForest
TIER2_MAX_MISS = 0.80   # 50–80% → QRILC
# >80% → no imputation

RANDOM_STATE = 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QRILC — Quantile Regression Imputation of Left-Censored data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def qrilc_impute(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    """
    QRILC imputation for a single protein (column).

    Fits a truncated normal distribution to the left tail of observed values,
    then draws missing values from that distribution.

    Logic:
      - mean_imp = Q(0.01) of observed values (1st percentile)
      - sd_imp   = sd(observed) * 0.3 (narrower spread)
      - Draw from TruncNorm(mean_imp, sd_imp) bounded by [global_min - 1, Q(0.05)]
    """
    observed = series.dropna()
    if len(observed) < 3:
        return series  # too few observations to estimate

    n_missing = series.isna().sum()
    if n_missing == 0:
        return series

    obs_std = observed.std()
    if obs_std == 0 or np.isnan(obs_std):
        obs_std = 0.5  # fallback

    # Left-tail parameters
    mean_imp = observed.quantile(0.01)
    sd_imp = obs_std * 0.3

    # Bounds: clip to reasonable range
    lower = observed.min() - 2.0  # slightly below minimum observed
    upper = observed.quantile(0.05)
    if upper <= mean_imp:
        upper = mean_imp + sd_imp

    # Convert to standard truncnorm parameters
    a = (lower - mean_imp) / sd_imp
    b = (upper - mean_imp) / sd_imp

    imputed_vals = truncnorm.rvs(a, b, loc=mean_imp, scale=sd_imp,
                                  size=n_missing,
                                  random_state=rng.integers(0, 2**31))

    result = series.copy()
    result[result.isna()] = imputed_vals
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MissForest — IterativeImputer with RandomForest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def missforest_impute(data: pd.DataFrame) -> pd.DataFrame:
    """
    MissForest imputation using sklearn's IterativeImputer with
    RandomForestRegressor as the estimator.

    Good for MAR data where relationships between proteins can be leveraged.
    """
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    imputer = IterativeImputer(
        estimator=rf,
        max_iter=10,
        random_state=RANDOM_STATE,
        initial_strategy="median",
        skip_complete=True,
    )

    imputed_array = imputer.fit_transform(data.values)
    return pd.DataFrame(imputed_array, index=data.index, columns=data.columns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Diagnostics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_diagnostics(original_df, imputed_df, sample_cols, tier_labels, save_path):
    """Generate diagnostic plots comparing pre- and post-imputation distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Imputation Diagnostics", fontsize=15, fontweight="bold")

    # Plot 1: Overall distribution
    ax = axes[0, 0]
    orig_vals = original_df[sample_cols].values.flatten()
    orig_vals = orig_vals[~np.isnan(orig_vals)]
    imp_vals = imputed_df[sample_cols].values.flatten()
    imp_vals = imp_vals[~np.isnan(imp_vals)]
    ax.hist(orig_vals, bins=60, alpha=0.6, label="Original (observed)", color="#2196F3", density=True)
    ax.hist(imp_vals, bins=60, alpha=0.4, label="After imputation", color="#FF5722", density=True)
    ax.set_xlabel("log₂ intensity")
    ax.set_ylabel("Density")
    ax.set_title("Overall Distribution")
    ax.legend()

    # Plot 2: Tier 1 proteins (MissForest)
    ax = axes[0, 1]
    t1_prots = [p for p, t in tier_labels.items() if t == "Tier1"]
    if t1_prots:
        orig_t1 = original_df.loc[t1_prots, sample_cols].values.flatten()
        orig_t1 = orig_t1[~np.isnan(orig_t1)]
        imp_t1 = imputed_df.loc[t1_prots, sample_cols].values.flatten()
        imp_t1 = imp_t1[~np.isnan(imp_t1)]
        ax.hist(orig_t1, bins=40, alpha=0.6, label="Original", color="#2196F3", density=True)
        ax.hist(imp_t1, bins=40, alpha=0.4, label="MissForest", color="#4CAF50", density=True)
    ax.set_title(f"Tier 1: MissForest (n={len(t1_prots)})")
    ax.set_xlabel("log₂ intensity")
    ax.legend()

    # Plot 3: Tier 2 proteins (QRILC)
    ax = axes[1, 0]
    t2_prots = [p for p, t in tier_labels.items() if t == "Tier2"]
    if t2_prots:
        orig_t2 = original_df.loc[t2_prots, sample_cols].values.flatten()
        orig_t2 = orig_t2[~np.isnan(orig_t2)]
        imp_t2 = imputed_df.loc[t2_prots, sample_cols].values.flatten()
        imp_t2 = imp_t2[~np.isnan(imp_t2)]
        ax.hist(orig_t2, bins=40, alpha=0.6, label="Original", color="#2196F3", density=True)
        ax.hist(imp_t2, bins=40, alpha=0.4, label="QRILC", color="#FF9800", density=True)
    ax.set_title(f"Tier 2: QRILC (n={len(t2_prots)})")
    ax.set_xlabel("log₂ intensity")
    ax.legend()

    # Plot 4: Summary bar chart
    ax = axes[1, 1]
    t1_count = len([p for p, t in tier_labels.items() if t in ("Tier1", "Complete")])
    t2_count = len(t2_prots)
    t3_count = len([p for p, t in tier_labels.items() if t == "Tier3"])
    tier_names = ["Tier 1\n(MissForest)", "Tier 2\n(QRILC)", "Tier 3\n(No impute)"]
    tier_vals = [t1_count, t2_count, t3_count]
    bars = ax.bar(tier_names, tier_vals,
                  color=["#4CAF50", "#FF9800", "#9E9E9E"], edgecolor="white")
    for bar, count in zip(bars, tier_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha="center", fontweight="bold")
    ax.set_ylabel("Number of proteins")
    ax.set_title("Proteins per Tier")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved diagnostics → {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_imputation():
    print("\n" + "=" * 70)
    print("PHASE 1.5 — Missing Data Imputation (Tiered)")
    print("=" * 70)

    # ── 1. Load canonical log2 data ───────────────────────────────────────
    print("\n[1/5] Loading canonical_log2.csv...")
    canonical_path = os.path.join(OUTPUT_DIR, "canonical_log2.csv")
    df = pd.read_csv(canonical_path, index_col=0)

    sample_cols = [c for c in df.columns if c in SAMPLE_COLS]
    print(f"  Shape: {df.shape[0]} proteins × {len(sample_cols)} samples")

    # ── 2. Classify into tiers ────────────────────────────────────────────
    print("\n[2/5] Classifying proteins into imputation tiers...")
    miss_rate = df[sample_cols].isna().mean(axis=1)

    tier_labels = {}
    for prot_idx in df.index:
        mr = miss_rate[prot_idx]
        if mr == 0:
            tier_labels[prot_idx] = "Complete"
        elif mr <= TIER1_MAX_MISS:
            tier_labels[prot_idx] = "Tier1"
        elif mr <= TIER2_MAX_MISS:
            tier_labels[prot_idx] = "Tier2"
        else:
            tier_labels[prot_idx] = "Tier3"

    tier_counts = pd.Series(tier_labels).value_counts()
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} proteins")

    # ── 3. Apply MissForest to Tier 1 proteins ────────────────────────────
    print("\n[3/5] Applying MissForest to Tier 1 proteins (≤50% missing)...")
    tier1_idx = [i for i, t in tier_labels.items() if t in ("Tier1", "Complete")]
    tier1_data = df.loc[tier1_idx, sample_cols].copy()

    # Transpose: samples as rows, proteins as columns for IterativeImputer
    tier1_T = tier1_data.T
    n_missing_t1 = tier1_T.isna().sum().sum()
    print(f"  Tier 1 proteins: {len(tier1_idx)} ({n_missing_t1} missing values)")

    if n_missing_t1 > 0:
        tier1_imputed_T = missforest_impute(tier1_T)
        tier1_imputed = tier1_imputed_T.T
        print(f"  MissForest complete: {n_missing_t1} values imputed")
    else:
        tier1_imputed = tier1_data
        print("  No missing values in Tier 1 — skipped")

    # ── 4. Apply QRILC to Tier 2 proteins ─────────────────────────────────
    print("\n[4/5] Applying QRILC to Tier 2 proteins (50–80% missing)...")
    tier2_idx = [i for i, t in tier_labels.items() if t == "Tier2"]
    n_missing_t2 = df.loc[tier2_idx, sample_cols].isna().sum().sum()
    print(f"  Tier 2 proteins: {len(tier2_idx)} ({n_missing_t2} missing values)")

    rng = np.random.default_rng(RANDOM_STATE)
    tier2_imputed = df.loc[tier2_idx, sample_cols].copy()
    for prot_idx in tier2_idx:
        tier2_imputed.loc[prot_idx] = qrilc_impute(
            tier2_imputed.loc[prot_idx], rng
        )
    print(f"  QRILC complete: {n_missing_t2} values imputed")

    # ── 5. Assemble final imputed dataset ─────────────────────────────────
    print("\n[5/5] Assembling imputed dataset...")
    imputed_df = df.copy()

    # Replace Tier 1 + Complete
    for idx in tier1_idx:
        imputed_df.loc[idx, sample_cols] = tier1_imputed.loc[idx, sample_cols]

    # Replace Tier 2
    for idx in tier2_idx:
        imputed_df.loc[idx, sample_cols] = tier2_imputed.loc[idx, sample_cols]

    # Tier 3 stays unchanged (NaN remains)
    tier3_idx = [i for i, t in tier_labels.items() if t == "Tier3"]

    # Sanity checks
    total_orig_missing = df[sample_cols].isna().sum().sum()
    total_new_missing = imputed_df[sample_cols].isna().sum().sum()
    total_imputed = total_orig_missing - total_new_missing
    tier3_missing = df.loc[tier3_idx, sample_cols].isna().sum().sum()

    print(f"\n  ── Summary ──")
    print(f"  Original missing values: {total_orig_missing}")
    print(f"  Imputed (Tier 1 + 2):    {total_imputed}")
    print(f"  Remaining (Tier 3):      {total_new_missing}")
    print(f"  Tier 3 missing (verify): {tier3_missing}")

    # ── Save outputs ──────────────────────────────────────────────────────
    out_csv = os.path.join(IMPUTED_DIR, "canonical_log2.csv")
    imputed_df.to_csv(out_csv)
    print(f"\n  Saved imputed data → {out_csv}")

    # Save report
    report = {
        "total_proteins": len(df),
        "tier1_count": len([i for i in tier_labels.values() if i in ("Tier1", "Complete")]),
        "tier2_count": len(tier2_idx),
        "tier3_count": len(tier3_idx),
        "tier1_method": "MissForest (IterativeImputer + RandomForest)",
        "tier2_method": "QRILC (left-censored truncated normal)",
        "tier3_method": "No imputation (>80% missing)",
        "total_orig_missing": int(total_orig_missing),
        "total_imputed": int(total_imputed),
        "total_remaining_missing": int(total_new_missing),
        "tier_thresholds": {"tier1_max": TIER1_MAX_MISS, "tier2_max": TIER2_MAX_MISS},
    }
    report_path = os.path.join(IMPUTED_DIR, "imputation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report → {report_path}")

    # Diagnostics
    plot_diagnostics(df, imputed_df, sample_cols, tier_labels,
                     os.path.join(FIGURES_DIR, "imputation_diagnostics.png"))

    # Copy other files needed by downstream phases
    import shutil
    for fname in ["pathway_gene_sets.json", "gene_symbol_map.csv", "pathway_summary.csv"]:
        src = os.path.join(OUTPUT_DIR, fname)
        dst = os.path.join(IMPUTED_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print(f"  Copied pathway files to {IMPUTED_DIR}")

    return imputed_df, report


if __name__ == "__main__":
    run_imputation()
