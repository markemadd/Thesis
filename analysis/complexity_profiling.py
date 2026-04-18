"""
Task 4 — Computational Complexity Analysis + Runtime Profiling

Instruments each pipeline phase with time.perf_counter() and tracemalloc.
Documents theoretical Big-O complexity in comments.
Produces:
  - output/complexity_profiling.csv       (wall time + peak memory per phase)
  - output/runtime_profile.png            (horizontal bar chart, 300 DPI)

Usage:
    python -m analysis.complexity_profiling

Design notes:
  - Uses pre-computed output files where possible to avoid API calls.
  - Phase 1 is measured via direct profiling (loads data, simulates imputation).
  - All measurements use time.perf_counter() (monotonic, higher resolution than time.time()).
  - Peak memory uses tracemalloc (tracks Python heap allocations only).
"""

import os
import json
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Union

warnings.filterwarnings('ignore')

from pipeline.config import (
    OUTPUT_DIR, SAMPLE_COLS, PATIENT_COLS, CONTROL_COLS, RANDOM_STATE,
    STABILITY_N_BOOTSTRAP, STABILITY_THRESHOLD,
    ELASTIC_NET_L1_RATIOS, ELASTIC_NET_CV_FOLDS,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
IN_CANONICAL   = os.path.join(OUTPUT_DIR, "canonical_log2.csv")
IN_PATHWAYS    = os.path.join(OUTPUT_DIR, "pathway_gene_sets.json")
IN_SCORES      = os.path.join(OUTPUT_DIR, "pathway_scores.csv")
IN_NONRED      = os.path.join(OUTPUT_DIR, "nonredundant_pathways.csv")
IN_STABLE      = os.path.join(OUTPUT_DIR, "stable_pathways.csv")
OUT_CSV        = os.path.join(OUTPUT_DIR, "complexity_profiling.csv")
OUT_PNG        = os.path.join(OUTPUT_DIR, "runtime_profile.png")

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
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#DDDDDD',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROFILING CONTEXT MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@contextmanager
def profile_phase(name: str, results: list):
    """
    Context manager that measures wall-clock time and peak memory for a code block.

    Usage:
        with profile_phase("Phase 3 ssGSEA", results):
            run_phase3()

    Appends a dict with keys: phase, wall_time_s, peak_mem_mb to `results`.
    """
    tracemalloc.start()
    t_start = time.perf_counter()
    try:
        yield
    finally:
        t_end = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        wall_time = t_end - t_start
        peak_mb = peak_mem / (1024 * 1024)

        results.append({
            'phase': name,
            'wall_time_s': round(wall_time, 3),
            'peak_mem_mb': round(peak_mb, 1),
        })
        print(f"  [{name}] {wall_time:.2f}s | peak {peak_mb:.1f} MB")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PHASE PROFILES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def profile_phase1_load(results: list):
    """
    Phase 1: Preprocessing + Imputation
    Big-O: O(N * p * k) where N=samples, p=proteins, k=KNN neighbors (k=5)
    Dominant term: KNN imputation across the sparse protein matrix.
    Here we profile the data loading + imputation simulation on canonical data
    (full phase1 requires raw input CSV; we estimate from canonical reload).
    """
    # Big-O: O(N × P) for detection-rate filtering; O(N × P × k) KNN imputation
    # N = 30 samples, P = ~3,000 proteins (CSF proteome), k = 5

    with profile_phase("Phase 1: Preprocessing + Imputation  [O(N·P·k), N=30,P≈3k,k=5]", results):
        # Simulate the dominant I/O + data structure operations of phase 1
        data_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "Documentation")
        raw_csv  = os.path.join(data_dir, "normalized df-1.csv")
        # Profile the dominant I/O + imputation cost using canonical data as proxy.
        # We avoid re-running run_imputation() to prevent overwriting output files.
        _ = pd.read_csv(IN_CANONICAL)
        # Simulate KNN imputation: the dominant cost of Phase 1
        X = _.select_dtypes(include=[np.number]).values
        # Introduce synthetic NAs to give KNNImputer realistic work
        rng_local = np.random.RandomState(0)
        X_sparse = X.copy().astype(float)
        na_mask = rng_local.rand(*X_sparse.shape) < 0.25  # 25% missingness
        X_sparse[na_mask] = np.nan
        from sklearn.impute import KNNImputer
        imp = KNNImputer(n_neighbors=5)
        _ = imp.fit_transform(X_sparse)


def profile_phase2_load(results: list):
    """
    Phase 2: Pathway Mapping (Gene Symbol → Pathway Gene Sets)
    Big-O: O(P × G) where P=pathways (~3,000), G=genes per pathway (avg ~50)
    Dominant term: JSON parsing + gene set construction from pre-built files.
    """
    with profile_phase("Phase 2: Pathway Mapping            [O(P·G), P≈3k,G_avg≈50]", results):
        # Profile: load and validate the gene set JSON
        with open(IN_PATHWAYS) as f:
            gateway_sets = json.load(f)

        # Build gene → pathway index (typical phase2 operation)
        gene_to_pathways: dict = {}
        for pid, info in gateway_sets.items():
            for gene in info.get('genes', []):
                gene_to_pathways.setdefault(gene, []).append(pid)


def profile_phase3_scoring(results: list):
    """
    Phase 3: Weighted ssGSEA Pathway Scoring
    Big-O: O(S × P × N·log N) where S=samples(30), P=pathways(2585), N=proteins per sample
    Dominant term: sort + cumsum per (sample, pathway) pair.
    Per-pathway: rank sort O(N log N) + walk O(N) = O(N log N).
    Total: O(S × P × N log N).
    """
    with profile_phase("Phase 3: ssGSEA Scoring             [O(S·P·N·logN), S=30,P=2585,N≈3k]", results):
        from pipeline.phase3_pathway_scoring import compute_pathway_scores
        canonical = pd.read_csv(IN_CANONICAL)
        with open(IN_PATHWAYS) as f:
            pathway_sets = json.load(f)

        # Use current combined weights (α=β=1)
        curated_ad_genes = {
            'APOE': 0.95, 'CLU': 0.85, 'BIN1': 0.80, 'PICALM': 0.75,
            'APP': 0.95, 'MAPT': 0.90, 'PSEN1': 0.90, 'PSEN2': 0.85,
            'TREM2': 0.80, 'NEFL': 0.75, 'CST3': 0.65, 'CHI3L1': 0.65,
        }
        all_genes = canonical['gene_symbol'].dropna().unique().tolist()
        protein_weights = {g: 1.0 + curated_ad_genes.get(g, 0.3) for g in all_genes}

        _ = compute_pathway_scores(canonical, pathway_sets, protein_weights)


def profile_stage1(results: list, scores_df, pathway_sets):
    """
    Stage 1: De-Redundancy (Jaccard + Pearson correlation clustering)
    Big-O: O(P² × G) Jaccard + O(P² × S) correlation + O(P² log P) clustering
    where P=pathways(2585), G=avg genes, S=samples(30).
    Dominant: Jaccard matrix construction O(P²·G).
    """
    with profile_phase("Stage 1: De-Redundancy              [O(P²·G+P²·S), P=2585,G_avg≈50,S=30]", results):
        from pipeline.phase5_dim_reduction import stage1_deredundancy
        _, _ = stage1_deredundancy(scores_df, pathway_sets)


def profile_stage2(results: list, scores_df, nonred_ids):
    """
    Stage 2: IQR-Based Variance Filtering
    Big-O: O(P' × S) where P'=non-redundant pathways (~575), S=samples(30).
    Dominant: compute IQR per pathway across samples.
    """
    with profile_phase("Stage 2: Variance Filtering         [O(P'·S), P'=575,S=30]", results):
        from pipeline.phase5_dim_reduction import stage2_variance_filter
        _ = stage2_variance_filter(scores_df, nonred_ids)


def profile_stage3(results: list, scores_df, filtered_ids):
    """
    Stage 3: Elastic Net + Stability Selection (B=200 bootstraps)
    Big-O: O(B × S × F × C) where B=bootstraps(200), S=samples(30),
    F=pathways after filtering(~473), C=CV folds(5) × ElasticNet iterations.
    Dominant: SAGA solver for LogisticRegressionCV, O(max_iter × n × p) per fold.
    Total: O(B × C × max_iter × S × F).
    """
    with profile_phase("Stage 3: Stability Selection        [O(B·C·iter·S·F), B=200,C=5,S=30,F≈473]", results):
        from pipeline.phase5_dim_reduction import stage3_stability_selection
        _ = stage3_stability_selection(scores_df, filtered_ids)


def profile_stage4_loocv(results: list, scores_df, stable_df):
    """
    Stage 4a: Standard LOOCV (Logistic Regression)
    Big-O: O(n² × p) where n=samples(30), p=stable pathways(13).
    Per fold: train on (n-1) samples over p features → O(n × p) for LBFGS.
    Total n folds: O(n² × p).
    """
    with profile_phase("Stage 4a: Standard LOOCV            [O(n²·p), n=30,p=13]", results):
        from pipeline.phase5_dim_reduction import loocv_evaluate, build_feature_matrix
        stable_ids = stable_df['pathway_id'].tolist()
        X, y, _ = build_feature_matrix(scores_df, stable_ids)
        _ = loocv_evaluate(X, y, 'logistic')


def profile_stage4_nested(results: list, scores_df, filtered_ids):
    """
    Stage 4b: Nested LOOCV (stability selection embedded per fold)
    Big-O: O(n × B_inner × C × iter × (n-1) × F + n² × p)
    where n=30, B_inner=100, C=5 inner CV folds, F≈473 features, p=selected.
    Dominant: inner stability selection × n outer folds.
    This is the most expensive operation in the entire pipeline.
    """
    with profile_phase("Stage 4b: Nested LOOCV              [O(n·B·C·S·F), n=30,B=100,C=5,F≈473]", results):
        from pipeline.phase5_dim_reduction import nested_loocv
        _ = nested_loocv(scores_df, filtered_ids)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BIG-O COMPLEXITY TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BIG_O_MAP = {
    "Phase 1: Preprocessing + Imputation  [O(N·P·k), N=30,P≈3k,k=5]":
        "O(N·P·k) — KNN imputation; N=samples, P=proteins, k=5 neighbours",
    "Phase 2: Pathway Mapping            [O(P·G), P≈3k,G_avg≈50]":
        "O(P·G) — gene-set index construction; P=pathways, G=avg genes/pathway",
    "Phase 3: ssGSEA Scoring             [O(S·P·N·logN), S=30,P=2585,N≈3k]":
        "O(S·P·N·log N) — rank-sort + cumsum per (sample,pathway); S=samples",
    "Stage 1: De-Redundancy              [O(P²·G+P²·S), P=2585,G_avg≈50,S=30]":
        "O(P²·G + P²·S) — Jaccard matrix + score correlation; P=pathways",
    "Stage 2: Variance Filtering         [O(P'·S), P'=575,S=30]":
        "O(P'·S) — IQR per pathway; P'=non-redundant pathways, S=samples",
    "Stage 3: Stability Selection        [O(B·C·iter·S·F), B=200,C=5,S=30,F≈473]":
        "O(B·C·iter·S·F) — SAGA Elastic Net × B bootstrap × C inner folds",
    "Stage 4a: Standard LOOCV            [O(n²·p), n=30,p=13]":
        "O(n²·p) — LBFGS logistic regression × n LOOCV folds",
    "Stage 4b: Nested LOOCV              [O(n·B·C·S·F), n=30,B=100,C=5,F≈473]":
        "O(n·B_inner·C·iter·S·F) — inner stability selection per LOOCV fold",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OUTPUT: CSV + PLOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_csv(results: list, out_path: str):
    """Save profiling results to CSV with Big-O annotations."""
    rows = []
    for r in results:
        rows.append({
            'phase': r['phase'].split('[')[0].strip(),
            'wall_time_s': r['wall_time_s'],
            'peak_mem_mb': r['peak_mem_mb'],
            'big_o': BIG_O_MAP.get(r['phase'], '—'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  ✓ CSV → {out_path}")
    return df


def plot_runtime_profile(results: list, out_path: str):
    """
    Horizontal bar chart of wall-clock time per phase (300 DPI).
    Colour-coded by pipeline stage, with peak memory annotated.
    """
    labels = [r['phase'].split('[')[0].strip() for r in results]
    times  = [r['wall_time_s'] for r in results]
    mems   = [r['peak_mem_mb'] for r in results]
    n = len(results)

    # Colour palette: gradient from teal → coral (early → late stages)
    colours = [
        TEAL, TEAL,         # Phase 1-2
        BLUE,               # Phase 3
        GOLD, GOLD,         # Stage 1-2
        '#9B59B6', '#9B59B6',  # Stage 3-4a (purple)
        CORAL,              # Stage 4b (most expensive)
    ][:n]

    fig, ax = plt.subplots(figsize=(13, max(6, n * 0.85)))
    fig.patch.set_facecolor(WHITE)

    bars = ax.barh(
        range(n), times, color=colours, edgecolor=WHITE,
        linewidth=0.6, height=0.68, alpha=0.88,
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    # Annotate bars with time + memory
    for i, (bar, t, m) in enumerate(zip(bars, times, mems)):
        x_pos = t + max(times) * 0.01
        label_txt = f"{t:.2f}s  |  {m:.0f} MB"
        ax.text(x_pos, i, label_txt,
                va='center', ha='left', fontsize=9, color=NAVY)

    # Log-scale x-axis if range is large
    if max(times) / (min(times) + 1e-6) > 50:
        ax.set_xscale('log')
        ax.set_xlabel('Wall-Clock Time (seconds, log scale)', fontsize=12)
    else:
        ax.set_xlabel('Wall-Clock Time (seconds)', fontsize=12)

    ax.set_title(
        'AD Proteomics Pipeline — Runtime & Memory Profile\n'
        '(Wall-clock time per phase, labels show time | peak memory)',
        fontsize=14, color=NAVY, pad=12
    )

    # Total runtime annotation
    total = sum(times)
    ax.text(0.98, 0.02,
            f'Total: {total:.1f}s ({total/60:.1f} min)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, color=NAVY, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4FF',
                      edgecolor=BLUE, alpha=0.92))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=WHITE, edgecolor='none')
    plt.close()
    print(f"  ✓ Runtime chart → {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("\n" + "=" * 75)
    print("  TASK 4 — Computational Complexity Analysis + Runtime Profiling")
    print("=" * 75)
    print("  Using pre-computed output files where possible. All phases")
    print("  profiled with time.perf_counter() + tracemalloc.")
    print()

    results: list[dict] = []

    # ── Load shared data once ──────────────────────────────────────────────
    print("  Loading shared data...")
    scores_df = pd.read_csv(IN_SCORES, index_col=0)
    with open(IN_PATHWAYS) as f:
        pathway_sets = json.load(f)
    nonred_df   = pd.read_csv(IN_NONRED)
    stable_df   = pd.read_csv(IN_STABLE)
    nonred_ids  = nonred_df['pathway_id'].tolist()
    filtered_ids = nonred_ids  # proxy (stage2 would further filter these)

    print(f"  Scores: {scores_df.shape}, Non-redundant: {len(nonred_ids)}, "
          f"Stable: {len(stable_df)}\n")

    # ── Profile each phase ─────────────────────────────────────────────────
    print("[Profiling pipeline phases — this runs the actual computations]\n")

    profile_phase1_load(results)
    profile_phase2_load(results)
    profile_phase3_scoring(results)
    profile_stage1(results, scores_df, pathway_sets)
    profile_stage2(results, scores_df, nonred_ids)
    profile_stage3(results, scores_df, filtered_ids)
    profile_stage4_loocv(results, scores_df, stable_df)
    profile_stage4_nested(results, scores_df, filtered_ids)

    # ── Print summary ──────────────────────────────────────────────────────
    total_time = sum(r['wall_time_s'] for r in results)
    total_mem  = max(r['peak_mem_mb'] for r in results)

    print("\n" + "=" * 75)
    print("  PROFILING SUMMARY")
    print("=" * 75)
    print(f"  {'Phase':<50} {'Time (s)':>9} {'Peak MB':>9}")
    print("  " + "─" * 70)
    for r in results:
        short = r['phase'].split('[')[0].strip()[:50]
        print(f"  {short:<50} {r['wall_time_s']:>9.2f} {r['peak_mem_mb']:>9.1f}")
    print("  " + "─" * 70)
    print(f"  {'TOTAL':<50} {total_time:>9.2f} {total_mem:>9.1f}")
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Peak memory:   {total_mem:.1f} MB\n")

    print("\n  BIG-O COMPLEXITY NOTES:")
    for phase_key, complexity in BIG_O_MAP.items():
        short = phase_key.split('[')[0].strip()
        print(f"  {short}:")
        print(f"    → {complexity}")

    print("=" * 75)

    # ── Save outputs ───────────────────────────────────────────────────────
    print("\n[Saving outputs...]")
    df = save_csv(results, OUT_CSV)
    plot_runtime_profile(results, OUT_PNG)
    print("\n  Done.")


if __name__ == '__main__':
    main()
