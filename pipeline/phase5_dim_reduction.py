"""
Phase 5 – Pathway Dimensionality Reduction

A 4-stage framework for reducing ~2,500 pathway activity scores to a compact,
stable, biologically interpretable set suitable for downstream classification.

Stages:
  1. De-redundancy   — Jaccard + score-correlation clustering (ProMS-inspired)
  2. Variance filter  — IQR-based removal of uninformative pathways
  3. Stability selection — Elastic Net on 200 bootstrap subsamples
  4. Validation       — LOOCV (Logistic Regression + Random Forest) + permutation test

Designed for small-n (n ≈ 30), high-p sparse proteomics datasets.
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score

from pipeline.config import (
    OUTPUT_DIR, SAMPLE_COLS, PATIENT_COLS, CONTROL_COLS,
    RANDOM_STATE,
    JACCARD_THRESHOLD, JACCARD_SENSITIVITY, SCORE_CORR_WEIGHT,
    IQR_PERCENTILE_CUTOFF,
    STABILITY_N_BOOTSTRAP, STABILITY_THRESHOLD,
    ELASTIC_NET_L1_RATIOS, ELASTIC_NET_CV_FOLDS,
    PERMUTATION_N,
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── File paths ────────────────────────────────────────────────────────────────
IN_SCORES     = os.path.join(OUTPUT_DIR, "pathway_scores.csv")
IN_PATHWAYS   = os.path.join(OUTPUT_DIR, "pathway_gene_sets.json")
OUT_NONRED    = os.path.join(OUTPUT_DIR, "nonredundant_pathways.csv")
OUT_STABLE    = os.path.join(OUTPUT_DIR, "stable_pathways.csv")
OUT_REPORT    = os.path.join(OUTPUT_DIR, "reduction_report.json")
OUT_VALID     = os.path.join(OUTPUT_DIR, "validation_results.csv")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 1: De-Redundancy (Jaccard + Score Correlation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_jaccard_matrix(pathway_sets: dict) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise Jaccard similarity matrix for pathway gene sets."""
    pw_ids = list(pathway_sets.keys())
    n = len(pw_ids)

    # Pre-compute gene sets
    gene_sets = [set(pathway_sets[pid]['genes']) for pid in pw_ids]

    # Build upper-triangular Jaccard matrix
    jaccard = np.zeros((n, n))
    for i in range(n):
        jaccard[i, i] = 1.0
        for j in range(i + 1, n):
            intersection = len(gene_sets[i] & gene_sets[j])
            union = len(gene_sets[i] | gene_sets[j])
            if union > 0:
                jaccard[i, j] = intersection / union
                jaccard[j, i] = jaccard[i, j]

    return jaccard, pw_ids


def compute_combined_similarity(jaccard: np.ndarray,
                                 scores_df: pd.DataFrame,
                                 pw_ids: list[str],
                                 use_corr: bool = True) -> np.ndarray:
    """
    Combine Jaccard (structural) and score correlation (functional) similarity.
    Returns max(Jaccard, |Pearson_r|) for each pair.
    """
    if not use_corr:
        return jaccard

    sample_cols = [c for c in scores_df.columns if c in SAMPLE_COLS]
    n = len(pw_ids)

    # Compute Pearson correlation of score profiles
    score_matrix = scores_df.loc[pw_ids, sample_cols].values.astype(float)
    # Handle any NaN by filling with row means
    row_means = np.nanmean(score_matrix, axis=1, keepdims=True)
    nan_mask = np.isnan(score_matrix)
    score_matrix[nan_mask] = np.broadcast_to(row_means, score_matrix.shape)[nan_mask]

    # Correlation matrix
    score_corr = np.corrcoef(score_matrix)
    score_corr = np.nan_to_num(score_corr, nan=0.0)

    # Combined: max(Jaccard, |correlation|)
    combined = np.maximum(jaccard, np.abs(score_corr))

    return combined


def cluster_and_select_representatives(similarity: np.ndarray,
                                        scores_df: pd.DataFrame,
                                        pw_ids: list[str],
                                        pathway_sets: dict,
                                        threshold: float) -> list[str]:
    """
    Hierarchical clustering on similarity matrix.
    Pick one representative per cluster (highest IQR in score profile).
    """
    n = len(pw_ids)
    sample_cols = [c for c in scores_df.columns if c in SAMPLE_COLS]

    # Convert similarity to distance
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    # Ensure symmetry and non-negativity
    distance = np.maximum(distance, 0.0)
    distance = (distance + distance.T) / 2.0

    # Condensed distance for linkage
    condensed = squareform(distance, checks=False)

    # Hierarchical clustering (average linkage)
    Z = linkage(condensed, method='average')

    # Cut at threshold: distance < (1 - threshold) means similarity > threshold
    cluster_labels = fcluster(Z, t=1.0 - threshold, criterion='distance')

    # For each cluster, pick the representative with highest score IQR
    cluster_to_pws = defaultdict(list)
    for idx, cid in enumerate(cluster_labels):
        cluster_to_pws[cid].append(pw_ids[idx])

    representatives = []
    for cid, members in cluster_to_pws.items():
        if len(members) == 1:
            representatives.append(members[0])
        else:
            # Pick member with highest IQR
            best_pw = None
            best_iqr = -1.0
            for pw_id in members:
                if pw_id in scores_df.index:
                    vals = scores_df.loc[pw_id, sample_cols].values.astype(float)
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
                        if iqr > best_iqr:
                            best_iqr = iqr
                            best_pw = pw_id
            if best_pw:
                representatives.append(best_pw)
            else:
                representatives.append(members[0])

    return representatives


def stage1_deredundancy(scores_df: pd.DataFrame,
                        pathway_sets: dict) -> tuple[list[str], dict]:
    """
    Stage 1: Remove redundant pathways via Jaccard + score correlation clustering.
    Runs sensitivity analysis at multiple thresholds.
    """
    print("\n[Stage 1] De-Redundancy (Jaccard + Score Correlation)")
    print("─" * 55)

    pw_ids_all = [pid for pid in pathway_sets.keys() if pid in scores_df.index]
    print(f"  Input: {len(pw_ids_all)} pathways")

    # Compute similarity matrices
    print("  Computing Jaccard similarity matrix...")
    jaccard, pw_ids = compute_jaccard_matrix(
        {pid: pathway_sets[pid] for pid in pw_ids_all})

    # Align pw_ids with scores_df index
    pw_ids = [pid for pid in pw_ids if pid in scores_df.index]
    jaccard_aligned, _ = compute_jaccard_matrix(
        {pid: pathway_sets[pid] for pid in pw_ids})

    print("  Computing combined similarity (Jaccard + |Pearson r|)...")
    combined = compute_combined_similarity(
        jaccard_aligned, scores_df, pw_ids, use_corr=SCORE_CORR_WEIGHT)

    # Sensitivity analysis
    sensitivity_results = {}
    for t in JACCARD_SENSITIVITY:
        reps = cluster_and_select_representatives(
            combined, scores_df, pw_ids, pathway_sets, t)
        sensitivity_results[str(t)] = {
            'threshold': t,
            'n_input': len(pw_ids),
            'n_output': len(reps),
            'reduction_pct': round((1 - len(reps) / len(pw_ids)) * 100, 1),
        }
        print(f"  Threshold {t}: {len(pw_ids)} → {len(reps)} "
              f"({sensitivity_results[str(t)]['reduction_pct']}% reduction)")

    # Use the primary threshold for actual selection
    representatives = cluster_and_select_representatives(
        combined, scores_df, pw_ids, pathway_sets, JACCARD_THRESHOLD)

    print(f"\n  ✓ Selected threshold {JACCARD_THRESHOLD}: "
          f"{len(pw_ids)} → {len(representatives)} non-redundant pathways")

    return representatives, sensitivity_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 2: Variance Filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stage2_variance_filter(scores_df: pd.DataFrame,
                            pathway_ids: list[str]) -> list[str]:
    """
    Stage 2: Remove pathways with very low IQR (uninformative flat scores).
    """
    print("\n[Stage 2] Variance Filtering (IQR-based)")
    print("─" * 55)
    print(f"  Input: {len(pathway_ids)} pathways")

    sample_cols = [c for c in scores_df.columns if c in SAMPLE_COLS]

    # Compute IQR for each pathway
    iqrs = {}
    for pw_id in pathway_ids:
        if pw_id in scores_df.index:
            vals = scores_df.loc[pw_id, sample_cols].values.astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                iqrs[pw_id] = np.percentile(vals, 75) - np.percentile(vals, 25)
            else:
                iqrs[pw_id] = 0.0

    if not iqrs:
        return pathway_ids

    all_iqrs = np.array(list(iqrs.values()))
    cutoff = np.percentile(all_iqrs, IQR_PERCENTILE_CUTOFF)

    retained = [pid for pid, iqr in iqrs.items() if iqr > cutoff]

    print(f"  IQR range: {all_iqrs.min():.4f} – {all_iqrs.max():.4f}")
    print(f"  Cutoff (p{IQR_PERCENTILE_CUTOFF}): {cutoff:.4f}")
    print(f"  ✓ Retained: {len(retained)} / {len(pathway_ids)} "
          f"(removed {len(pathway_ids) - len(retained)} low-variance pathways)")

    return retained


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 3: Elastic Net + Stability Selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_feature_matrix(scores_df: pd.DataFrame,
                          pathway_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build the X (pathways × samples transposed to samples × pathways) and y (labels).
    """
    sample_cols = [c for c in scores_df.columns if c in SAMPLE_COLS]
    patient_set = set(PATIENT_COLS)
    control_set = set(CONTROL_COLS)

    # Ensure pathway_ids are in scores_df
    valid_ids = [pid for pid in pathway_ids if pid in scores_df.index]

    X = scores_df.loc[valid_ids, sample_cols].values.astype(float).T  # (n_samples, n_pathways)
    y = np.array([1 if s in patient_set else 0 for s in sample_cols])  # 1=AD, 0=Control

    # Fill any NaNs with column means
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]

    return X, y, valid_ids


def stage3_stability_selection(scores_df: pd.DataFrame,
                                pathway_ids: list[str]) -> pd.DataFrame:
    """
    Stage 3: Elastic Net + Stability Selection.

    Runs ElasticNetCV on B bootstrap subsamples.
    Tracks which pathways are selected (non-zero coef) in each run.
    Returns pathways selected in >= STABILITY_THRESHOLD fraction of runs.
    """
    print(f"\n[Stage 3] Elastic Net + Stability Selection "
          f"(B={STABILITY_N_BOOTSTRAP})")
    print("─" * 55)

    X, y, valid_ids = build_feature_matrix(scores_df, pathway_ids)
    n_samples, n_features = X.shape
    print(f"  Feature matrix: {n_samples} samples × {n_features} pathways")
    print(f"  Labels: {y.sum()} AD, {(1-y).sum()} Control")

    # Track selection counts
    selection_counts = np.zeros(n_features)
    coef_sums = np.zeros(n_features)
    successful_runs = 0

    rng = np.random.RandomState(RANDOM_STATE)

    for b in range(STABILITY_N_BOOTSTRAP):
        # Bootstrap subsample (same size, with replacement)
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]

        # Skip if bootstrap has only one class
        if len(np.unique(y_boot)) < 2:
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_boot)

        try:
            # ElasticNetCV with class-weighted labels
            # Use SGDClassifier-like approach: logistic loss + elastic net penalty
            model = LogisticRegressionCV(
                penalty='elasticnet',
                solver='saga',
                l1_ratios=ELASTIC_NET_L1_RATIOS,
                Cs=10,
                cv=min(ELASTIC_NET_CV_FOLDS, min(np.bincount(y_boot))),
                max_iter=5000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                scoring='roc_auc',
                n_jobs=-1,
            )
            model.fit(X_scaled, y_boot)

            # Record non-zero coefficients
            coefs = model.coef_.flatten()
            selected = np.abs(coefs) > 1e-8
            selection_counts += selected.astype(float)
            coef_sums += coefs
            successful_runs += 1

        except Exception as e:
            continue  # Skip failed bootstrap iteration

        if (b + 1) % 50 == 0:
            print(f"    Completed {b + 1}/{STABILITY_N_BOOTSTRAP} bootstraps...")

    print(f"  Successful runs: {successful_runs}/{STABILITY_N_BOOTSTRAP}")

    if successful_runs == 0:
        print("  ⚠ No successful bootstrap runs — returning all pathways")
        return pd.DataFrame({
            'pathway_id': valid_ids,
            'selection_prob': [0.0] * len(valid_ids),
            'mean_coef': [0.0] * len(valid_ids),
            'direction': [''] * len(valid_ids),
        })

    # Compute selection probabilities
    selection_probs = selection_counts / successful_runs
    mean_coefs = coef_sums / successful_runs

    # Build results DataFrame
    results = pd.DataFrame({
        'pathway_id': valid_ids,
        'selection_prob': selection_probs,
        'mean_coef': mean_coefs,
        'direction': ['↑ in AD' if c > 0 else '↓ in AD' if c < 0 else '—'
                      for c in mean_coefs],
    })

    # Filter by stability threshold
    stable = results[results['selection_prob'] >= STABILITY_THRESHOLD].copy()
    stable = stable.sort_values('selection_prob', ascending=False).reset_index(drop=True)

    print(f"\n  Selection probability distribution:")
    for pct in [0.9, 0.8, 0.7, 0.6, 0.5]:
        count = (selection_probs >= pct).sum()
        print(f"    ≥ {pct:.0%}: {count} pathways")

    print(f"\n  ✓ Stable pathways (≥ {STABILITY_THRESHOLD:.0%}): {len(stable)}")

    return stable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 4: Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def loocv_evaluate(X: np.ndarray, y: np.ndarray,
                   classifier_type: str) -> dict:
    """
    Leave-One-Out Cross-Validation with the specified classifier.
    Returns accuracy, AUC, and Cohen's kappa.
    """
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    y_proba = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if classifier_type == 'logistic':
            model = LogisticRegressionCV(
                penalty='l2', solver='lbfgs',
                cv=min(5, min(np.bincount(y_train))),
                max_iter=5000,
                class_weight='balanced',
                random_state=RANDOM_STATE,
            )
        elif classifier_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=3,  # Shallow trees for small n
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)
        y_proba[test_idx] = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = np.nan
    kappa = cohen_kappa_score(y, y_pred)

    return {'accuracy': round(acc, 3), 'auc': round(auc, 3), 'kappa': round(kappa, 3)}


def permutation_test(X: np.ndarray, y: np.ndarray,
                     real_accuracy: float) -> dict:
    """
    Permutation test: shuffle labels N times, run LOOCV each time.

    Test statistic = LOOCV accuracy on the selected pathways.
    If real accuracy >> permuted accuracy distribution → pathways are genuine.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n_perm_better = 0
    perm_accuracies = []

    for p in range(PERMUTATION_N):
        y_perm = rng.permutation(y)

        # Quick LOOCV with logistic regression on shuffled labels
        loo = LeaveOneOut()
        y_pred = np.zeros(len(y_perm))

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y_perm[train_idx]

            if len(np.unique(y_train)) < 2:
                y_pred[test_idx] = rng.choice([0, 1])
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            try:
                model = LogisticRegressionCV(
                    penalty='l2', solver='lbfgs',
                    cv=min(5, min(np.bincount(y_train))),
                    max_iter=2000,
                    class_weight='balanced',
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train_s, y_train)
                y_pred[test_idx] = model.predict(X_test_s)
            except Exception:
                y_pred[test_idx] = rng.choice([0, 1])

        perm_acc = accuracy_score(y_perm, y_pred)
        perm_accuracies.append(perm_acc)

        if perm_acc >= real_accuracy:
            n_perm_better += 1

        if (p + 1) % 100 == 0:
            print(f"    Permutation {p + 1}/{PERMUTATION_N}...")

    p_value = (n_perm_better + 1) / (PERMUTATION_N + 1)

    return {
        'p_value': round(p_value, 4),
        'n_permutations': PERMUTATION_N,
        'real_accuracy': real_accuracy,
        'mean_perm_accuracy': round(np.mean(perm_accuracies), 3),
        'std_perm_accuracy': round(np.std(perm_accuracies), 3),
        'max_perm_accuracy': round(np.max(perm_accuracies), 3),
    }


def biological_cross_reference(stable_pathways: pd.DataFrame,
                                 pathway_sets: dict) -> list[dict]:
    """
    Cross-reference selected pathways with known AD biology.
    """
    ad_keywords = [
        'alzheimer', 'amyloid', 'tau', 'neurodegeneration', 'complement',
        'lipid', 'cholesterol', 'apolipoprotein', 'neuroinflammation',
        'synapse', 'synaptic', 'autophagy', 'lysosome', 'endocytosis',
        'apoptosis', 'oxidative stress', 'mitochondri', 'immune',
        'coagulation', 'insulin', 'calcium', 'neurotransmitter',
    ]

    matches = []
    for _, row in stable_pathways.iterrows():
        pw_id = row['pathway_id']
        if pw_id in pathway_sets:
            pw_name = pathway_sets[pw_id]['name'].lower()
            matched_keywords = [kw for kw in ad_keywords if kw in pw_name]
            if matched_keywords:
                matches.append({
                    'pathway_id': pw_id,
                    'pathway_name': pathway_sets[pw_id]['name'],
                    'ad_keywords': ', '.join(matched_keywords),
                    'source': pathway_sets[pw_id]['source'],
                })

    return matches


def _inner_stability_selection(X_train: np.ndarray, y_train: np.ndarray,
                                n_boot: int = 100) -> np.ndarray:
    """
    Run a lightweight stability selection on the training fold.
    Returns boolean mask of selected features.
    Uses fewer bootstraps than the full Stage 3 for speed.
    """
    n_samples, n_features = X_train.shape
    selection_counts = np.zeros(n_features)
    successful = 0
    rng = np.random.RandomState(RANDOM_STATE)

    for b in range(n_boot):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        Xb, yb = X_train[boot_idx], y_train[boot_idx]

        if len(np.unique(yb)) < 2:
            continue

        scaler = StandardScaler()
        Xb_s = scaler.fit_transform(Xb)

        try:
            model = LogisticRegressionCV(
                penalty='elasticnet', solver='saga',
                l1_ratios=ELASTIC_NET_L1_RATIOS,
                Cs=10,
                cv=min(ELASTIC_NET_CV_FOLDS, min(np.bincount(yb))),
                max_iter=5000,
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
        return np.ones(n_features, dtype=bool)  # fallback: use all

    probs = selection_counts / successful
    return probs >= STABILITY_THRESHOLD


def nested_loocv(scores_df: pd.DataFrame,
                 filtered_ids: list[str]) -> dict:
    """
    Nested LOOCV: embeds stability selection INSIDE each fold.

    For each of the n leave-one-out folds:
      1. Remove 1 test sample → n-1 training samples
      2. Re-run stability selection on training samples only
      3. Train logistic regression on selected features
      4. Predict the held-out sample

    This eliminates information leakage from feature selection and gives
    an unbiased estimate of generalization accuracy.

    Reference: Ambroise & McLachlan (2002) PNAS — Selection bias in
    gene extraction on the basis of microarray gene-expression data.
    """
    # Use fewer bootstraps per fold for tractability (100 instead of 200)
    N_INNER_BOOT = 100

    X_full, y_full, valid_ids = build_feature_matrix(scores_df, filtered_ids)
    n_samples, n_features = X_full.shape

    print(f"\n  [4a-bis] Nested LOOCV (stability selection inside each fold)")
    print(f"    Full matrix: {n_samples} samples × {n_features} pathways")
    print(f"    Inner bootstraps per fold: {N_INNER_BOOT}")

    loo = LeaveOneOut()
    y_pred = np.zeros(n_samples)
    y_proba = np.zeros(n_samples)
    fold_n_features = []  # track how many features each fold selects

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(X_full)):
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]

        # Step 1: Re-run stability selection on training data only
        feature_mask = _inner_stability_selection(
            X_train, y_train, n_boot=N_INNER_BOOT
        )
        n_sel = feature_mask.sum()
        fold_n_features.append(n_sel)

        if n_sel == 0:
            # No features selected — predict majority class
            y_pred[test_idx] = np.round(y_train.mean())
            y_proba[test_idx] = y_train.mean()
            continue

        # Step 2: Train on selected features only
        X_train_sel = X_train[:, feature_mask]
        X_test_sel = X_test[:, feature_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s = scaler.transform(X_test_sel)

        try:
            model = LogisticRegressionCV(
                penalty='l2', solver='lbfgs',
                cv=min(5, min(np.bincount(y_train))),
                max_iter=5000,
                class_weight='balanced',
                random_state=RANDOM_STATE,
            )
            model.fit(X_train_s, y_train)
            y_pred[test_idx] = model.predict(X_test_s)
            y_proba[test_idx] = model.predict_proba(X_test_s)[:, 1]
        except Exception:
            y_pred[test_idx] = np.round(y_train.mean())
            y_proba[test_idx] = y_train.mean()

        if (fold_i + 1) % 10 == 0:
            print(f"    Fold {fold_i + 1}/{n_samples} "
                  f"(selected {n_sel} features)")

    acc = accuracy_score(y_full, y_pred)
    try:
        auc = roc_auc_score(y_full, y_proba)
    except ValueError:
        auc = np.nan
    kappa = cohen_kappa_score(y_full, y_pred)

    avg_features = np.mean(fold_n_features)
    min_features = np.min(fold_n_features)
    max_features = np.max(fold_n_features)

    print(f"    ──────────────────────────")
    print(f"    Nested LOOCV Accuracy: {acc:.3f}")
    print(f"    Nested LOOCV AUC:      {auc:.3f}")
    print(f"    Nested LOOCV κ:        {kappa:.3f}")
    print(f"    Features per fold: mean={avg_features:.1f}, "
          f"min={min_features}, max={max_features}")

    return {
        'accuracy': round(acc, 3),
        'auc': round(auc, 3),
        'kappa': round(kappa, 3),
        'features_per_fold_mean': round(avg_features, 1),
        'features_per_fold_min': int(min_features),
        'features_per_fold_max': int(max_features),
    }


def stage4_validation(scores_df: pd.DataFrame,
                       stable_pathways: pd.DataFrame,
                       pathway_sets: dict,
                       filtered_ids: list[str] | None = None) -> dict:
    """
    Stage 4: Model-agnostic validation.
    - Standard LOOCV (on pre-selected features — has selection bias)
    - Nested LOOCV (re-selects features per fold — unbiased)
    - Permutation test
    - Biological cross-reference
    """
    print(f"\n[Stage 4] Model-Agnostic Validation")
    print("─" * 55)

    stable_ids = stable_pathways['pathway_id'].tolist()
    X, y, valid_ids = build_feature_matrix(scores_df, stable_ids)
    print(f"  Validation matrix: {X.shape[0]} samples × {X.shape[1]} pathways")

    results = {}

    # 4a. Standard LOOCV (on pre-selected features — has selection bias)
    print("\n  [4a] Standard LOOCV (features pre-selected — optimistic estimate)")
    for clf_name in ['logistic', 'rf']:
        clf_label = 'Logistic Regression' if clf_name == 'logistic' else 'Random Forest'
        metrics = loocv_evaluate(X, y, clf_name)
        results[f'loocv_{clf_name}'] = metrics
        print(f"    {clf_label}: Acc={metrics['accuracy']:.3f}  "
              f"AUC={metrics['auc']:.3f}  κ={metrics['kappa']:.3f}")

    # 4a-bis. Nested LOOCV (re-selects features per fold — unbiased)
    if filtered_ids is not None:
        nested_metrics = nested_loocv(scores_df, filtered_ids)
        results['nested_loocv'] = nested_metrics
    else:
        print("\n  [4a-bis] Nested LOOCV skipped (filtered_ids not provided)")

    # 4b. Permutation test
    print(f"\n  [4b] Permutation test ({PERMUTATION_N} shuffles)...")
    real_acc = results['loocv_logistic']['accuracy']
    perm_results = permutation_test(X, y, real_acc)
    results['permutation'] = perm_results
    print(f"    Real LOOCV accuracy: {perm_results['real_accuracy']:.3f}")
    print(f"    Mean permuted accuracy: {perm_results['mean_perm_accuracy']:.3f} "
          f"± {perm_results['std_perm_accuracy']:.3f}")
    print(f"    Max permuted accuracy: {perm_results['max_perm_accuracy']:.3f}")
    print(f"    Empirical p-value: {perm_results['p_value']:.4f}")

    # 4c. Biological cross-reference
    print("\n  [4c] Biological cross-reference with AD literature")
    bio_matches = biological_cross_reference(stable_pathways, pathway_sets)
    results['biological_matches'] = len(bio_matches)
    results['biological_details'] = bio_matches

    if bio_matches:
        print(f"    Found {len(bio_matches)} pathways matching AD keywords:")
        for m in bio_matches[:10]:
            print(f"      • {m['pathway_name']} [{m['source']}] "
                  f"({m['ad_keywords']})")
    else:
        print("    ⚠ No direct AD keyword matches (may still be biologically relevant)")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_phase5():
    t0 = time.time()
    print("\n" + "=" * 70)
    print("PHASE 5 — Pathway Dimensionality Reduction")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    scores_df = pd.read_csv(IN_SCORES, index_col=0)
    with open(IN_PATHWAYS) as f:
        pathway_sets = json.load(f)

    sample_cols = [c for c in scores_df.columns if c in SAMPLE_COLS]
    print(f"  Pathway scores: {scores_df.shape[0]} pathways × {len(sample_cols)} samples")
    print(f"  Pathway gene sets: {len(pathway_sets)}")

    report = {
        'input_pathways': scores_df.shape[0],
        'input_samples': len(sample_cols),
    }

    # ── Stage 1: De-Redundancy ────────────────────────────────────────────
    nonredundant_ids, sensitivity = stage1_deredundancy(scores_df, pathway_sets)
    report['stage1_deredundancy'] = {
        'output': len(nonredundant_ids),
        'reduction_pct': round((1 - len(nonredundant_ids) / report['input_pathways']) * 100, 1),
        'sensitivity_analysis': sensitivity,
    }

    # Save non-redundant pathways
    nonred_records = []
    for pid in nonredundant_ids:
        info = pathway_sets.get(pid, {})
        nonred_records.append({
            'pathway_id': pid,
            'pathway_name': info.get('name', ''),
            'source': info.get('source', ''),
            'n_genes': info.get('size_matched', 0),
        })
    pd.DataFrame(nonred_records).to_csv(OUT_NONRED, index=False)
    print(f"\n  → Saved: {OUT_NONRED}")

    # ── Stage 2: Variance Filtering ───────────────────────────────────────
    filtered_ids = stage2_variance_filter(scores_df, nonredundant_ids)
    report['stage2_variance_filter'] = {
        'output': len(filtered_ids),
        'removed': len(nonredundant_ids) - len(filtered_ids),
    }

    # ── Stage 3: Stability Selection ──────────────────────────────────────
    stable_pathways = stage3_stability_selection(scores_df, filtered_ids)
    report['stage3_stability_selection'] = {
        'output': len(stable_pathways),
        'bootstrap_iterations': STABILITY_N_BOOTSTRAP,
        'threshold': STABILITY_THRESHOLD,
    }

    # Enrich stable_pathways with pathway metadata
    if len(stable_pathways) > 0:
        stable_pathways['pathway_name'] = stable_pathways['pathway_id'].map(
            lambda x: pathway_sets.get(x, {}).get('name', ''))
        stable_pathways['source'] = stable_pathways['pathway_id'].map(
            lambda x: pathway_sets.get(x, {}).get('source', ''))
        stable_pathways['n_genes'] = stable_pathways['pathway_id'].map(
            lambda x: pathway_sets.get(x, {}).get('size_matched', 0))

        # Save stable pathways
        stable_pathways.to_csv(OUT_STABLE, index=False)
        print(f"\n  → Saved: {OUT_STABLE}")

        # Print top stable pathways
        print(f"\n  Top stable pathways:")
        print(f"  {'Pathway':<45} {'Prob':>6} {'Dir':<10} {'Source':<8}")
        print("  " + "─" * 72)
        for _, row in stable_pathways.head(20).iterrows():
            name = str(row['pathway_name'])[:44]
            prob = row['selection_prob']
            direction = row['direction']
            source = str(row.get('source', ''))
            print(f"  {name:<45} {prob:>5.1%} {direction:<10} {source:<8}")

    # ── Stage 4: Validation ───────────────────────────────────────────────
    if len(stable_pathways) >= 2:
        validation = stage4_validation(
            scores_df, stable_pathways, pathway_sets,
            filtered_ids=filtered_ids,
        )
        report['stage4_validation'] = validation

        # Save validation results
        val_rows = []
        for clf in ['logistic', 'rf']:
            key = f'loocv_{clf}'
            if key in validation:
                val_rows.append({
                    'metric': f'LOOCV_{clf}',
                    **validation[key],
                })
        if 'permutation' in validation:
            val_rows.append({
                'metric': 'permutation_test',
                'accuracy': validation['permutation']['real_accuracy'],
                'auc': validation['permutation']['mean_perm_accuracy'],
                'kappa': validation['permutation']['p_value'],
            })
        if 'nested_loocv' in validation:
            val_rows.append({
                'metric': 'nested_LOOCV_logistic',
                **validation['nested_loocv'],
            })
        pd.DataFrame(val_rows).to_csv(OUT_VALID, index=False)
        print(f"\n  → Saved: {OUT_VALID}")
    else:
        print("\n  ⚠ Too few stable pathways for validation — skipping Stage 4")
        report['stage4_validation'] = {'skipped': True, 'reason': 'too few pathways'}

    # ── Save report ───────────────────────────────────────────────────────
    # Remove non-serializable items from biological_details
    if 'stage4_validation' in report:
        bio = report['stage4_validation'].get('biological_details', [])
        report['stage4_validation']['biological_details'] = bio

    with open(OUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  → Saved: {OUT_REPORT}")

    # ── Final Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  REDUCTION FUNNEL")
    print(f"  {'Stage':<30} {'Count':>8} {'Δ':>10}")
    print(f"  {'─' * 50}")
    print(f"  {'Input pathways':<30} {report['input_pathways']:>8}")
    n1 = report['stage1_deredundancy']['output']
    print(f"  {'After de-redundancy':<30} {n1:>8} "
          f"{'−' + str(report['input_pathways'] - n1):>10}")
    n2 = report['stage2_variance_filter']['output']
    print(f"  {'After variance filter':<30} {n2:>8} "
          f"{'−' + str(n1 - n2):>10}")
    n3 = report['stage3_stability_selection']['output']
    print(f"  {'Stable pathways':<30} {n3:>8} "
          f"{'−' + str(n2 - n3):>10}")
    print(f"  {'─' * 50}")
    pct = round((1 - n3 / report['input_pathways']) * 100, 1) if report['input_pathways'] > 0 else 0
    print(f"  Total reduction: {report['input_pathways']} → {n3} ({pct}%)")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    return report


if __name__ == '__main__':
    run_phase5()
