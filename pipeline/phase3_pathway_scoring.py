"""
Phase 3 – Pathway Activity Scoring (Rewritten)

Steps:
  1. Load log2-transformed canonical data and pathway gene sets
  2. Fetch protein importance weights from DisGeNET (AD association)
  3. Fetch network centrality from STRING PPI
  4. Compute weighted ssGSEA scores per sample per pathway (sparse-aware)
  5. Save pathway activity score matrix
"""

import os, json, time
import numpy as np
import pandas as pd
import requests

from pipeline.config import (
    OUTPUT_DIR, SAMPLE_COLS, PATIENT_COLS, CONTROL_COLS,
    STRING_SPECIES, STRING_API_BASE,
)

# ── File paths ────────────────────────────────────────────────────────────────
IN_CANONICAL  = os.path.join(OUTPUT_DIR, "canonical_log2.csv")
IN_PATHWAYS   = os.path.join(OUTPUT_DIR, "pathway_gene_sets.json")
OUT_SCORES    = os.path.join(OUTPUT_DIR, "pathway_scores.csv")
OUT_WEIGHTS   = os.path.join(OUTPUT_DIR, "protein_weights.csv")
OUT_SUMMARY   = os.path.join(OUTPUT_DIR, "pathway_summary.csv")

# ── DisGeNET API ──────────────────────────────────────────────────────────────
DISGENET_API = "https://www.disgenet.org/api"
AD_DISEASE_ID = "C0002395"  # UMLS CUI for Alzheimer's Disease


def fetch_disgenet_weights(gene_symbols: list[str]) -> dict[str, float]:
    """
    Fetch gene-disease association (GDA) scores from DisGeNET for AD.

    The GDA score (0–1) represents how strongly a gene is associated with
    Alzheimer's Disease in the literature. Higher = more AD-relevant.

    If the API is unavailable, falls back to a curated set of known AD genes.
    """
    print("  Fetching AD gene-disease association scores from DisGeNET...")
    weights = {}

    # DisGeNET public API (no key needed for basic queries)
    try:
        # Query in batches of 100
        for i in range(0, len(gene_symbols), 100):
            batch = gene_symbols[i:i+100]
            url = f"{DISGENET_API}/gda/disease/{AD_DISEASE_ID}"
            params = {
                'source': 'ALL',
                'format': 'json',
            }
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                for entry in data:
                    gene = entry.get('gene_symbol', '')
                    score = entry.get('score', 0.0)
                    if gene in set(gene_symbols):
                        weights[gene] = score
                print(f"    Retrieved {len(weights)} AD-associated genes from DisGeNET")
                break  # Full disease query returns all genes at once
            elif resp.status_code == 429:
                print(f"    Rate limited, waiting 5s...")
                time.sleep(5)
            else:
                print(f"    DisGeNET returned {resp.status_code}, using fallback")
                break

            time.sleep(1)

    except Exception as e:
        print(f"    ⚠ DisGeNET API error: {e}")

    # Fallback: curated AD-relevant gene weights
    if len(weights) < 10:
        print("    Using curated AD gene weights as fallback...")
        curated_ad_genes = {
            # Top AD genetic risk factors
            'APOE': 0.95, 'CLU': 0.85, 'BIN1': 0.80, 'PICALM': 0.75,
            'CR1': 0.70, 'ABCA7': 0.70, 'TREM2': 0.80, 'SORL1': 0.75,
            # Core AD biomarkers
            'APP': 0.95, 'MAPT': 0.90, 'PSEN1': 0.90, 'PSEN2': 0.85,
            'BACE1': 0.80, 'BACE2': 0.60,
            # Neuroprotective / CSF biomarkers
            'CST3': 0.65, 'TTR': 0.60, 'GSN': 0.55, 'SERPINA3': 0.60,
            'CHI3L1': 0.65, 'NRGN': 0.70, 'NEFL': 0.75,
            # Complement system
            'C1QA': 0.60, 'C1QB': 0.60, 'C1QC': 0.60,
            'C2': 0.50, 'C3': 0.65, 'C4A': 0.55, 'C4B': 0.55,
            'CFH': 0.55, 'CFI': 0.50, 'CFB': 0.50, 'C5': 0.50,
            # Lipid metabolism
            'APOA1': 0.50, 'APOA4': 0.40, 'APOB': 0.40,
            'APOC1': 0.50, 'APOC3': 0.45, 'APOD': 0.55, 'APOM': 0.40,
            'APOL1': 0.40,
            # Inflammation / cytokines
            'CRP': 0.55, 'SAA1': 0.50, 'SAA2': 0.45,
            'SERPINA1': 0.50, 'SERPINF2': 0.40,
            # Coagulation
            'FGA': 0.40, 'FGB': 0.40, 'FGG': 0.40, 'PLG': 0.35,
            # Synaptic / neuronal
            'SYN1': 0.60, 'SYP': 0.55, 'SNAP25': 0.60, 'STX1A': 0.55,
            # Plasma / QC
            'ALB': 0.20, 'TF': 0.30, 'HP': 0.35, 'HPX': 0.30,
        }
        for gene, score in curated_ad_genes.items():
            if gene in set(gene_symbols) and gene not in weights:
                weights[gene] = score

    print(f"    Total genes with AD weights: {len(weights)}")
    return weights


def fetch_string_centrality(gene_symbols: list[str]) -> dict[str, float]:
    """
    Fetch network degree centrality from STRING PPI network.

    Higher degree = more interaction partners = more central in the network.
    Returns normalized scores (0–1).
    """
    print("  Fetching STRING PPI network centrality...")
    centrality = {}

    try:
        # Get interaction counts in batches
        batch_size = 200
        all_interactions = {}

        for i in range(0, len(gene_symbols), batch_size):
            batch = gene_symbols[i:i+batch_size]
            url = f"{STRING_API_BASE}/json/interaction_partners"
            params = {
                'identifiers': '\r'.join(batch),
                'species': STRING_SPECIES,
                'limit': 0,  # just count
                'required_score': 700,  # high confidence
            }

            try:
                resp = requests.post(url, data=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    for interaction in data:
                        gene_a = interaction.get('preferredName_A', '')
                        gene_b = interaction.get('preferredName_B', '')
                        all_interactions[gene_a] = all_interactions.get(gene_a, 0) + 1
                        all_interactions[gene_b] = all_interactions.get(gene_b, 0) + 1
                else:
                    print(f"    STRING batch {i//batch_size} returned {resp.status_code}")
            except Exception as e:
                print(f"    STRING batch {i//batch_size} error: {e}")

            time.sleep(1)

        # Normalize to 0–1
        if all_interactions:
            max_degree = max(all_interactions.values())
            for gene in gene_symbols:
                if gene in all_interactions:
                    centrality[gene] = all_interactions[gene] / max_degree
            print(f"    Retrieved centrality for {len(centrality)} genes "
                  f"(max degree: {max_degree})")

    except Exception as e:
        print(f"    ⚠ STRING API error: {e}")

    return centrality


def compute_protein_weights(gene_symbols: list[str]) -> dict[str, float]:
    """
    Compute combined protein importance weights from databases.

    Weight = 0.6 * DisGeNET_AD_score  +  0.4 * STRING_centrality
    Default weight = 0.3 for unscored proteins (they still contribute, just less).
    """
    disgenet_w = fetch_disgenet_weights(gene_symbols)
    string_w = fetch_string_centrality(gene_symbols)

    combined = {}
    for gene in gene_symbols:
        d_score = disgenet_w.get(gene, 0.0)
        s_score = string_w.get(gene, 0.0)

        if d_score > 0 or s_score > 0:
            # Weighted combination: prioritize disease relevance
            combined[gene] = 0.6 * d_score + 0.4 * s_score
        else:
            combined[gene] = 0.3  # default baseline weight

    return combined


# ── Sparse-aware ssGSEA ──────────────────────────────────────────────────────

def ssgsea_score(ranked_values: np.ndarray, gene_mask: np.ndarray,
                 weights: np.ndarray | None = None) -> float:
    """
    Compute a single ssGSEA enrichment score for one sample & one pathway.

    Parameters:
        ranked_values: protein abundances sorted descending (only observed values)
        gene_mask: boolean mask — True for pathway members in the ranked list
        weights: importance weights for each position (same length as ranked_values)

    Returns:
        Enrichment score (float). Higher = pathway more active in this sample.
    """
    n = len(ranked_values)
    n_hit = gene_mask.sum()
    n_miss = n - n_hit

    if n_hit == 0 or n_miss == 0:
        return 0.0

    if weights is None:
        weights = np.ones(n)

    # Weighted cumulative sums
    hit_weights = np.where(gene_mask, np.abs(ranked_values) * weights, 0.0)
    hit_sum = hit_weights.sum()

    if hit_sum == 0:
        return 0.0

    # Normalized running sums
    p_hit = np.cumsum(hit_weights) / hit_sum
    p_miss = np.cumsum(~gene_mask) / n_miss

    # Enrichment score = max deviation
    es = (p_hit - p_miss)
    # Use the sum of positive and negative deviations (standard ssGSEA)
    return es.sum() / n


def compute_pathway_scores(canonical_df: pd.DataFrame,
                           pathway_sets: dict,
                           protein_weights: dict) -> pd.DataFrame:
    """
    Compute weighted ssGSEA pathway activity scores for each sample.

    Handles sparse data by ranking only observed (non-NA) proteins per sample.
    """
    data_cols = [c for c in canonical_df.columns if c in SAMPLE_COLS]
    gene_col = 'gene_symbol'

    # Map gene symbols to row indices
    gene_to_idx = {}
    for idx, row in canonical_df.iterrows():
        gene = row[gene_col]
        if isinstance(gene, str) and len(gene) >= 2:
            gene_to_idx[gene] = idx

    results = {}
    n_pathways = len(pathway_sets)

    for pw_i, (pw_id, pw_info) in enumerate(pathway_sets.items()):
        pw_genes = set(pw_info['genes'])
        pw_gene_indices = {g: gene_to_idx[g] for g in pw_genes if g in gene_to_idx}

        if len(pw_gene_indices) < 3:
            continue

        scores = {}
        for sample in data_cols:
            # Get all observed (non-NA) values for this sample
            sample_data = canonical_df[sample]
            observed_mask = sample_data.notna()
            observed_genes = canonical_df.loc[observed_mask, gene_col].values
            observed_values = sample_data[observed_mask].values.astype(float)

            if len(observed_values) < 10:
                scores[sample] = np.nan
                continue

            # Sort by abundance (descending)
            sort_idx = np.argsort(-observed_values)
            sorted_values = observed_values[sort_idx]
            sorted_genes = observed_genes[sort_idx]

            # Create pathway membership mask
            gene_mask = np.array([g in pw_genes for g in sorted_genes])

            # Create weight vector
            weight_vec = np.array([
                protein_weights.get(g, 0.3) if isinstance(g, str) else 0.3
                for g in sorted_genes
            ])

            if gene_mask.sum() == 0:
                scores[sample] = 0.0
                continue

            scores[sample] = ssgsea_score(sorted_values, gene_mask, weight_vec)

        results[pw_id] = scores

        if (pw_i + 1) % 100 == 0:
            print(f"    Scored {pw_i + 1}/{n_pathways} pathways...")

    return pd.DataFrame(results).T  # pathways × samples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_phase3():
    print("\n" + "=" * 70)
    print("PHASE 3 — Pathway Activity Scoring (Rewritten)")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    canonical = pd.read_csv(IN_CANONICAL)
    with open(IN_PATHWAYS) as f:
        pathway_sets = json.load(f)

    data_cols = [c for c in canonical.columns if c in SAMPLE_COLS]
    print(f"  Canonical proteins: {len(canonical)}")
    print(f"  Pathways loaded: {len(pathway_sets)}")

    # Collect all gene symbols
    all_genes = [g for g in canonical['gene_symbol'].dropna().unique()
                 if isinstance(g, str) and len(g) >= 2]
    print(f"  Unique gene symbols: {len(all_genes)}")

    # ── 2. Fetch protein importance weights ───────────────────────────────
    print("\n[2/5] Computing protein importance weights...")
    protein_weights = compute_protein_weights(all_genes)

    # Save weights
    weight_rows = [{'gene_symbol': g, 'weight': w}
                   for g, w in sorted(protein_weights.items(), key=lambda x: -x[1])]
    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(OUT_WEIGHTS, index=False)
    print(f"  → Saved: {OUT_WEIGHTS}")

    # Show top-weighted proteins
    print("  Top 15 weighted proteins:")
    for _, row in weight_df.head(15).iterrows():
        print(f"    {row['gene_symbol']:<12} {row['weight']:.3f}")

    # ── 3. Compute pathway activity scores ────────────────────────────────
    print("\n[3/5] Computing weighted ssGSEA pathway activity scores...")
    scores_df = compute_pathway_scores(canonical, pathway_sets, protein_weights)
    print(f"  Score matrix: {scores_df.shape[0]} pathways × {scores_df.shape[1]} samples")

    # ── 4. Add pathway info and save ──────────────────────────────────────
    print("\n[4/5] Saving results...")

    # Add pathway metadata
    scores_df['pathway_name'] = scores_df.index.map(
        lambda x: pathway_sets.get(x, {}).get('name', ''))
    scores_df['source'] = scores_df.index.map(
        lambda x: pathway_sets.get(x, {}).get('source', ''))
    scores_df['n_genes'] = scores_df.index.map(
        lambda x: pathway_sets.get(x, {}).get('size_matched', 0))

    scores_df.to_csv(OUT_SCORES)
    print(f"  → Saved: {OUT_SCORES}")

    # ── 5. Summary: top pathways by AD vs Control difference ──────────────
    print("\n[5/5] Top pathways by AD vs Control difference")
    print("─" * 60)

    patient_cols = [c for c in scores_df.columns if c in PATIENT_COLS]
    control_cols = [c for c in scores_df.columns if c in CONTROL_COLS]

    if patient_cols and control_cols:
        scores_df['mean_AD'] = scores_df[patient_cols].mean(axis=1)
        scores_df['mean_Ctrl'] = scores_df[control_cols].mean(axis=1)
        scores_df['diff'] = scores_df['mean_AD'] - scores_df['mean_Ctrl']
        scores_df['abs_diff'] = scores_df['diff'].abs()

        # Save summary
        summary = scores_df[['pathway_name', 'source', 'n_genes',
                             'mean_AD', 'mean_Ctrl', 'diff']].copy()
        summary = summary.sort_values('diff', key=abs, ascending=False)
        summary.to_csv(OUT_SUMMARY)
        print(f"  → Saved: {OUT_SUMMARY}")

        # Print top 20
        print(f"\n  {'Pathway':<45} {'Source':<8} {'Genes':>5} {'Diff':>8}")
        print("  " + "─" * 70)
        for _, row in summary.head(20).iterrows():
            name = row['pathway_name'][:44]
            print(f"  {name:<45} {row['source']:<8} {row['n_genes']:>5} "
                  f"{row['diff']:>+8.3f}")

    print("\n" + "=" * 70)
    return scores_df


if __name__ == '__main__':
    run_phase3()
