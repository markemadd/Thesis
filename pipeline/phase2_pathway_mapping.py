"""
Phase 2 – Pathway Mapping (Rewritten)

Steps:
  1. Load raw data + mapped_proteins.csv
  2. Separate Ig clonotypes from canonical proteins  →  save Ig for future use
  3. Improve gene-symbol mapping for unmapped canonical proteins
  4. Log2-transform all non-NA values
  5. Deduplicate (keep lowest-missingness entry per protein)
  6. Map gene symbols → pathways via gProfiler (GO, KEGG, Reactome, WikiPathways)
  7. Save outputs
"""

import os, re, json, time
import numpy as np
import pandas as pd
import requests
from gprofiler import GProfiler

from pipeline.config import (
    INPUT_CSV, OUTPUT_DIR, SAMPLE_COLS,
    MIN_PATHWAY_PROTEINS, MAX_PATHWAY_PROTEINS,
    BASE_DIR,
)

# ── File paths ────────────────────────────────────────────────────────────────
MAPPED_CSV      = os.path.join(BASE_DIR, "mapped_proteins.csv")
OUT_IG_CSV      = os.path.join(OUTPUT_DIR, "ig_clonotypes.csv")
OUT_CANONICAL   = os.path.join(OUTPUT_DIR, "canonical_log2.csv")
OUT_GENE_MAP    = os.path.join(OUTPUT_DIR, "gene_symbol_map.csv")
OUT_PATHWAYS    = os.path.join(OUTPUT_DIR, "pathway_gene_sets.json")

# ── Ig clonotype detection ────────────────────────────────────────────────────
IG_PATTERNS = [
    r'ighv', r'iglv', r'igkv',
    r'heavy chain variable', r'light chain variable',
    r'igh\b', r'igl\b', r'igk\b',
    r'immunoglobulin', r'\big\b',
    r'clone', r'clonotype', r'\bcdr\b',
    r'_heavy_', r'_light_',
    r'ighg', r'igha', r'ighm', r'ighe',
    r'iglc', r'igkc',
    r'heavy chain constant', r'light chain constant',
    r'10e8',
]
_ig_re = re.compile('|'.join(IG_PATTERNS), re.IGNORECASE)


def is_ig_protein(name: str) -> bool:
    """Return True if the protein name matches Ig clonotype patterns."""
    return bool(_ig_re.search(name))


# ── Gene symbol extraction heuristics ─────────────────────────────────────────
# Pattern: "GENE_SYMBOL protein" or "GENE_SYMBOL" or "protein name (GENE_SYMBOL)"
_gene_symbol_re = re.compile(r'^([A-Z][A-Z0-9]{1,14})\b')
_paren_gene_re  = re.compile(r'\(([A-Z][A-Z0-9]{1,14})\)')


def extract_gene_symbol_from_name(protein_name: str) -> str | None:
    """Try to extract a likely gene symbol from a UniProt descriptive name."""
    name = protein_name.strip()

    # Case 1: Name starts with what looks like a gene symbol (e.g., "ACTG1 protein")
    m = _gene_symbol_re.match(name)
    if m:
        candidate = m.group(1)
        # Exclude common false positives
        if candidate not in ('EC', 'DNA', 'RNA', 'ATP', 'ADP', 'GTP', 'NAD',
                             'MHC', 'HLA', 'ABC', 'BAG', 'BET', 'LOW'):
            return candidate

    # Case 2: Gene symbol in parentheses (e.g., "Clusterin (CLU)")
    for m in _paren_gene_re.finditer(name):
        candidate = m.group(1)
        if len(candidate) >= 2 and candidate not in ('EC', 'CD', 'IG'):
            return candidate

    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_phase2():
    print("\n" + "*" * 70)
    print("PHASE 2 — Pathway Mapping")
    print("*" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    raw = pd.read_csv(INPUT_CSV)
    print(f"  Raw data: {raw.shape[0]} proteins × {raw.shape[1] - 1} samples")

    # Load existing mapping from mapping.py output
    if os.path.exists(MAPPED_CSV):
        mapped = pd.read_csv(MAPPED_CSV)
        print(f"  Existing mapping: {mapped['Query_original'].nunique()} proteins mapped")
    else:
        mapped = pd.DataFrame(columns=['Entry', 'Gene Names', 'Protein names',
                                       'Query_original', 'Query_clean'])
        print("  ⚠ No mapped_proteins.csv found — will map from scratch")

    # ── 2. Separate Ig from canonical ─────────────────────────────────────
    print("\n[2/7] Separating Ig clonotypes from canonical proteins...")
    raw['is_ig'] = raw['Protein.names'].apply(is_ig_protein)

    ig_df = raw[raw['is_ig']].drop(columns=['is_ig']).reset_index(drop=True)
    canonical_df = raw[~raw['is_ig']].drop(columns=['is_ig']).reset_index(drop=True)

    print(f"  Ig clonotypes: {len(ig_df)} (saved for future use)")
    print(f"  Canonical proteins: {len(canonical_df)}")

    ig_df.to_csv(OUT_IG_CSV, index=False)
    print(f"  → Saved: {OUT_IG_CSV}")

    # ── 3. Build gene symbol map ──────────────────────────────────────────
    print("\n[3/7] Building gene symbol map for canonical proteins...")

    # Start from existing mapped_proteins.csv
    # Build: protein_name → (gene_symbol, uniprot_id)
    gene_map = {}  # protein_name -> {'gene': str, 'uniprot': str}

    if len(mapped) > 0:
        for _, row in mapped.iterrows():
            pname = row['Query_original']
            genes = str(row.get('Gene Names', ''))
            entry = str(row.get('Entry', ''))
            if genes and genes != 'nan' and pname not in gene_map:
                primary_gene = genes.split()[0]  # first gene symbol
                gene_map[pname] = {'gene': primary_gene, 'uniprot': entry}

    canonical_names = canonical_df['Protein.names'].tolist()
    mapped_count = sum(1 for n in canonical_names if n in gene_map)
    print(f"  From mapped_proteins.csv: {mapped_count} canonical proteins have gene symbols")

    # Try regex extraction for unmapped canonical proteins
    unmapped_names = [n for n in canonical_names if n not in gene_map]
    regex_recovered = 0
    for name in unmapped_names:
        symbol = extract_gene_symbol_from_name(name)
        if symbol:
            gene_map[name] = {'gene': symbol, 'uniprot': ''}
            regex_recovered += 1

    print(f"  Regex extraction recovered: {regex_recovered} additional symbols")

    # Try gProfiler g:Convert for remaining unmapped
    still_unmapped = [n for n in canonical_names if n not in gene_map]
    if still_unmapped:
        print(f"  Attempting gProfiler g:Convert for {len(still_unmapped)} remaining...")
        gp = GProfiler(return_dataframe=True)
        # Extract the "primary name" (before first parenthesis) for query
        query_names = []
        for name in still_unmapped:
            m = re.match(r'^([^(]+)', name)
            clean = m.group(1).strip() if m else name.strip()
            query_names.append(clean)

        try:
            # g:Convert with organism filter
            conv_result = gp.convert(
                organism='hsapiens',
                query=query_names,
                target_namespace='ENSG',
            )
            if conv_result is not None and len(conv_result) > 0:
                gp_recovered = 0
                for _, row in conv_result.iterrows():
                    incoming = row.get('incoming', '')
                    name_col = row.get('name', '')
                    # Find original protein name that matches this query
                    for orig_name, clean_name in zip(still_unmapped, query_names):
                        if clean_name == incoming and orig_name not in gene_map:
                            if name_col and str(name_col) != 'nan':
                                gene_map[orig_name] = {'gene': name_col, 'uniprot': ''}
                                gp_recovered += 1
                                break
                print(f"  gProfiler g:Convert recovered: {gp_recovered} additional symbols")
        except Exception as e:
            print(f"  ⚠ gProfiler g:Convert failed: {e}")

    total_mapped = sum(1 for n in canonical_names if n in gene_map)
    print(f"  Total canonical proteins with gene symbols: {total_mapped}/{len(canonical_names)}")

    # Save the gene map
    gene_map_rows = []
    for pname, info in gene_map.items():
        gene_map_rows.append({
            'protein_name': pname,
            'gene_symbol': info['gene'],
            'uniprot_id': info['uniprot'],
        })
    gene_map_df = pd.DataFrame(gene_map_rows)
    gene_map_df.to_csv(OUT_GENE_MAP, index=False)
    print(f"  → Saved: {OUT_GENE_MAP}")

    # ── 4. Log2 transform ─────────────────────────────────────────────────
    print("\n[4/7] Log2-transforming raw intensities...")

    data_cols = [c for c in canonical_df.columns if c in SAMPLE_COLS]
    canonical_log2 = canonical_df.copy()
    for col in data_cols:
        vals = canonical_log2[col]
        # Only transform non-NA positive values
        mask = vals.notna() & (vals > 0)
        canonical_log2.loc[mask, col] = np.log2(vals[mask])
        # Set zero/negative to NA (shouldn't happen with MS data but safety)
        canonical_log2.loc[vals.notna() & (vals <= 0), col] = np.nan

    print(f"  Value range after log2: "
          f"{canonical_log2[data_cols].min().min():.1f} — "
          f"{canonical_log2[data_cols].max().max():.1f}")

    # ── 5. Deduplicate ────────────────────────────────────────────────────
    print("\n[5/7] Deduplicating protein entries...")
    original_count = len(canonical_log2)

    # Add gene symbol column
    canonical_log2['gene_symbol'] = canonical_log2['Protein.names'].map(
        lambda n: gene_map.get(n, {}).get('gene', None)
    )

    # For proteins with the same gene symbol, keep the one with least missing data
    canonical_log2['missing_count'] = canonical_log2[data_cols].isna().sum(axis=1)

    # Group by gene symbol (for mapped proteins) and by name (for unmapped)
    canonical_log2['dedup_key'] = canonical_log2.apply(
        lambda r: r['gene_symbol'] if r['gene_symbol'] else r['Protein.names'],
        axis=1
    )
    canonical_log2 = (canonical_log2
                      .sort_values('missing_count')
                      .drop_duplicates(subset=['dedup_key'], keep='first')
                      .drop(columns=['missing_count', 'dedup_key'])
                      .reset_index(drop=True))

    print(f"  Before: {original_count} → After: {len(canonical_log2)} "
          f"(removed {original_count - len(canonical_log2)} duplicates)")

    # Save log2-transformed canonical data
    canonical_log2.to_csv(OUT_CANONICAL, index=False)
    print(f"  → Saved: {OUT_CANONICAL}")

    # ── 6. Pathway mapping via gProfiler ──────────────────────────────────
    print("\n[6/7] Mapping gene symbols → pathways via gProfiler...")

    # Collect all unique gene symbols from our canonical proteins
    all_genes = [g for g in canonical_log2['gene_symbol'].dropna().unique()
                 if isinstance(g, str) and len(g) >= 2]
    print(f"  Querying gProfiler with {len(all_genes)} unique gene symbols...")

    gp = GProfiler(return_dataframe=True)
    try:
        enrichment = gp.profile(
            organism='hsapiens',
            query=list(all_genes),
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC', 'WP'],
            no_evidences=False,       # We need the intersections
            no_iea=False,             # Include electronically-annotated GO
            significance_threshold_method='fdr',
            user_threshold=1.0,       # Get ALL pathways, we'll filter ourselves
        )
    except Exception as e:
        print(f"  ⚠ gProfiler query failed: {e}")
        enrichment = pd.DataFrame()

    if len(enrichment) > 0:
        print(f"  Raw results: {len(enrichment)} pathway annotations")

        # Build pathway gene sets
        pathway_sets = {}
        for _, row in enrichment.iterrows():
            pw_id = row['native']
            pw_name = row['name']
            source = row['source']
            # 'intersections' contains the genes from our query that are in this pathway
            genes_in_pw = row.get('intersections', [])
            if isinstance(genes_in_pw, str):
                genes_in_pw = [g.strip() for g in genes_in_pw.split(',') if g.strip()]

            n_genes = len(genes_in_pw)
            if MIN_PATHWAY_PROTEINS <= n_genes <= MAX_PATHWAY_PROTEINS:
                if pw_id not in pathway_sets:
                    pathway_sets[pw_id] = {
                        'name': pw_name,
                        'source': source,
                        'genes': genes_in_pw,
                        'size_total': int(row.get('term_size', 0)),
                        'size_matched': n_genes,
                        'p_value': float(row.get('p_value', 1.0)),
                    }

        print(f"  Pathways retained (size {MIN_PATHWAY_PROTEINS}–{MAX_PATHWAY_PROTEINS}): "
              f"{len(pathway_sets)}")

        # Count per source
        source_counts = {}
        for pw in pathway_sets.values():
            src = pw['source']
            source_counts[src] = source_counts.get(src, 0) + 1
        for src, cnt in sorted(source_counts.items()):
            print(f"    {src}: {cnt} pathways")
    else:
        pathway_sets = {}
        print("  ⚠ No pathway results returned")

    # Save pathway gene sets
    with open(OUT_PATHWAYS, 'w') as f:
        json.dump(pathway_sets, f, indent=2)
    print(f"  → Saved: {OUT_PATHWAYS}")

    # ── 7. Summary ────────────────────────────────────────────────────────
    print("\n[7/7] Phase 2 Summary")
    print("─" * 50)
    print(f"  Ig clonotypes saved:          {len(ig_df)}")
    print(f"  Canonical proteins (log2):    {len(canonical_log2)}")
    print(f"  With gene symbols:            {canonical_log2['gene_symbol'].notna().sum()}")
    print(f"  Unique gene symbols:          {len(all_genes)}")
    print(f"  Pathways mapped:              {len(pathway_sets)}")
    print("=" * 70)

    return {
        'canonical_log2': canonical_log2,
        'pathway_sets': pathway_sets,
        'gene_map': gene_map,
        'n_ig': len(ig_df),
        'n_canonical': len(canonical_log2),
    }


if __name__ == '__main__':
    run_phase2()
