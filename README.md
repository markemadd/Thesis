# AD-CSF Proteomics Pipeline

**Pathway-Level Proteomic Biomarker Discovery in Alzheimer's Disease Cerebrospinal Fluid**

A computational pipeline that identifies stable pathway-level biomarkers distinguishing Alzheimer's disease (AD) patients from healthy controls using CSF proteomics data. The pipeline reduces ~2,500 pathway activity scores to a compact panel of 13 discriminative biomarkers through a 4-stage dimensionality reduction framework.

---

## Key Results

| Metric | Value |
|---|---|
| Input proteins | 5,410 (961 canonical, 4,449 Ig clonotypes) |
| Samples | 30 (20 AD, 10 Control) |
| Pathways scored | 2,585 |
| Stable biomarker pathways | **13** |
| LOOCV Accuracy | **96.7%** |
| Permutation test p-value | **0.001** |

The 13 stable pathways converge on **four biological themes** consistent with AD literature:
1. 🧠 **Energy/Metabolic Crisis** — ketone biosynthesis ↑, glycerolipid ↑
2. 🩸 **Blood-Brain Barrier Collapse** — basement membrane ↓, angiogenesis ↑
3. 🛡️ **Immune Dysregulation** — T cell proliferation ↓, complement regulation ↓
4. 🔬 **Cellular Breakdown** — superoxide removal ↓, membrane trafficking ↑

---

## Pipeline Architecture

```
Raw CSF Proteomics Data (5,410 proteins × 30 samples)
    │
    ├── Ig Clonotypes (4,449) ──────→ [Future: Immune Repertoire Analysis]
    │
    └── Canonical Proteins (961)
         │
         ├── Phase 1: Preprocessing
         │   ├── Deduplication → 377 proteins
         │   ├── Log₂ transformation
         │   └── Gene symbol mapping (UniProt API)
         │
         ├── Phase 2: Pathway Mapping
         │   └── gProfiler → GO, KEGG, Reactome, WikiPathways → 2,585 pathways
         │
         ├── Phase 3: Pathway Scoring
         │   └── Weighted ssGSEA (DisGeNET + STRING PPI weights)
         │
         ├── Phase 5: Dimensionality Reduction
         │   ├── Stage 1: De-redundancy (Jaccard + correlation clustering)
         │   ├── Stage 2: Variance filter (IQR-based)
         │   ├── Stage 3: Stability selection (Elastic Net × 200 bootstraps)
         │   └── Stage 4: Validation (LOOCV + nested LOOCV + permutation test)
         │
         ├── Differential Analysis (Mann-Whitney U + FDR)
         │
         └── POC Classifier (L2 Logistic Regression, 13 pathway features)
```

---

## Project Structure

```
├── pipeline/
│   ├── config.py                  # All parameters & paths
│   ├── phase1_imputation.py       # Tiered imputation (MissForest + QRILC)
│   ├── phase2_pathway_mapping.py  # Protein → gene → pathway mapping
│   ├── phase3_pathway_scoring.py  # Weighted ssGSEA scoring
│   ├── phase5_dim_reduction.py    # 4-stage reduction + nested LOOCV
│   ├── run_all.py                 # Pipeline orchestrator
│   └── utils.py                   # Shared utilities
├── eda.py                         # 6-figure exploratory data analysis
├── differential_analysis.py       # Mann-Whitney U + volcano plots
├── poc_classifier.py              # Proof-of-concept AD classifier
├── run_imputed_pipeline.py        # Imputation experiment runner
├── Documentation/
│   └── normalized df-1.csv        # Raw input data
├── output/                        # Baseline results
├── output_imputed/                # Imputed experiment results
└── model/                         # Saved POC classifier
```

---

## Installation

### Requirements

- Python ≥ 3.10
- Core dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Optional (for API-based features)

```bash
pip install requests  # For UniProt, STRING, DisGeNET API calls
```

---

## Usage

### Run the Full Pipeline

```bash
# Run phases sequentially (2 → 3 → 5)
python3 -m pipeline.run_all --phases 2 3 5
```

### Run Individual Phases

```bash
# Phase 2: Pathway mapping
python3 -m pipeline.phase2_pathway_mapping

# Phase 3: Pathway scoring (ssGSEA)
python3 -m pipeline.phase3_pathway_scoring

# Phase 5: Dimensionality reduction + validation
python3 -m pipeline.phase5_dim_reduction
```

### Imputation Experiment

```bash
# Run tiered imputation (MissForest + QRILC)
python3 -m pipeline.phase1_imputation

# Re-run pipeline on imputed data
python3 run_imputed_pipeline.py
```

### EDA & Differential Analysis

```bash
python3 eda.py                    # Generates 6 figures in output/
python3 differential_analysis.py  # Mann-Whitney U tests + visualizations
```

### POC Classifier

```bash
# Train and save model
python3 poc_classifier.py --train

# View model summary
python3 poc_classifier.py --info

# Predict on new samples
python3 poc_classifier.py --predict new_patients.csv

# Predict + evaluate against known labels
python3 poc_classifier.py --predict validation.csv --label-col diagnosis
```

---

## Methods

### Pathway Scoring
**Weighted ssGSEA** — single-sample Gene Set Enrichment Analysis enhanced with protein importance weights from DisGeNET (gene-disease association) and STRING (PPI network centrality).

### Dimensionality Reduction
A 4-stage funnel designed for small-n, high-p proteomics:

| Stage | Method | Reference |
|---|---|---|
| De-redundancy | Jaccard + Pearson clustering (t=0.8) | [Merico et al., 2010](https://doi.org/10.1371/journal.pone.0013984) |
| Variance filter | IQR < 5th percentile removed | Standard practice |
| Stability selection | Elastic Net × 200 bootstraps (≥60% threshold) | [Meinshausen & Bühlmann, 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x) |
| Validation | LOOCV + nested LOOCV + 1000× permutation test | [Ambroise & McLachlan, 2002](https://doi.org/10.1073/pnas.102102699) |

### Imputation Strategy
Tiered approach based on missingness mechanism:
- **Tier 1 (≤50% missing):** MissForest — iterative RF imputation for MAR data
- **Tier 2 (50–80%):** QRILC — left-censored imputation for MNAR data  
- **Tier 3 (>80%):** No imputation — too sparse

### Overfitting Control
**Nested LOOCV** embeds feature selection inside each CV fold, eliminating information leakage from pre-selecting features on the full dataset.

---

## Key Configuration

All parameters are centralized in `pipeline/config.py`:

```python
STABILITY_N_BOOTSTRAP = 200       # Bootstrap iterations
STABILITY_THRESHOLD = 0.60        # Minimum selection rate
JACCARD_THRESHOLD = 0.8           # De-redundancy cutoff
PERMUTATION_N = 1000              # Permutation test shuffles
ELASTIC_NET_L1_RATIOS = [0.5, 0.7, 0.9]
```

---

## References

- Subramanian et al. (2005). Gene set enrichment analysis. *PNAS*, 102(43), 15545–15550
- Barbie et al. (2009). ssGSEA method. *Nature*, 462, 108–112
- Meinshausen & Bühlmann (2010). Stability selection. *JRSS-B*, 72(4), 417–473
- Stekhoven & Bühlmann (2012). MissForest. *Bioinformatics*, 28(1), 112–118
- Lazar et al. (2016). QRILC for left-censored data. *J Proteome Res*, 15(4), 1116–1125
- Gate et al. (2020). T cells in AD CSF. *Nature*, 577, 399–404
- Hong et al. (2016). Complement-mediated synaptic pruning. *Science*, 352, 712–716

---

## License

This project is part of a bachelor's thesis.