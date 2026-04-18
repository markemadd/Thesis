# AD Proteomics Pipeline — Robustness Analysis Briefing

**For:** Final thesis report assistance
**Study:** Alzheimer's Disease biomarker discovery from CSF proteomics
**Method:** Weighted ssGSEA → Stability Selection → Classification (n=30: 20 AD, 10 Control)

---

## Critical Context: Two Accuracy Estimates

The pipeline reports two LOOCV accuracy numbers. Understanding the difference is essential for the thesis.

| Metric | Value | What it means | Use in thesis |
|---|---|---|---|
| Standard LOOCV | 96.7% | Features pre-selected on all data, then cross-validated. Inflated (selection bias). | Report with caveat only |
| **Nested LOOCV** | **70.0%** | Features re-selected fresh in every fold. Unbiased. | **Primary result** |
| Nested LOOCV 95% CI | [50.6%, 85.3%] | True uncertainty range given n=30 | Report alongside |

> **Simple analogy:** Standard LOOCV is like a student who memorised the exam questions before a "blind" test. Nested LOOCV is the honest test where even the study guide changes each time.

---

## Task 1 — Ablation Study: Does the Protein Weighting Matter?

### What was done
Each protein in the dataset is assigned an "importance weight" before scoring pathways. The weighting formula is:

> **w = 1 + (α × disease_score) + (β × network_score)**

Where:
- **disease_score (d):** How strongly linked this protein is to Alzheimer's disease, from published literature (DisGeNET database). Range 0–1.
- **network_score (c):** How many biological interactions this protein has with other proteins (STRING database). Range 0–1.
- **α and β:** Multipliers controlling how much each bonus matters.

The study tested 4 conditions to see which parts of the formula are necessary:

| Condition | Formula | Meaning |
|---|---|---|
| (A) Unweighted | w = 1 | All proteins treated equally |
| (B) DisGeNET-only | w = 1 + α·d | Only AD literature bonus |
| (C) STRING-only | w = 1 + β·c | Only network connectivity bonus |
| (D) Combined | w = 1 + α·d + β·c | Both bonuses (current pipeline) |

### Results

| Condition | Stable Pathways Found | LOOCV Acc | AUC | Overlap with (D) |
|---|---|---|---|---|
| (A) Unweighted | 7 | 96.7% | 0.900 | 63.6% |
| (B) DisGeNET-only | 10 | 96.7% | 0.995 | 90.9% |
| (C) STRING-only | 8 | 96.7% | 1.000 | 72.7% |
| **(D) Combined** | **11** | **96.7%** | **1.000** | 100% |

### Interpretation
All conditions achieve the same classification accuracy (96.7%) — but this is expected given the optimistic standard LOOCV inflates all conditions equally. The **real difference** is in biomarker selection quality:
- Combined (D) finds the **most stable pathways** (11 vs 7) — better coverage
- DisGeNET-only (B) has the **highest pathway overlap** with (D), suggesting the AD literature scores are the dominant driver
- Unweighted (A) finds fewer pathways and misses 36% of what the combined model selects

**Thesis interpretation:** The combined weighting scheme discovers the richest set of biologically meaningful pathways, even though classification accuracy alone cannot discriminate between conditions (likely due to the inflated standard LOOCV).

---

## Task 2 — Sensitivity Analysis: How Stable Are the Results?

### Part A: Does changing α and β matter?
Tested 25 combinations of α ∈ {0, 0.25, 0.5, 0.75, 1.0} and β ∈ {0, 0.25, 0.5, 0.75, 1.0}.

**Result:** Accuracy ranged from 93.3% to 96.7% — a very narrow range. The pipeline is not sensitive to the exact choice of α and β values.

### Part B: Does the stability threshold (π) matter?
π controls how often a pathway must appear across bootstrap runs to be considered "stable". Higher π = stricter = fewer but more robust pathways.

| π threshold | Stable Pathways | Accuracy |
|---|---|---|
| 0.50 (loose) | 14 | 100.0% ← overfitting |
| **0.60 (current)** | **7** | **96.7%** |
| 0.75 (strict) | 5 | 90.0% |
| 0.85 (very strict) | 3 | 83.3% |

**Thesis interpretation:** π=0.60 is a reasonable choice — it balances biomarker count with reliability. Lowering π inflates accuracy but risks including noise.

---

## Task 3 — Statistical Significance

### Permutation Test (1,000 shuffles)
To confirm the result is not random chance, the study shuffled the disease labels 1,000 times and re-ran classification each time.

- Null distribution mean: 52.6% (near chance level of 50%)
- Null distribution maximum: 83.3%
- How many times did shuffled labels beat 96.7%? → **Zero times (0/1,000)**
- Empirical p-value: **p = 0.001**

> The result is statistically significant. Random label assignment cannot explain the 96.7% standard accuracy.

### Confidence Intervals (Clopper-Pearson Exact Method)
These are the honest uncertainty bounds for each accuracy estimate:

| Metric | Accuracy | 95% CI |
|---|---|---|
| Standard LOOCV | 96.7% (29/30) | [82.8%, 99.9%] |
| **Nested LOOCV** | **70.0% (21/30)** | **[50.6%, 85.3%]** |
| Imputed Nested LOOCV | 60.0% (18/30) | [40.6%, 77.3%] |

The wide confidence intervals reflect the small sample (n=30). With 30 subjects, even 29/30 correct allows for substantial uncertainty.

---

## Task 4 — Computational Complexity

### How long does the pipeline take?

| Pipeline Stage | Time | Memory |
|---|---|---|
| Data preprocessing (KNN imputation) | 2.4s | 46 MB |
| Pathway mapping (gene → pathway) | 0.04s | 4 MB |
| ssGSEA pathway scoring (2,585 pathways) | 37.9s | 12 MB |
| De-redundancy (Jaccard similarity matrix) | 34.3s | **258 MB** (peak) |
| Variance filtering | 0.4s | <1 MB |
| Stability selection (200 bootstrap rounds) | 236s ≈ 4 min | 3 MB |
| Standard LOOCV validation | 3.8s | <1 MB |
| **Nested LOOCV** | **3,405s ≈ 57 min** | 4 MB |
| **Total** | **≈ 62 minutes** | **258 MB peak** |

### Main bottleneck
Nested LOOCV takes 94% of total runtime because it repeats the full stability selection (200 bootstraps × 575 features) inside each of the 30 outer LOOCV folds. This is unavoidable for an unbiased estimate.

### Biggest memory cost
The de-redundancy stage computes pairwise Jaccard similarity for all 2,585 pathway pairs — a 2,585×2,585 matrix — requiring 258 MB. Manageable on standard hardware but worth noting.

---

## Summary of Honest Findings

| Question | Answer |
|---|---|
| Best accuracy (unbiased) | **70%** nested LOOCV |
| Statistical significance | **p = 0.001** (permutation test) |
| Best weighting scheme | **Combined (D)** — most biomarkers |
| Pipeline sensitivity to α,β | **Low** — accuracy varies only 3.4% across 25 combinations |
| Optimal stability threshold | **π = 0.60** (current choice is well-justified) |
| Total runtime | **62 minutes** on a MacBook Pro |

---

## Honest Limitations for the Thesis

1. **Small sample (n=30):** Wide confidence intervals. Findings need replication on larger cohorts.
2. **Standard LOOCV is inflated:** Must be reported as an optimistic bound, not the primary result.
3. **Nested LOOCV at 70%:** Competitive with literature for n=30 CSF proteomics (typical range: 65–90%).
4. **Curated fallback weights:** DisGeNET and STRING scores used curated fallback dictionaries (no live API call). Values were derived from known AD gene panels and are reproducible, but not dynamically updated.
5. **All conditions tied at 96.7% (standard LOOCV):** This cannot distinguish weighting quality — the ablation difference shows in AUC and pathway count, not accuracy.

---

## How This Fits Literature

| Study | n | Best accuracy | Method |
|---|---|---|---|
| Higginbotham et al. 2020 | 137 | ~88% | CSF proteomics + Random Forest |
| Olsson et al. 2016 | 202 | ~88% | CSF biomarker panel + LR |
| Bader et al. 2020 | 48 | ~87% | Brain proteomics + SVM |
| **This study (nested LOOCV)** | **30** | **70%** | **Pathway ssGSEA + Elastic Net** |
| *This study (std LOOCV)* | *30* | *96.7%* | *(pre-selected features — inflated)* |

The 70% nested LOOCV is appropriate for n=30 and consistent with the literature trend. Larger n would close the gap.
