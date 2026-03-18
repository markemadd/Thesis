# AD Proteomics Pipeline: Missing Data Imputation & Pathway-Based Dimensionality Reduction

## Full Implementation Plan for Coding Agent

---

## 1. Project Overview

**Dataset**: Normalized plasma mass spectrometry proteomics data (`normalized df-1.csv`)
- 5,410 total proteins × 30 samples (20 AD patients P1–P20, 10 controls C1–C10)
- 85.2% overall missingness
- Working set: 1,765 proteins with ≥10% detection rate
- Column 0 = `Protein.names`, columns 1–20 = patients, columns 21–30 = controls

**Goal**: Build a complete pipeline with two major phases:
1. **Phase 1** — Mechanism-aware missing data imputation using a 3-test classification framework
2. **Phase 2** — Pathway-based dimensionality reduction using STRING/KEGG/Reactome databases with network-derived protein importance weights

---

## 2. Phase 1: Missing Data Imputation

### 2.1 Data Loading & Preprocessing

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("normalized df-1.csv")

# Separate metadata
protein_names = df['Protein.names']
patient_cols = [f'P{i}' for i in range(1, 21)]
control_cols = [f'C{i}' for i in range(1, 11)]
sample_cols = patient_cols + control_cols
data = df[sample_cols].copy()

# Create group labels
group_labels = ['AD'] * 20 + ['Control'] * 10

# Filter to working set: proteins with ≥10% detection (detected in ≥3 of 30 samples)
detection_counts = data.notna().sum(axis=1)
working_mask = detection_counts >= 3  # ≥10% of 30
working_data = data[working_mask].copy()
working_names = protein_names[working_mask].reset_index(drop=True)

# Log10 transform all positive values (leave NaN as NaN)
log_data = working_data.apply(lambda x: np.log10(x.where(x > 0)))
```

### 2.2 Compute Global Intensity Thresholds

```python
all_values = log_data.values.flatten()
all_values = all_values[~np.isnan(all_values)]
global_p25 = np.percentile(all_values, 25)
global_p75 = np.percentile(all_values, 75)
```

### 2.3 Three-Test Classification Framework

Each protein with missing values is classified into one of four missingness mechanisms using three sequential tests.

#### Test A — Intensity Profile (frac_low)

For each protein, compute the fraction of its detected values that fall below `global_p25`:

```python
def compute_frac_low(row, threshold):
    detected = row.dropna()
    if len(detected) == 0:
        return np.nan
    return (detected < threshold).sum() / len(detected)

frac_low = log_data.apply(lambda row: compute_frac_low(row, global_p25), axis=1)
```

**Interpretation**: `frac_low > 0.5` → protein is near instrument floor → likely MNAR-LOD

#### Test B — QC-Missingness Correlation (r_miss_qc)

**Step B1**: Build a sample QC score from 5 anchor proteins (high-abundance, universally detected, degradation-stable). Select the 5 proteins with 100% detection and the highest median intensity:

```python
full_detection = log_data.notna().all(axis=0)  # This checks columns; we need rows
# Actually: find proteins detected in ALL 30 samples
fully_detected_mask = log_data.notna().sum(axis=1) == 30
fully_detected_data = log_data[fully_detected_mask]
median_intensities = fully_detected_data.median(axis=1)
top5_anchors = median_intensities.nlargest(5).index

# Compute per-sample QC score = mean z-score of anchor proteins
anchor_vals = log_data.loc[top5_anchors]
anchor_means = anchor_vals.mean(axis=1)
anchor_stds = anchor_vals.std(axis=1)
anchor_z = anchor_vals.subtract(anchor_means, axis=0).divide(anchor_stds, axis=0)
sample_qc = anchor_z.mean(axis=0)  # One QC score per sample
```

**Step B2**: For each protein with missing values, compute point-biserial correlation between missingness indicator (1=missing, 0=detected) and sample QC score:

```python
from scipy.stats import pointbiserialr

def compute_r_miss_qc(row, qc_scores):
    missing_indicator = row.isna().astype(int)
    if missing_indicator.sum() == 0 or missing_indicator.sum() == len(row):
        return 0.0
    r, p = pointbiserialr(missing_indicator, qc_scores)
    return r

r_miss_qc = log_data.apply(lambda row: compute_r_miss_qc(row, sample_qc), axis=1)
```

**Interpretation**: `r_miss_qc < -0.35` → protein goes missing in low-QC samples → likely MNAR-DEG (degradation)

#### Test C — Group-Biased Missingness (Fisher's Exact Test)

```python
from scipy.stats import fisher_exact

def compute_fisher(row, patient_cols, control_cols):
    p_detected = row[patient_cols].notna().sum()
    p_missing = row[patient_cols].isna().sum()
    c_detected = row[control_cols].notna().sum()
    c_missing = row[control_cols].isna().sum()
    table = [[p_detected, c_detected], [p_missing, c_missing]]
    _, pval = fisher_exact(table)
    return pval

fisher_p = log_data.apply(lambda row: compute_fisher(row, patient_cols, control_cols), axis=1)

# Also compute group exclusivity flags
patient_det = log_data[patient_cols].notna().sum(axis=1)
control_det = log_data[control_cols].notna().sum(axis=1)
patient_exclusive = (patient_det >= 3) & (control_det == 0)
control_exclusive = (control_det >= 3) & (patient_det == 0)
```

### 2.4 Decision Tree — Assign Missingness Mechanism

Apply this decision tree **in order** for each protein with any missing values:

```python
def classify_missingness(idx):
    det_count = detection_counts_working[idx]
    if det_count == 30:
        return 'OBSERVED'

    # Test C first: check for biological exclusivity
    if (patient_exclusive[idx] or control_exclusive[idx]) and frac_low[idx] < 0.3:
        return 'MNAR-BIO'
    if fisher_p[idx] < 0.05 and frac_low[idx] < 0.3:
        return 'MNAR-BIO'

    # Test A: check for LOD
    if frac_low[idx] > 0.5:
        return 'MNAR-LOD'

    # Test B: check for degradation
    if r_miss_qc[idx] < -0.35:
        return 'MNAR-DEG'

    # Default
    return 'MCAR'

mechanism = pd.Series({idx: classify_missingness(idx) for idx in log_data.index})
```

### 2.5 Mechanism-Matched Imputation

```python
from sklearn.impute import KNNImputer

imputed_data = log_data.copy()
flags = pd.DataFrame(0, index=log_data.index, columns=['lod_imputed','deg_imputed','bio_binary'])

for idx in log_data.index:
    mech = mechanism[idx]
    row = log_data.loc[idx]
    missing_mask = row.isna()

    if mech == 'OBSERVED':
        continue

    elif mech == 'MNAR-LOD':
        # Group-aware LOD/sqrt(2)
        patient_vals = row[patient_cols].dropna()
        control_vals = row[control_cols].dropna()
        lod_patient = patient_vals.min() / np.sqrt(2) if len(patient_vals) > 0 else row.dropna().min() / np.sqrt(2)
        lod_control = control_vals.min() / np.sqrt(2) if len(control_vals) > 0 else row.dropna().min() / np.sqrt(2)
        for col in patient_cols:
            if pd.isna(imputed_data.loc[idx, col]):
                imputed_data.loc[idx, col] = lod_patient
        for col in control_cols:
            if pd.isna(imputed_data.loc[idx, col]):
                imputed_data.loc[idx, col] = lod_control
        flags.loc[idx, 'lod_imputed'] = 1

    elif mech == 'MNAR-DEG':
        # Group median × 0.7
        patient_vals = row[patient_cols].dropna()
        control_vals = row[control_cols].dropna()
        med_patient = patient_vals.median() * 0.7 if len(patient_vals) > 0 else row.dropna().median() * 0.7
        med_control = control_vals.median() * 0.7 if len(control_vals) > 0 else row.dropna().median() * 0.7
        for col in patient_cols:
            if pd.isna(imputed_data.loc[idx, col]):
                imputed_data.loc[idx, col] = med_patient
        for col in control_cols:
            if pd.isna(imputed_data.loc[idx, col]):
                imputed_data.loc[idx, col] = med_control
        flags.loc[idx, 'deg_imputed'] = 1

    elif mech == 'MNAR-BIO':
        # Binary indicator only — do NOT impute quantitative values
        flags.loc[idx, 'bio_binary'] = 1
        # Leave NaN as NaN; create separate binary matrix later

    elif mech == 'MCAR':
        pass  # Handle with KNN below

# KNN imputation for MCAR proteins (batch)
mcar_mask = mechanism == 'MCAR'
if mcar_mask.any():
    mcar_data = imputed_data.loc[mcar_mask].T  # KNN expects samples×features
    imputer = KNNImputer(n_neighbors=5)
    mcar_imputed = imputer.fit_transform(mcar_data)
    imputed_data.loc[mcar_mask] = pd.DataFrame(mcar_imputed, index=mcar_data.columns, columns=mcar_data.index).T

# Create binary detection matrix for MNAR-BIO proteins
bio_binary_matrix = log_data.notna().astype(int)
```

### 2.6 Sample QC Flagging

```python
# Flag samples with QC < -1.5 for downstream weight reduction
sample_qc_flags = (sample_qc < -1.5).astype(int)
sample_weights = pd.Series(1.0, index=sample_cols)
sample_weights[sample_qc < -1.5] = 0.7
```

### 2.7 Output of Phase 1

Save the following artifacts:
1. `imputed_matrix.csv` — fully imputed log10 quantitative matrix (MNAR-BIO proteins remain NaN)
2. `imputation_flags.csv` — per-protein flags (lod_imputed, deg_imputed, bio_binary)
3. `mechanism_labels.csv` — per-protein mechanism classification
4. `binary_detection_matrix.csv` — 0/1 detection matrix for all proteins
5. `sample_qc_scores.csv` — per-sample QC scores and weights

---

## 3. Phase 2: Pathway-Based Dimensionality Reduction

### 3.1 Overview

Transform the 1,765-protein feature space into a ~50–150 pathway activity score space. This involves:
1. Mapping proteins to biological pathways using STRING, KEGG, and Reactome
2. Computing protein importance weights within each pathway using network topology
3. Computing weighted pathway activity scores per sample
4. Using the pathway score matrix for downstream UMAP/PCA

### 3.2 Protein-to-Pathway Mapping

Use three complementary databases. The protein names in the CSV are human-readable names; you will need to map them to standard IDs first.

#### Step 1: Map protein names to UniProt/STRING IDs

```python
import requests, time

def map_to_string_ids(protein_names, species=9606):
    """Map protein names to STRING identifiers via the STRING API."""
    string_ids = {}
    batch_size = 100  # STRING API limit
    name_list = list(protein_names)

    for i in range(0, len(name_list), batch_size):
        batch = name_list[i:i+batch_size]
        identifiers = '%0d'.join(batch)
        url = f"https://string-db.org/api/json/get_string_ids"
        params = {'identifiers': identifiers, 'species': species, 'limit': 1}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            for item in response.json():
                string_ids[item['queryItem']] = {
                    'stringId': item['stringId'],
                    'preferredName': item['preferredName']
                }
        time.sleep(1)  # Rate limiting
    return string_ids
```

#### Step 2: Get pathway enrichment from STRING

```python
def get_string_enrichment(string_ids_list, species=9606):
    """Get KEGG and Reactome pathway enrichment via STRING API."""
    identifiers = '%0d'.join(string_ids_list)
    url = "https://string-db.org/api/json/enrichment"
    params = {'identifiers': identifiers, 'species': species}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()
        pathways = [r for r in results if r['category'] in
                    ['KEGG', 'Reactome', 'Process', 'Component']]
        return pathways
    return []
```

#### Step 3: Get full pathway membership from KEGG REST API

```python
def get_kegg_pathway_genes(pathway_id):
    """Get all genes in a KEGG pathway (e.g., 'hsa04610' for complement/coagulation)."""
    url = f"https://rest.kegg.jp/link/hsa/{pathway_id}"
    response = requests.get(url)
    genes = []
    if response.status_code == 200:
        for line in response.text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) == 2:
                genes.append(parts[1].replace('hsa:', ''))
    return genes

def get_all_kegg_pathways():
    """Get list of all human KEGG pathways."""
    url = "https://rest.kegg.jp/list/pathway/hsa"
    response = requests.get(url)
    pathways = {}
    for line in response.text.strip().split('\n'):
        parts = line.split('\t')
        pathways[parts[0].replace('path:', '')] = parts[1]
    return pathways
```

#### Step 4: Get pathway membership from Reactome REST API

```python
def get_reactome_pathway_proteins(pathway_id):
    """Get all proteins in a Reactome pathway."""
    url = f"https://reactome.org/ContentService/data/participants/{pathway_id}"
    response = requests.get(url)
    proteins = []
    if response.status_code == 200:
        for entity in response.json():
            if 'displayName' in entity:
                proteins.append(entity['displayName'])
    return proteins

def get_reactome_pathways_for_protein(uniprot_id):
    """Get all Reactome pathways containing a protein."""
    url = f"https://reactome.org/ContentService/data/mapping/UniProt/{uniprot_id}/pathways"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []
```

#### Step 5: Build the protein-to-pathway mapping table

```python
def build_pathway_protein_map(protein_names, string_id_map):
    """
    Build a comprehensive mapping: pathway_id -> list of protein indices.
    Combines STRING enrichment, KEGG membership, and Reactome membership.
    """
    pathway_map = {}  # pathway_id -> {'name': str, 'proteins': [indices], 'source': str}

    # Use STRING enrichment to identify relevant pathways
    mapped_ids = [v['stringId'] for v in string_id_map.values()]
    enrichment = get_string_enrichment(mapped_ids)

    for entry in enrichment:
        pid = entry['term']
        pathway_map[pid] = {
            'name': entry['description'],
            'source': entry['category'],
            'proteins': entry.get('inputGenes', []),
            'p_value': entry.get('p_value', 1.0)
        }

    # Supplement with KEGG full membership
    kegg_pathways = get_all_kegg_pathways()
    for pathway_id in kegg_pathways:
        genes = get_kegg_pathway_genes(pathway_id)
        # Cross-reference with our protein list
        # ... (match gene symbols to protein names in dataset)

    return pathway_map
```

### 3.3 Protein Importance Weighting Within Pathways

Each protein receives a weight **within each pathway** it belongs to. The weight is a product of four components:

#### Weight Component 1: Detection Reliability (w1)

```python
detection_rate = log_data.notna().sum(axis=1) / 30
reliability_factor = pd.Series(1.0, index=log_data.index)
reliability_factor[mechanism == 'MNAR-DEG'] = 0.7

w1 = np.sqrt(detection_rate) * reliability_factor
```

#### Weight Component 2: Pathway Specificity (w2)

Proteins appearing in many pathways carry redundant information; penalize them:

```python
# Count how many pathways each protein appears in
n_pathways_per_protein = pd.Series(0, index=log_data.index)
for pid, pinfo in pathway_map.items():
    for protein_idx in pinfo['proteins']:
        n_pathways_per_protein[protein_idx] += 1

w2 = 1.0 / np.log2(n_pathways_per_protein + 1)
```

#### Weight Component 3: Differential Signal (w3)

Up-weight proteins with evidence of group-level differences:

```python
from scipy.stats import mannwhitneyu

def compute_log2fc(row, patient_cols, control_cols):
    p_vals = row[patient_cols].dropna()
    c_vals = row[control_cols].dropna()
    if len(p_vals) < 3 or len(c_vals) < 3:
        return 0.0
    return np.abs(p_vals.mean() - c_vals.mean())  # Already log10, so difference = log2FC approx

log2fc = imputed_data.apply(lambda row: compute_log2fc(row, patient_cols, control_cols), axis=1)
# Normalize 0-1 within each pathway during score computation
```

#### Weight Component 4: Network Centrality from STRING PPI (w4) — NEW

This is the key innovation for protein importance. Use the STRING protein-protein interaction network to compute centrality measures:

```python
import networkx as nx

def build_pathway_subnetwork(pathway_proteins, string_ppi_data, min_score=400):
    """
    Build a subnetwork for proteins within a pathway using STRING PPI data.
    string_ppi_data is a DataFrame with columns: protein1, protein2, combined_score
    """
    G = nx.Graph()
    for _, row in string_ppi_data.iterrows():
        if (row['protein1'] in pathway_proteins and
            row['protein2'] in pathway_proteins and
            row['combined_score'] >= min_score):
            G.add_edge(row['protein1'], row['protein2'],
                       weight=row['combined_score'] / 1000.0)
    return G

def compute_network_centrality(G):
    """Compute multiple centrality measures for each node."""
    if len(G.nodes) == 0:
        return {}
    centrality = {}
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight='weight')
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except:
        pagerank = {n: 1.0/len(G.nodes) for n in G.nodes}

    for node in G.nodes:
        # Combined centrality: average of normalized degree, betweenness, pagerank
        centrality[node] = (degree.get(node, 0) +
                           betweenness.get(node, 0) +
                           pagerank.get(node, 0)) / 3.0
    return centrality
```

**How to get STRING PPI data:**

```python
# Option A: Download bulk file (recommended for full analysis)
# wget https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz
# Then load:
string_ppi = pd.read_csv('9606.protein.links.full.v12.0.txt.gz', sep=' ')

# Option B: Use STRING API for specific protein sets
def get_string_interactions(identifiers, species=9606):
    url = "https://string-db.org/api/json/network"
    params = {
        'identifiers': '%0d'.join(identifiers),
        'species': species,
        'required_score': 400  # Medium confidence
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else []
```

#### Combining All Weights

```python
def compute_final_weights(pathway_proteins, w1, w2, w3, w4_centrality):
    """Compute normalized final weight for each protein in a pathway."""
    weights = {}
    for prot in pathway_proteins:
        w1_val = w1.get(prot, 0.5)
        w2_val = w2.get(prot, 0.5)
        w3_val = w3.get(prot, 0.0)
        w4_val = w4_centrality.get(prot, 0.1)  # Default low if not in network
        weights[prot] = w1_val * w2_val * (1 + w3_val) * (1 + w4_val)

    # Normalize so weights sum to 1 within pathway
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights
```

### 3.4 Pathway Activity Score Computation

Two complementary methods should be implemented:

#### Method A: Custom Weighted Pathway Scores

```python
def compute_pathway_scores(imputed_data, pathway_map, final_weights, sample_weights):
    """
    Compute weighted pathway activity score per sample per pathway.
    score(pathway P, sample s) = sum(w_i * x_i_s) * sample_qc_weight_s
    """
    pathway_scores = pd.DataFrame(index=imputed_data.columns)  # samples × pathways

    for pid, pinfo in pathway_map.items():
        proteins = pinfo['proteins']
        weights = final_weights[pid]
        score = pd.Series(0.0, index=imputed_data.columns)

        for prot in proteins:
            if prot in imputed_data.index:
                w = weights.get(prot, 0.0)
                vals = imputed_data.loc[prot]
                score += w * vals.fillna(0)

        # Apply sample QC weighting
        score *= sample_weights
        pathway_scores[pinfo['name']] = score

    return pathway_scores
```

#### Method B: ssGSEA via GSEApy (recommended as complementary validation)

```python
import gseapy as gp

# Prepare gene sets in GMT-like format
gene_sets = {}
for pid, pinfo in pathway_map.items():
    gene_sets[pinfo['name']] = list(pinfo['proteins'])

# Run ssGSEA
ssgsea_results = gp.ssgsea(
    data=imputed_data,         # proteins × samples (log10 values)
    gene_sets=gene_sets,
    outdir=None,
    sample_norm_method='rank',
    min_size=5,
    max_size=500,
    permutation_num=0
)

ssgsea_scores = ssgsea_results.res2d.pivot(
    index='Name', columns='Term', values='NES'
)
```

### 3.5 Dimensionality Reduction on Pathway Scores

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# Standardize pathway scores
scaler = StandardScaler()
pathway_scaled = scaler.fit_transform(pathway_scores)

# PCA
pca = PCA(n_components=10)
pca_result = pca.fit_transform(pathway_scaled)
explained_var = pca.explained_variance_ratio_

# UMAP
reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, n_components=2, random_state=42)
umap_result = reducer.fit_transform(pathway_scaled)

# Validate: check AD vs Control separation
# Visualize with group coloring
```

### 3.6 WGCNA — Weighted Gene Co-expression Network Analysis (Third Complementary Method)

WGCNA identifies **data-driven modules** of co-expressed proteins without requiring predefined pathway annotations. This complements the pathway-based approach by discovering novel co-regulation patterns not captured in curated databases.

#### Why WGCNA for This Dataset

- Discovers co-expression modules directly from the data — no pathway database required
- Can reveal novel protein groupings specific to AD plasma proteomics
- Hub proteins within WGCNA modules can be compared against STRING-derived centrality
- Module eigengenes (ME) serve as an independent dimensionality reduction

#### Implementation with pyWGCNA

```python
import PyWGCNA

# Prepare data: samples × proteins (transposed from our proteins × samples format)
# Use only imputed quantitative proteins (exclude MNAR-BIO)
quant_mask = mechanism != 'MNAR-BIO'
wgcna_input = imputed_data.loc[quant_mask].T  # 30 samples × ~1400 proteins

# Initialize pyWGCNA
pyWGCNA_obj = PyWGCNA.WGCNA(
    name='AD_proteomics',
    species='homo sapiens',
    geneExpPath=None,   # We pass the DataFrame directly
    outputPath='wgcna_output/'
)
pyWGCNA_obj.geneExpr = PyWGCNA.GeneExp(wgcna_input)

# Step 1: Find optimal soft-thresholding power
# For proteomics, use SIGNED network (proteins up/down in same direction cluster together)
pyWGCNA_obj.findModules(
    networkType='signed',
    TOMType='signed',
    minModuleSize=10,     # Minimum proteins per module
    deepSplit=2,          # Module splitting sensitivity (0-4)
    mergeCutHeight=0.25   # Merge similar modules
)

# Step 2: Extract module eigengenes (one score per module per sample)
module_eigengenes = pyWGCNA_obj.datME  # DataFrame: 30 samples × n_modules

# Step 3: Correlate modules with AD/Control status
import scipy.stats as stats
group_numeric = pd.Series([1]*20 + [0]*10, index=wgcna_input.index)  # 1=AD, 0=Control
module_trait_cor = {}
for module in module_eigengenes.columns:
    r, p = stats.pointbiserialr(group_numeric, module_eigengenes[module])
    module_trait_cor[module] = {'r': r, 'p': p}

# Step 4: Identify hub proteins per module
# Hub = high module membership (kME) + high intramodular connectivity
hub_proteins = {}
for module_name in pyWGCNA_obj.datME.columns:
    module_color = module_name.replace('ME', '')
    module_genes = pyWGCNA_obj.datExpr.columns[
        pyWGCNA_obj.moduleColors == module_color
    ]
    # Compute kME (module eigengene-based connectivity)
    kME = wgcna_input[module_genes].corrwith(module_eigengenes[module_name])
    hub_proteins[module_color] = kME.nlargest(10).index.tolist()
```

#### WGCNA-Specific Proteomics Considerations

- **Use signed network**: Ensures proteins with opposite regulation are in different modules
- **Soft threshold**: Typically β = 6–12 for proteomics; check scale-free topology fit (R² > 0.8)
- **Missing data**: WGCNA requires complete data — run AFTER imputation (Phase 1)
- **Small sample size (n=30)**: Reduce `minModuleSize` to 10 (default 30 is for transcriptomics)
- **Validation**: Compare WGCNA modules against KEGG/Reactome pathway membership to check biological coherence

#### Integration with Pathway Scores

```python
# Combine WGCNA module eigengenes with pathway activity scores
combined_features = pd.concat([
    pathway_scores,          # From Section 3.4 (curated pathway scores)
    module_eigengenes         # From WGCNA (data-driven modules)
], axis=1)

# Run dimensionality reduction on combined features
from sklearn.decomposition import PCA
combined_scaled = StandardScaler().fit_transform(combined_features)
pca_combined = PCA(n_components=10).fit_transform(combined_scaled)
umap_combined = umap.UMAP(n_neighbors=10, min_dist=0.3).fit_transform(combined_scaled)
```

---

## 4. Detailed STRING/KEGG/Reactome Pathway Retrieval Protocol

### 4.1 Recommended AD-Relevant Pathways to Retrieve

Based on the dataset biology and AD literature, prioritize these pathways (grouped by biological theme):

**Core AD & Neurodegeneration:**

| Pathway | Database | ID | Why |
|---------|----------|-----|-----|
| Alzheimer disease | KEGG | hsa05010 | Central AD pathway: Aβ, tau, synaptic loss |
| Neurotrophin signaling | KEGG | hsa04722 | NGF/BDNF survival signaling |
| Amyloid fiber formation | Reactome | R-HSA-977225 | Aβ aggregation cascade |

**Complement & Coagulation (core paper findings):**

| Pathway | Database | ID | Why |
|---------|----------|-----|-----|
| Complement and coagulation cascades | KEGG | hsa04610 | Core finding of original paper |
| Complement cascade | Reactome | R-HSA-166658 | Extended complement (C1s patient-exclusive) |
| Hemostasis | Reactome | R-HSA-109582 | Coagulation cascade, Factor XI |

**Neuroinflammation & Immune:**

| Pathway | Database | ID | Why |
|---------|----------|-----|-----|
| Innate immune system | Reactome | R-HSA-168249 | Neuroinflammation, TREM2 |
| Cytokine signaling in immune | Reactome | R-HSA-1280215 | Inflammatory cytokines in AD |
| NF-kappa B signaling | KEGG | hsa04064 | Chronic neuroinflammation |
| NLRP3 inflammasome | Reactome | R-HSA-844615 | Inflammasome activation in AD |

**Autophagy & Lysosomal:**

| Pathway | Database | ID | Why |
|---------|----------|-----|-----|
| Autophagy | KEGG | hsa04140 | Aβ/tau clearance failure |
| Lysosome | KEGG | hsa04142 | Cystatin-C, cathepsins |
| mTOR signaling | KEGG | hsa04150 | Autophagy regulation, tau phosphorylation |

**PI3K-Akt & Survival Signaling:**

| Pathway | Database | ID | Why |
|---------|----------|-----|-----|
| PI3K-Akt signaling | KEGG | hsa04151 | Neuron survival, GSK3β → tau |
| Insulin signaling | KEGG | hsa04910 | Brain insulin resistance in AD |
| AMPK signaling | KEGG | hsa04152 | Energy sensing, mTOR regulation |

**Synaptic & Neurotransmission:**

| Pathway | Database | ID | Why |
|---------|----------|-----|-----|
| Synaptic vesicle cycle | KEGG | hsa04721 | Synaptic dysfunction |
| Long-term potentiation | KEGG | hsa04720 | Learning/memory impairment |
| Cholinergic synapse | KEGG | hsa04725 | Cholinergic deficit in AD |
| Calcium signaling | KEGG | hsa04020 | Ca²⁺ dysregulation → excitotoxicity |

**Lipid & Oxidative Stress:**

| Pathway | Database | ID | Why |
|---------|----------|-----|-----|
| Lipid metabolism | Reactome | R-HSA-556833 | ApoE/ApoF dysregulation |
| Oxidative stress response | GO | GO:0006979 | PRDX2, Ceruloplasmin |
| Iron homeostasis | Reactome | R-HSA-917937 | Ceruloplasmin, ferroptosis |
| PPAR signaling | KEGG | hsa03320 | Lipid metabolism, neuroinflammation |

**Extracellular Matrix & Other:**

| Pathway | Database | ID | Why |
|---------|----------|-----|-----|
| ECM-receptor interaction | KEGG | hsa04512 | AD ECM remodeling |
| N-Glycan biosynthesis | KEGG | hsa00510 | Glycosylation changes in AD CSF |

### 4.2 Full Pipeline to Retrieve All Proteins per Pathway

```python
# STEP 1: Get all KEGG human pathways
all_kegg = get_all_kegg_pathways()  # ~350 pathways
print(f"Found {len(all_kegg)} KEGG pathways")

# STEP 2: For each pathway, get member genes and match to dataset
pathway_protein_sets = {}
for pw_id, pw_name in all_kegg.items():
    genes = get_kegg_pathway_genes(pw_id)
    # Convert KEGG gene IDs to gene symbols using KEGG API
    url = f"https://rest.kegg.jp/conv/uniprot/{pw_id}"
    # Match to dataset protein names
    matched = [g for g in genes if g in protein_name_to_idx_map]
    if len(matched) >= 3:
        pathway_protein_sets[pw_id] = {
            'name': pw_name, 'proteins': matched, 'source': 'KEGG'}
    time.sleep(0.5)

# STEP 3: Supplement with Reactome
for uniprot_id in all_uniprot_ids:
    reactome_pathways = get_reactome_pathways_for_protein(uniprot_id)
    for pw in reactome_pathways:
        pw_id = pw['stId']
        if pw_id not in pathway_protein_sets:
            # Get full members
            members = get_reactome_pathway_proteins(pw_id)
            matched = [m for m in members if m in protein_name_to_idx_map]
            if len(matched) >= 3:
                pathway_protein_sets[pw_id] = {
                    'name': pw['displayName'], 'proteins': matched, 'source': 'Reactome'}
    time.sleep(0.3)

# STEP 4: For each pathway, pull STRING PPI subnetwork and compute centrality
for pw_id, pw_info in pathway_protein_sets.items():
    string_ids = [string_id_map[p]['stringId'] for p in pw_info['proteins']
                  if p in string_id_map]
    interactions = get_string_interactions(string_ids)
    G = nx.Graph()
    for inter in interactions:
        G.add_edge(inter['preferredName_A'], inter['preferredName_B'],
                   weight=inter['score'])
    pw_info['network'] = G
    pw_info['centrality'] = compute_network_centrality(G)
    time.sleep(1)
```

### 4.3 Protein Name Matching Strategy

The CSV uses human-readable protein names (e.g., "Ceruloplasmin", "Complement C7"). These need to be mapped to standard identifiers. Use this cascade:

1. **STRING API `get_string_ids`** — handles most common names
2. **UniProt ID mapping API** (`https://rest.uniprot.org/idmapping/`) — for ambiguous names
3. **Manual curation** — for Ig clonotypes (e.g., "IG c704_heavy") which won't map to standard pathways

```python
# UniProt mapping fallback
def uniprot_search(protein_name, species='Human'):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        'query': f'protein_name:"{protein_name}" AND organism_name:"{species}"',
        'format': 'json', 'size': 1
    }
    response = requests.get(url, params=params)
    if response.status_code == 200 and response.json()['results']:
        return response.json()['results'][0]['primaryAccession']
    return None
```

---

## 5. Edge Cases and Special Handling

### 5.1 MNAR-BIO Proteins in Pathway Scores

Proteins classified as MNAR-BIO (295 patient-exclusive, 21 control-exclusive) should contribute to pathway scores **only through binary detection**, not quantitative values:

```python
# For MNAR-BIO proteins in a pathway:
# Use binary detection (0/1) multiplied by a small fixed weight
# rather than imputed quantitative values
bio_contribution = bio_binary_matrix.loc[bio_proteins] * 0.1  # Scaled down
```

### 5.2 Transthyretin Edge Case

TTR (r_miss_qc = −0.40) has both degradation and biological signals. Flag in output:

```python
edge_case_proteins = ['Transthyretin']  # Add others as discovered
# Flag in pathway output with dual annotation
```

### 5.3 Sparse Proteins (10–50% detection)

Use the degradation-stratified two-part model:
- Binary indicator: 1=detected, 0=not detected, −1=unknown (missing in QC-flagged sample)
- Quantitative: LOD/√2 for MNAR-LOD; group-median×0.7 for MNAR-DEG; 0 for MNAR-BIO

---

## 6. Verification Plan

### 6.1 Phase 1 Verification

1. **Count check**: Verify mechanism distribution (expect ~40% MNAR-LOD, ~10% MNAR-DEG, ~15% MNAR-BIO, ~35% MCAR among proteins with missing values)
2. **Spot checks**: Verify known proteins against documentation:
   - Lumican → MNAR-LOD (frac_low=0.82)
   - ATIII → MNAR-DEG (r_miss_qc=−0.441)
   - Ceruloplasmin → MNAR-BIO (9/20 patients, 0/10 controls)
3. **No fabricated biology**: Confirm MNAR-BIO proteins retain NaN, not imputed values

### 6.2 Phase 2 Verification

1. **Pathway coverage**: ≥50% of 1,765 proteins should map to at least one pathway
2. **Complement pathway proof-of-concept**: Verify intra-pathway Spearman r ≈ 0.52
3. **AD/Control separation**: UMAP/PCA should show meaningful but not perfect separation
4. **Bootstrap stability**: Resample 80% of samples 100×, check cluster stability (Jaccard >0.75)

---

## 7. Required Python Packages

```bash
pip install pandas numpy scipy scikit-learn gseapy networkx umap-learn requests matplotlib seaborn PyWGCNA
```

## 8. Expected Output Files

| File | Description |
|------|-------------|
| `imputed_matrix.csv` | Log10 imputed protein matrix (1,765 × 30) |
| `imputation_flags.csv` | Per-protein mechanism labels and flags |
| `binary_detection_matrix.csv` | 0/1 detection matrix |
| `sample_qc_scores.csv` | Per-sample QC scores and weights |
| `pathway_protein_map.json` | Pathway → protein membership with sources |
| `protein_weights.csv` | Per-protein per-pathway weights (w1×w2×w3×w4) |
| `pathway_activity_scores.csv` | Samples × pathways weighted scores |
| `ssgsea_scores.csv` | ssGSEA enrichment scores (validation) |
| `wgcna_modules.csv` | WGCNA module assignments per protein |
| `wgcna_eigengenes.csv` | Module eigengenes (30 samples × n_modules) |
| `wgcna_hub_proteins.json` | Top hub proteins per module |
| `combined_features.csv` | Pathway scores + WGCNA eigengenes combined |
| `pca_results.csv` | PCA coordinates and explained variance |
| `umap_results.csv` | UMAP 2D coordinates |

---

## 9. Implementation Order

1. Load and preprocess data (Section 2.1–2.2)
2. Run three-test framework (Section 2.3)
3. Classify missingness mechanisms (Section 2.4)
4. Apply mechanism-matched imputation (Section 2.5)
5. Save Phase 1 outputs (Section 2.7)
6. Map protein names to STRING/UniProt IDs (Section 3.2, Step 1)
7. Retrieve pathway memberships from KEGG + Reactome (Section 4.2)
8. Download STRING PPI network; build pathway subnetworks (Section 3.3)
9. Compute protein weights: w1, w2, w3, w4 (Section 3.3)
10. Compute pathway activity scores — both custom and ssGSEA (Section 3.4)
11. Run WGCNA to find data-driven protein modules (Section 3.6)
12. Run PCA and UMAP on combined pathway + WGCNA scores (Section 3.5 + 3.6)
13. Validate and save all outputs (Section 6)

---

## 10. Sparse Matrix vs. Imputation: Scientific Rationale

### 10.1 The Question

With 85.2% overall missingness, a legitimate scientific question is: **should we impute at all, or use sparse-native methods that operate directly on incomplete data?**

### 10.2 Arguments FOR Sparse Matrix (No Imputation)

| Argument | Detail |
|----------|--------|
| No fabricated values | Every imputed value is an estimate, not a measurement |
| Preserves uncertainty | NaN explicitly communicates "we don't know" |
| Some methods handle it natively | XGBoost, missForest, and sparse PCA can handle NaN |
| NMF can decompose incomplete matrices | Non-negative Matrix Factorization fills gaps implicitly during factorization |
| Avoids imputation bias | Wrong imputation model → systematic errors in downstream analysis |

### 10.3 Arguments FOR Mechanism-Aware Imputation (Our Approach)

| Argument | Detail |
|----------|--------|
| 85% missingness is across ALL 5,410 proteins | The **working set** (1,765 proteins, ≥10% detection) has ~40–60% missingness — much more tractable |
| Missingness is NOT random | Different proteins are missing for different biological/technical reasons; a sparse matrix treats all NaN identically |
| MNAR-LOD is informative | A protein below detection limit IS a measurement: it means "below X". LOD/√2 correctly encodes this |
| MNAR-BIO is already sparse | The 295 patient-exclusive proteins are kept as binary flags — effectively a sparse representation |
| Pathway scoring requires values | Weighted pathway sums, ssGSEA, and GSVA all require ranked or complete data |
| WGCNA requires complete data | Correlation networks cannot be computed with NaN entries |

### 10.4 Recommended Hybrid Approach

The scientifically strongest strategy is **hybrid**: use mechanism-aware imputation as the primary pipeline, but run a **sparse validation track** in parallel to confirm that imputation is not driving the results.

```python
# === SPARSE VALIDATION TRACK ===

# Method 1: NMF on incomplete matrix (fills gaps implicitly)
from sklearn.decomposition import NMF

# NMF requires non-negative values. Shift log10 data so minimum = 0
shift = np.nanmin(log_data.values)
log_data_shifted = log_data - shift
# Fill NaN with 0 for NMF initialization (NMF iterates to fill gaps)
log_data_filled = log_data_shifted.fillna(0)

nmf_model = NMF(n_components=20, init='nndsvda', max_iter=500, random_state=42)
W = nmf_model.fit_transform(log_data_filled.T)   # 30 samples × 20 components
H = nmf_model.components_                          # 20 components × 1765 proteins
# W matrix provides a sparse, data-driven dimensionality reduction

# Method 2: Missing-value-aware distance for UMAP
from scipy.spatial.distance import pdist, squareform

def pairwise_distance_with_nan(data_matrix):
    """Compute pairwise distances using only shared non-missing proteins."""
    n = data_matrix.shape[1]  # samples
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            col_i = data_matrix.iloc[:, i]
            col_j = data_matrix.iloc[:, j]
            shared = col_i.notna() & col_j.notna()
            if shared.sum() < 10:
                dist[i,j] = dist[j,i] = np.nan
            else:
                diff = col_i[shared] - col_j[shared]
                dist[i,j] = dist[j,i] = np.sqrt((diff**2).mean())  # RMSE-like
    return dist

nan_dist = pairwise_distance_with_nan(log_data)
# Fill any NaN distances with max distance
nan_dist = np.nan_to_num(nan_dist, nan=np.nanmax(nan_dist))

# UMAP from precomputed distance
sparse_umap = umap.UMAP(metric='precomputed', n_neighbors=10).fit_transform(nan_dist)
```

### 10.5 Validation: Compare Sparse vs. Imputed Results

```python
from scipy.stats import spearmanr

# Compare UMAP embeddings from imputed vs. sparse approaches
# If both show similar AD/Control separation → imputation is not driving results
# If they diverge → investigate which proteins cause the divergence

# Quantitative comparison:
# 1. Procrustes analysis between imputed UMAP and sparse UMAP
from scipy.spatial import procrustes
mtx1, mtx2, disparity = procrustes(umap_result, sparse_umap)
print(f"Procrustes disparity: {disparity:.4f}")  # Lower = more similar

# 2. Check if same samples cluster together
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
labels_imputed = KMeans(n_clusters=2).fit_predict(umap_result)
labels_sparse = KMeans(n_clusters=2).fit_predict(sparse_umap)
ari = adjusted_rand_score(labels_imputed, labels_sparse)
print(f"Adjusted Rand Index: {ari:.4f}")  # >0.8 = strong agreement
```

### 10.6 Bottom Line

> **The mechanism-aware imputation is scientifically more accurate than a naive sparse matrix** for this specific dataset because:
> 1. It distinguishes WHY each value is missing (LOD vs. degradation vs. biology vs. random)
> 2. MNAR-BIO proteins are already treated as sparse (binary-only, no fabricated values)
> 3. MNAR-LOD imputation with LOD/√2 encodes real information ("below detection floor")
> 4. The sparse validation track provides a safety net to confirm results
>
> A raw sparse matrix would incorrectly treat a Ceruloplasmin NaN in a control sample
> (biologically absent) the same as a Lumican NaN (below instrument sensitivity) — losing
> critical biological signal.