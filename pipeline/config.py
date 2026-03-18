"""
Configuration for the AD Proteomics Pipeline.
All paths, parameters, and constants in one place.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Documentation")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV = os.path.join(DATA_DIR, "normalized df-1.csv")

# ── Sample layout ──────────────────────────────────────────────────────────
PATIENT_COLS = [f"P{i}" for i in range(1, 21)]   # P1–P20 (20 AD patients)
CONTROL_COLS = [f"C{i}" for i in range(1, 11)]    # C1–C10 (10 controls)
SAMPLE_COLS = PATIENT_COLS + CONTROL_COLS
GROUP_LABELS = ["AD"] * 20 + ["Control"] * 10

# ── Detection thresholds ──────────────────────────────────────────────────
MIN_DETECTION_RATE = 0.10          # ≥10% → working set (≥3 of 30 samples)
MIN_DETECTION_COUNT = 3            # ceiling(0.10 * 30)

# ── Three-test framework thresholds ───────────────────────────────────────
FRAC_LOW_THRESHOLD = 0.50          # Test A: above this → MNAR-LOD
FRAC_LOW_BIO_CEILING = 0.30        # Test C gate: must be below this for MNAR-BIO
R_MISS_QC_THRESHOLD = -0.35        # Test B: below this → MNAR-DEG
FISHER_P_THRESHOLD = 0.05          # Test C: below this → MNAR-BIO candidate
GROUP_EXCLUSIVE_MIN = 3            # Minimum detections in one group with 0 in other

# ── Imputation parameters ─────────────────────────────────────────────────
DEG_IMPUTATION_FACTOR = 0.70       # Multiply group median by this for MNAR-DEG
KNN_NEIGHBORS = 5                  # k for KNN imputation of MCAR proteins

# ── Sample QC ──────────────────────────────────────────────────────────────
N_ANCHOR_PROTEINS = 5              # Number of anchor proteins for QC score
SAMPLE_QC_FLAG_THRESHOLD = -1.5    # Samples below this get down-weighted
SAMPLE_QC_DOWNWEIGHT = 0.70        # Weight factor for flagged samples

# ── Pathway mapping ───────────────────────────────────────────────────────
STRING_SPECIES = 9606              # Homo sapiens
STRING_API_BASE = "https://string-db.org/api"
STRING_MIN_SCORE = 400             # Medium confidence for PPI edges
STRING_BATCH_SIZE = 100            # Max proteins per API call
API_RATE_LIMIT = 1.0               # Seconds between API calls

KEGG_API_BASE = "https://rest.kegg.jp"
REACTOME_API_BASE = "https://reactome.org/ContentService"

MIN_PATHWAY_PROTEINS = 3           # Minimum matched proteins to keep a pathway
MAX_PATHWAY_PROTEINS = 500         # Maximum size (skip genome-wide terms)

# ── Protein weighting ─────────────────────────────────────────────────────
DEG_RELIABILITY_FACTOR = 0.70      # w1 reliability multiplier for MNAR-DEG proteins

# ── WGCNA ──────────────────────────────────────────────────────────────────
WGCNA_MIN_MODULE_SIZE = 10         # Small due to n=30 samples
WGCNA_DEEP_SPLIT = 2
WGCNA_MERGE_CUT_HEIGHT = 0.25
WGCNA_NETWORK_TYPE = "signed"

# ── Dimensionality reduction ──────────────────────────────────────────────
PCA_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.3
UMAP_N_COMPONENTS = 2
NMF_N_COMPONENTS = 20
RANDOM_STATE = 42

# ── Phase 5: Pathway Reduction ────────────────────────────────────────────
JACCARD_THRESHOLD = 0.8              # De-redundancy Jaccard cutoff
JACCARD_SENSITIVITY = [0.7, 0.8, 0.9]  # Sensitivity analysis thresholds
SCORE_CORR_WEIGHT = True             # Use max(Jaccard, |r|) for clustering
IQR_PERCENTILE_CUTOFF = 5            # Remove pathways below this IQR percentile
STABILITY_N_BOOTSTRAP = 200          # Bootstrap iterations for stability selection
STABILITY_THRESHOLD = 0.60           # Must be selected in ≥60% of runs
ELASTIC_NET_L1_RATIOS = [0.5, 0.7, 0.9]  # l1_ratio grid for ElasticNetCV
ELASTIC_NET_CV_FOLDS = 5             # Inner CV folds for tuning
PERMUTATION_N = 1000                 # Label shuffles for permutation test

# ── AD-relevant KEGG pathways to prioritize ───────────────────────────────
AD_KEGG_PATHWAYS = {
    "hsa05010": "Alzheimer disease",
    "hsa04610": "Complement and coagulation cascades",
    "hsa04722": "Neurotrophin signaling",
    "hsa04064": "NF-kappa B signaling",
    "hsa04140": "Autophagy",
    "hsa04142": "Lysosome",
    "hsa04150": "mTOR signaling",
    "hsa04151": "PI3K-Akt signaling",
    "hsa04910": "Insulin signaling",
    "hsa04152": "AMPK signaling",
    "hsa04721": "Synaptic vesicle cycle",
    "hsa04720": "Long-term potentiation",
    "hsa04725": "Cholinergic synapse",
    "hsa04020": "Calcium signaling",
    "hsa03320": "PPAR signaling",
    "hsa04512": "ECM-receptor interaction",
    "hsa00510": "N-Glycan biosynthesis",
}

AD_REACTOME_PATHWAYS = {
    "R-HSA-166658": "Complement cascade",
    "R-HSA-109582": "Hemostasis",
    "R-HSA-168249": "Innate immune system",
    "R-HSA-1280215": "Cytokine signaling in immune system",
    "R-HSA-977225": "Amyloid fiber formation",
    "R-HSA-556833": "Lipid metabolism",
    "R-HSA-917937": "Iron homeostasis",
}
