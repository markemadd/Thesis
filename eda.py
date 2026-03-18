"""
Comprehensive Exploratory Data Analysis — CSF Proteomics Dataset
================================================================
Professional-grade EDA covering both canonical proteins and Ig clonotypes.
Generates publication-quality figures saved to output/eda/

Usage:
    python eda.py
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Documentation")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
EDA_DIR    = os.path.join(OUTPUT_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

INPUT_CSV  = os.path.join(DATA_DIR, "normalized df-1.csv")
PATIENT_COLS = [f"P{i}" for i in range(1, 21)]
CONTROL_COLS = [f"C{i}" for i in range(1, 11)]
SAMPLE_COLS  = PATIENT_COLS + CONTROL_COLS

# ── Style ─────────────────────────────────────────────────────────────────────
NAVY     = '#1B2A4A'
BLUE     = '#2E86AB'
TEAL     = '#00B4D8'
GOLD     = '#F2A900'
CORAL    = '#E85D75'
MINT     = '#48BF84'
SLATE    = '#64748B'
LIGHT_BG = '#F8FAFC'
WHITE    = '#FFFFFF'

AD_COLOR   = CORAL
CTRL_COLOR = TEAL
IG_COLOR   = SLATE
CAN_COLOR  = BLUE

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.facecolor': WHITE,
    'axes.edgecolor': '#E2E8F0',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#CBD5E1',
    'figure.facecolor': LIGHT_BG,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': LIGHT_BG,
})

# ── Ig classification ─────────────────────────────────────────────────────────
IG_PATTERNS = [
    r'ighv', r'iglv', r'igkv', r'heavy chain variable',
    r'light chain variable', r'igh\b', r'igl\b', r'igk\b',
    r'immunoglobulin', r'\big\b', r'clone', r'clonotype', r'\bcdr\b',
    r'_heavy_', r'_light_', r'ighg', r'igha', r'ighm', r'ighe',
    r'iglc', r'igkc', r'heavy chain constant', r'light chain constant', r'10e8',
]
_ig_re = re.compile('|'.join(IG_PATTERNS), re.IGNORECASE)

def is_ig(name): return bool(_ig_re.search(str(name)))

def classify_canonical(name):
    nl = str(name).lower()
    if any(k in nl for k in ['complement', 'c1q', 'c1r', 'c1s', 'c4b-binding',
                              'factor b', 'factor h', 'factor i', 'c2 ', 'c3 ', 'c4 ', 'c5 ']):
        return 'Complement'
    if any(k in nl for k in ['apolipoprotein', 'apoa', 'apob', 'apoc', 'apod', 'apoe',
                              'apol', 'apom', 'clusterin']):
        return 'Apolipoprotein'
    if 'ec ' in nl or re.search(r'\bec \d+\.\d+', nl):
        return 'Enzyme'
    if any(k in nl for k in ['serpin', 'inhibitor', 'alpha-2-macroglobulin',
                              'antithrombin', 'cystatin']):
        return 'Protease Inhibitor'
    if any(k in nl for k in ['fibrinogen', 'coagulation', 'prothrombin', 'plasminogen',
                              'factor v', 'factor ix', 'factor x', 'von willebrand']):
        return 'Coagulation'
    if any(k in nl for k in ['albumin', 'transthyretin', 'transferrin', 'haptoglobin',
                              'hemopexin', 'alpha-2-hs', 'fetuin', 'ceruloplasmin']):
        return 'Plasma Abundant'
    if any(k in nl for k in ['chemokine', 'interleukin', 'cytokine', 'growth factor',
                              'igfbp', 'tumor necrosis']):
        return 'Cytokine/GF'
    if any(k in nl for k in ['kinase', 'phosphatase', 'akap']):
        return 'Kinase/Phosphatase'
    if any(k in nl for k in ['receptor', 'glutamate']):
        return 'Receptor'
    if any(k in nl for k in ['neural', 'neuro', 'brain', 'synap', 'secretogranin',
                              'chromogranin', 'kinesin']):
        return 'Neural/Brain'
    if any(k in nl for k in ['cadherin', 'collagen', 'laminin', 'fibulin', 'fibronectin',
                              'extracellular matrix']):
        return 'ECM/Adhesion'
    return 'Other'

def classify_ig_subtype(name):
    nl = str(name).lower()
    if 'ighv' in nl or 'heavy chain variable' in nl or '_heavy_' in nl:
        return 'Heavy Variable'
    if 'iglv' in nl or 'igkv' in nl or 'light chain variable' in nl or '_light_' in nl:
        return 'Light Variable'
    return 'Other Ig'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE 1: Executive Overview Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig1_executive_overview(raw, ig_df, can_df):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('CSF Proteomics — Executive Overview', fontsize=22, fontweight='bold',
                 color=NAVY, y=0.98)
    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.35)

    # ── 1A: Dataset composition donut ──
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [len(ig_df), len(can_df)]
    colors = [IG_COLOR, CAN_COLOR]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=['Ig Clonotypes', 'Canonical'],
        colors=colors, autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.4, edgecolor=WHITE, linewidth=2),
        textprops=dict(fontsize=10))
    for at in autotexts: at.set_fontweight('bold')
    ax1.set_title(f'Protein Composition\n(N={len(raw):,})', fontsize=12, fontweight='bold')

    # ── 1B: Missingness by group ──
    ax2 = fig.add_subplot(gs[0, 1])
    for label, df, color in [('Ig', ig_df, IG_COLOR), ('Canonical', can_df, CAN_COLOR)]:
        miss_pct = df[SAMPLE_COLS].isna().mean(axis=1) * 100
        ax2.hist(miss_pct, bins=30, alpha=0.7, color=color, label=label, edgecolor=WHITE)
    ax2.set_xlabel('Missing %')
    ax2.set_ylabel('Number of Proteins')
    ax2.set_title('Missingness Distribution')
    ax2.legend(frameon=True, fancybox=True)

    # ── 1C: Detection rate per sample ──
    ax3 = fig.add_subplot(gs[0, 2:])
    det_rates = raw[SAMPLE_COLS].notna().sum() / len(raw) * 100
    colors_bar = [AD_COLOR if s.startswith('P') else CTRL_COLOR for s in SAMPLE_COLS]
    bars = ax3.bar(SAMPLE_COLS, det_rates, color=colors_bar, edgecolor=WHITE, linewidth=0.5)
    ax3.axhline(det_rates.mean(), color=NAVY, linestyle='--', alpha=0.7, label=f'Mean: {det_rates.mean():.1f}%')
    ax3.set_ylabel('Proteins Detected (%)')
    ax3.set_title('Detection Rate by Sample')
    ax3.set_xticklabels(SAMPLE_COLS, rotation=45, ha='right', fontsize=9)
    ax3.legend(frameon=True)

    # ── 1D: Intensity distribution (raw vs log2) ──
    ax4 = fig.add_subplot(gs[1, 0:2])
    raw_vals = raw[SAMPLE_COLS].values.flatten()
    raw_vals = raw_vals[~np.isnan(raw_vals)]
    log_vals = np.log2(raw_vals[raw_vals > 0])
    ax4.hist(log_vals, bins=80, color=BLUE, alpha=0.8, edgecolor=WHITE, linewidth=0.3)
    ax4.set_xlabel('Log₂(Intensity)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Intensity Distribution (Log₂-transformed)')
    ax4.axvline(np.median(log_vals), color=CORAL, linestyle='--', lw=2,
                label=f'Median: {np.median(log_vals):.1f}')
    ax4.legend(frameon=True)

    # ── 1E: Canonical category breakdown ──
    ax5 = fig.add_subplot(gs[1, 2:])
    can_df_copy = can_df.copy()
    can_df_copy['category'] = can_df_copy['Protein.names'].apply(classify_canonical)
    cat_counts = can_df_copy['category'].value_counts()
    cat_colors = sns.color_palette('Set2', n_colors=len(cat_counts))
    bars = ax5.barh(cat_counts.index, cat_counts.values, color=cat_colors, edgecolor=WHITE)
    for bar, val in zip(bars, cat_counts.values):
        ax5.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                 str(val), va='center', fontsize=9, fontweight='bold')
    ax5.set_xlabel('Number of Proteins')
    ax5.set_title('Canonical Protein Categories')
    ax5.invert_yaxis()

    plt.savefig(os.path.join(EDA_DIR, '01_executive_overview.png'))
    plt.close()
    print("  ✓ Figure 1: Executive Overview")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE 2: Missingness Deep Dive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig2_missingness(raw, ig_df, can_df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Missingness Analysis', fontsize=20, fontweight='bold', color=NAVY, y=0.98)

    # ── 2A: Missingness heatmap (top 100 most variable canonical) ──
    ax = axes[0, 0]
    var_order = can_df[SAMPLE_COLS].var(axis=1).nlargest(80).index
    miss_matrix = can_df.loc[var_order, SAMPLE_COLS].isna().astype(int)
    cmap = LinearSegmentedColormap.from_list('miss', [WHITE, CORAL])
    sns.heatmap(miss_matrix, cmap=cmap, cbar_kws={'label': 'Missing', 'shrink': 0.5},
                ax=ax, yticklabels=False, xticklabels=True)
    ax.set_xticklabels(SAMPLE_COLS, rotation=45, ha='right', fontsize=8)
    ax.set_title('Missingness Pattern (Top 80 Variable Canonical)')

    # ── 2B: AD vs Control missingness by category ──
    ax = axes[0, 1]
    can_copy = can_df.copy()
    can_copy['category'] = can_copy['Protein.names'].apply(classify_canonical)
    cats = can_copy['category'].unique()
    ad_miss = []
    ctrl_miss = []
    for cat in cats:
        sub = can_copy[can_copy['category'] == cat]
        ad_miss.append(sub[PATIENT_COLS].isna().mean().mean() * 100)
        ctrl_miss.append(sub[CONTROL_COLS].isna().mean().mean() * 100)
    x = np.arange(len(cats))
    w = 0.35
    ax.barh(x - w/2, ad_miss, w, color=AD_COLOR, label='AD (P1–P20)', edgecolor=WHITE)
    ax.barh(x + w/2, ctrl_miss, w, color=CTRL_COLOR, label='Control (C1–C10)', edgecolor=WHITE)
    ax.set_yticks(x)
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel('Missing %')
    ax.set_title('Missingness: AD vs Control by Category')
    ax.legend(frameon=True, loc='lower right')
    ax.invert_yaxis()

    # ── 2C: Cumulative detection curve ──
    ax = axes[1, 0]
    for label, df, color in [('All', raw, NAVY), ('Canonical', can_df, CAN_COLOR), ('Ig', ig_df, IG_COLOR)]:
        miss_pct = df[SAMPLE_COLS].isna().mean(axis=1).sort_values()
        cumulative = np.arange(1, len(miss_pct)+1) / len(miss_pct) * 100
        ax.plot((1-miss_pct.values)*100, cumulative, color=color, lw=2, label=label)
    ax.set_xlabel('Detection Rate (%)')
    ax.set_ylabel('Cumulative % of Proteins')
    ax.set_title('Cumulative Detection Curve')
    ax.legend(frameon=True)
    ax.axvline(50, color=GOLD, linestyle=':', alpha=0.7, label='50% threshold')

    # ── 2D: Missing per sample AD vs Control ──
    ax = axes[1, 1]
    ad_det = can_df[PATIENT_COLS].notna().sum()
    ctrl_det = can_df[CONTROL_COLS].notna().sum()
    bp = ax.boxplot([ad_det.values, ctrl_det.values], labels=['AD (n=20)', 'Control (n=10)'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(AD_COLOR)
    bp['boxes'][1].set_facecolor(CTRL_COLOR)
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor(NAVY)
    ax.set_ylabel('Canonical Proteins Detected')
    ax.set_title('Detection Count: AD vs Control')
    t_stat, p_val = stats.mannwhitneyu(ad_det.values, ctrl_det.values, alternative='two-sided')
    ax.text(0.5, 0.92, f'Mann-Whitney p = {p_val:.3f}', transform=ax.transAxes,
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor=GOLD, alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EDA_DIR, '02_missingness_analysis.png'))
    plt.close()
    print("  ✓ Figure 2: Missingness Analysis")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE 3: Canonical Protein Deep Dive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig3_canonical_deep_dive(can_df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Canonical Proteins — Deep Dive', fontsize=20, fontweight='bold', color=NAVY, y=0.98)

    # Log2-transform for analysis
    log2_data = can_df[SAMPLE_COLS].copy()
    for col in SAMPLE_COLS:
        mask = log2_data[col].notna() & (log2_data[col] > 0)
        log2_data.loc[mask, col] = np.log2(log2_data.loc[mask, col])
        log2_data.loc[log2_data[col].notna() & (log2_data[col] <= 0), col] = np.nan

    # ── 3A: Top 20 most-detected proteins ──
    ax = axes[0, 0]
    det_rate = can_df[SAMPLE_COLS].notna().mean(axis=1)
    top_idx = det_rate.nlargest(20).index
    top_names = can_df.loc[top_idx, 'Protein.names'].apply(lambda x: x[:40])
    top_rates = det_rate.loc[top_idx] * 100
    bars = ax.barh(range(len(top_names)), top_rates.values, color=BLUE, edgecolor=WHITE, alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names.values, fontsize=8)
    ax.set_xlabel('Detection Rate (%)')
    ax.set_title('Top 20 Most-Detected Canonical Proteins')
    ax.invert_yaxis()
    for bar, val in zip(bars, top_rates.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}%', va='center', fontsize=8)

    # ── 3B: Intensity boxplots AD vs Control (top 12) ──
    ax = axes[0, 1]
    top12_idx = det_rate.nlargest(12).index
    plot_data = []
    for idx in top12_idx:
        name = can_df.loc[idx, 'Protein.names'][:25]
        for col in PATIENT_COLS:
            v = log2_data.loc[idx, col]
            if not np.isnan(v):
                plot_data.append({'Protein': name, 'Log2 Intensity': v, 'Group': 'AD'})
        for col in CONTROL_COLS:
            v = log2_data.loc[idx, col]
            if not np.isnan(v):
                plot_data.append({'Protein': name, 'Log2 Intensity': v, 'Group': 'Control'})
    if plot_data:
        pdf = pd.DataFrame(plot_data)
        sns.boxplot(data=pdf, x='Log2 Intensity', y='Protein', hue='Group',
                    palette={'AD': AD_COLOR, 'Control': CTRL_COLOR}, ax=ax,
                    fliersize=3, linewidth=0.8)
        ax.set_title('Intensity: AD vs Control (Top 12)')
        ax.legend(title='', frameon=True, fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    # ── 3C: Protein-protein correlation heatmap ──
    ax = axes[1, 0]
    well_det = det_rate.nlargest(30).index
    corr_data = log2_data.loc[well_det].T
    corr_data.columns = can_df.loc[well_det, 'Protein.names'].apply(lambda x: x[:20])
    corr = corr_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax,
                xticklabels=True, yticklabels=True, vmin=-1, vmax=1,
                cbar_kws={'shrink': 0.6, 'label': 'Pearson r'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    ax.set_title('Correlation: Top 30 Detected Proteins')

    # ── 3D: Dynamic range by category ──
    ax = axes[1, 1]
    can_copy = can_df.copy()
    can_copy['category'] = can_copy['Protein.names'].apply(classify_canonical)
    can_copy['median_log2'] = log2_data.median(axis=1)
    cat_order = can_copy.groupby('category')['median_log2'].median().sort_values(ascending=False).index
    cat_palette = dict(zip(cat_order, sns.color_palette('Set2', len(cat_order))))
    sns.boxplot(data=can_copy.dropna(subset=['median_log2']),
                x='median_log2', y='category', order=cat_order,
                palette=cat_palette, ax=ax, fliersize=2, linewidth=0.8)
    ax.set_xlabel('Median Log₂(Intensity)')
    ax.set_title('Abundance by Protein Category')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EDA_DIR, '03_canonical_deep_dive.png'))
    plt.close()
    print("  ✓ Figure 3: Canonical Deep Dive")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE 4: Ig Clonotype Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig4_ig_analysis(ig_df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Immunoglobulin Clonotype Analysis', fontsize=20, fontweight='bold',
                 color=NAVY, y=0.98)

    # Classify subtypes
    ig_df = ig_df.copy()
    ig_df['subtype'] = ig_df['Protein.names'].apply(classify_ig_subtype)

    # ── 4A: Ig subtype composition ──
    ax = axes[0, 0]
    sub_counts = ig_df['subtype'].value_counts()
    colors = [CORAL, TEAL, GOLD]
    wedges, texts, autotexts = ax.pie(
        sub_counts.values, labels=sub_counts.index,
        colors=colors[:len(sub_counts)], autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.4, edgecolor=WHITE, linewidth=2),
        textprops=dict(fontsize=10))
    for at in autotexts: at.set_fontweight('bold')
    ax.set_title(f'Ig Subtypes (N={len(ig_df):,})')

    # ── 4B: V-gene usage (top 20) ──
    ax = axes[0, 1]
    vgene_re = re.compile(r'(IG[HKL]V\d+-\d+)', re.IGNORECASE)
    vgenes = []
    for name in ig_df['Protein.names']:
        m = vgene_re.search(str(name))
        if m:
            vgenes.append(m.group(1).upper())
    if vgenes:
        vgene_counts = pd.Series(vgenes).value_counts().head(20)
        bars = ax.barh(range(len(vgene_counts)), vgene_counts.values,
                       color=BLUE, edgecolor=WHITE, alpha=0.85)
        ax.set_yticks(range(len(vgene_counts)))
        ax.set_yticklabels(vgene_counts.index, fontsize=9)
        ax.set_xlabel('Count')
        ax.set_title('Top 20 V-Gene Segments Used')
        ax.invert_yaxis()

    # ── 4C: Clonotype detection per sample (AD vs Control) ──
    ax = axes[1, 0]
    ad_clono = ig_df[PATIENT_COLS].notna().sum()
    ctrl_clono = ig_df[CONTROL_COLS].notna().sum()
    all_det = pd.concat([ad_clono, ctrl_clono])
    groups = ['AD'] * 20 + ['Control'] * 10
    sample_names = list(PATIENT_COLS) + list(CONTROL_COLS)
    colors_bar = [AD_COLOR] * 20 + [CTRL_COLOR] * 10
    ax.bar(range(30), all_det.values, color=colors_bar, edgecolor=WHITE, linewidth=0.5)
    ax.set_xticks(range(30))
    ax.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Ig Clonotypes Detected')
    ax.set_title('Clonotype Repertoire Size per Sample')
    # Add group means
    ax.axhline(ad_clono.mean(), color=AD_COLOR, linestyle='--', alpha=0.7, lw=1.5)
    ax.axhline(ctrl_clono.mean(), color=CTRL_COLOR, linestyle='--', alpha=0.7, lw=1.5)
    t_stat, p_val = stats.mannwhitneyu(ad_clono.values, ctrl_clono.values, alternative='two-sided')
    ax.text(0.5, 0.92, f'AD mean: {ad_clono.mean():.0f} | Ctrl mean: {ctrl_clono.mean():.0f} | p={p_val:.3f}',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=GOLD, alpha=0.3))

    # ── 4D: Shared clonotypes ──
    ax = axes[1, 1]
    n_samples_detected = ig_df[SAMPLE_COLS].notna().sum(axis=1)
    shared_counts = n_samples_detected.value_counts().sort_index()
    ax.bar(shared_counts.index, shared_counts.values, color=SLATE, edgecolor=WHITE)
    ax.set_xlabel('Number of Samples Detected In')
    ax.set_ylabel('Number of Clonotypes')
    ax.set_title('Clonotype Sharing Across Samples')
    ax.set_yscale('log')
    # Annotate key stats
    unique = (n_samples_detected == 1).sum()
    shared = (n_samples_detected >= 2).sum()
    ax.text(0.65, 0.85, f'Unique to 1 sample: {unique:,}\nShared (≥2): {shared:,}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor=MINT, alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EDA_DIR, '04_ig_clonotype_analysis.png'))
    plt.close()
    print("  ✓ Figure 4: Ig Clonotype Analysis")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE 5: AD-Relevant Protein Spotlight
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig5_ad_spotlight(can_df):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('AD-Relevant Protein Spotlight', fontsize=20, fontweight='bold',
                 color=NAVY, y=0.98)

    log2_data = can_df[SAMPLE_COLS].copy()
    for col in SAMPLE_COLS:
        mask = log2_data[col].notna() & (log2_data[col] > 0)
        log2_data.loc[mask, col] = np.log2(log2_data.loc[mask, col])
        log2_data.loc[log2_data[col].notna() & (log2_data[col] <= 0), col] = np.nan

    ad_proteins = [
        ('apolipoprotein e', 'ApoE — #1 AD Risk Factor'),
        ('clusterin', 'Clusterin (ApoJ) — #3 AD Risk'),
        ('transthyretin', 'Transthyretin — Aβ Inhibitor'),
        ('cystatin', 'Cystatin-C — AD Biomarker'),
        ('complement c3', 'Complement C3 — Neuroinflammation'),
        ('alpha-2-macroglobulin', 'α2-Macroglobulin — Aβ Clearance'),
    ]

    for i, (keyword, title) in enumerate(ad_proteins):
        ax = axes[i // 3, i % 3]
        mask = can_df['Protein.names'].str.lower().str.contains(keyword, na=False)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, f'{title}\nNot found', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            ax.set_title(title)
            continue

        idx = mask.idxmax()
        ad_vals = log2_data.loc[idx, PATIENT_COLS].dropna().values
        ctrl_vals = log2_data.loc[idx, CONTROL_COLS].dropna().values

        if len(ad_vals) > 0 and len(ctrl_vals) > 0:
            parts = ax.violinplot([ad_vals, ctrl_vals], positions=[1, 2], showmeans=True,
                                  showmedians=True, widths=0.6)
            for j, pc in enumerate(parts['bodies']):
                pc.set_facecolor(AD_COLOR if j == 0 else CTRL_COLOR)
                pc.set_alpha(0.7)
            parts['cmeans'].set_color(NAVY)
            parts['cmedians'].set_color(GOLD)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['AD', 'Control'])
            ax.set_ylabel('Log₂(Intensity)')

            t, p = stats.mannwhitneyu(ad_vals, ctrl_vals, alternative='two-sided')
            miss_pct = can_df.loc[idx, SAMPLE_COLS].isna().mean() * 100
            ax.set_title(f'{title}\np={p:.3f} | {miss_pct:.0f}% missing', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'{title}\nInsufficient data', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_title(title)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EDA_DIR, '05_ad_protein_spotlight.png'))
    plt.close()
    print("  ✓ Figure 5: AD Protein Spotlight")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIGURE 6: Sample Correlation & PCA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fig6_sample_analysis(can_df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Sample-Level Analysis (Canonical Proteins)',
                 fontsize=20, fontweight='bold', color=NAVY, y=0.98)

    log2_data = can_df[SAMPLE_COLS].copy()
    for col in SAMPLE_COLS:
        mask = log2_data[col].notna() & (log2_data[col] > 0)
        log2_data.loc[mask, col] = np.log2(log2_data.loc[mask, col])
        log2_data.loc[log2_data[col].notna() & (log2_data[col] <= 0), col] = np.nan

    # ── 6A: Sample correlation heatmap ──
    ax = axes[0]
    corr = log2_data.corr()
    annot_labels = ['AD'] * 20 + ['Ctrl'] * 10
    cmap = LinearSegmentedColormap.from_list('corr', [CORAL, WHITE, TEAL])
    sns.heatmap(corr, cmap='coolwarm', center=corr.median().median(), ax=ax,
                xticklabels=SAMPLE_COLS, yticklabels=SAMPLE_COLS,
                cbar_kws={'shrink': 0.6, 'label': 'Pearson r'})
    ax.set_xticklabels(SAMPLE_COLS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(SAMPLE_COLS, fontsize=8)
    ax.set_title('Inter-Sample Correlation')

    # ── 6B: PCA on well-detected proteins ──
    ax = axes[1]
    det = log2_data.notna().mean(axis=1)
    good = log2_data.loc[det >= 0.6].dropna()
    if len(good) >= 5:
        from sklearn.decomposition import PCA
        X = good.T.values
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        for i, s in enumerate(SAMPLE_COLS):
            color = AD_COLOR if s.startswith('P') else CTRL_COLOR
            marker = 's' if s.startswith('P') else 'o'
            ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker,
                       s=80, edgecolors=NAVY, linewidth=0.5, zorder=3)
            ax.annotate(s, (coords[i, 0], coords[i, 1]), fontsize=7,
                       xytext=(4, 4), textcoords='offset points')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'PCA — {len(good)} Proteins (≥60% detected)')
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=AD_COLOR,
                   markersize=10, label='AD'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CTRL_COLOR,
                   markersize=10, label='Control'),
        ]
        ax.legend(handles=legend_elements, frameon=True)
    else:
        ax.text(0.5, 0.5, 'Insufficient complete proteins for PCA',
                transform=ax.transAxes, ha='center')
        ax.set_title('PCA (insufficient data)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(EDA_DIR, '06_sample_analysis.png'))
    plt.close()
    print("  ✓ Figure 6: Sample Analysis")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("\n" + "═" * 60)
    print("  COMPREHENSIVE EDA — CSF Proteomics Dataset")
    print("═" * 60)

    # Load data
    print("\n  Loading data...")
    raw = pd.read_csv(INPUT_CSV)
    raw['is_ig'] = raw['Protein.names'].apply(is_ig)
    ig_df = raw[raw['is_ig']].drop(columns=['is_ig']).reset_index(drop=True)
    can_df = raw[~raw['is_ig']].drop(columns=['is_ig']).reset_index(drop=True)
    print(f"  Total: {len(raw):,} | Ig: {len(ig_df):,} | Canonical: {len(can_df):,}")
    print(f"  Saving figures to: {EDA_DIR}\n")

    # Generate figures
    fig1_executive_overview(raw, ig_df, can_df)
    fig2_missingness(raw, ig_df, can_df)
    fig3_canonical_deep_dive(can_df)
    fig4_ig_analysis(ig_df)
    fig5_ad_spotlight(can_df)
    fig6_sample_analysis(can_df)

    print(f"\n  ✅ All 6 figures saved to {EDA_DIR}")
    print("═" * 60)


if __name__ == '__main__':
    main()
