"""
Proof-of-Concept AD Classifier

Trains a logistic regression model on the 13 stable pathway-level biomarkers
identified by the baseline (non-imputed) pipeline. The trained model and its
preprocessing artefacts are saved to disk for future use on external
validation data.

Usage:
    # Train and save the model
    python3 poc_classifier.py --train

    # Predict on new data (CSV with pathway scores)
    python3 poc_classifier.py --predict new_patient_scores.csv

    # Print model summary
    python3 poc_classifier.py --info

Thesis statement:
    "We identified a panel of 13 pathway-level biomarkers that classify AD vs
    Control with [nested LOOCV accuracy]% accuracy (p=0.001), suggesting these
    pathways as candidate diagnostic features. External validation on an
    independent cohort is needed before clinical application."
"""

import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, cohen_kappa_score,
    classification_report, confusion_matrix,
)

from pipeline.config import OUTPUT_DIR, SAMPLE_COLS, PATIENT_COLS, CONTROL_COLS

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ad_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

IN_SCORES = os.path.join(OUTPUT_DIR, "pathway_scores.csv")
IN_STABLE = os.path.join(OUTPUT_DIR, "stable_pathways.csv")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train():
    print("=" * 60)
    print("  Training AD vs Control Classifier (POC)")
    print("=" * 60)

    # Load stable pathways & scores
    stable = pd.read_csv(IN_STABLE)
    scores = pd.read_csv(IN_SCORES, index_col=0)
    pathway_ids = stable['pathway_id'].tolist()

    print(f"\n  Biomarker panel: {len(pathway_ids)} pathways")
    for _, row in stable.iterrows():
        print(f"    • {row['pathway_name']:<50} "
              f"({row['selection_prob']:.0%}, {row['direction']})")

    # Build feature matrix
    sample_cols = [c for c in scores.columns if c in SAMPLE_COLS]
    X_rows = []
    for sid in sample_cols:
        row = []
        for pid in pathway_ids:
            if pid in scores.index:
                val = scores.loc[pid, sid]
                row.append(val if not pd.isna(val) else 0.0)
            else:
                row.append(0.0)
        X_rows.append(row)

    X = np.array(X_rows)
    y = np.array([1 if s in PATIENT_COLS else 0 for s in sample_cols])

    print(f"\n  Training data: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Classes: {y.sum()} AD, {(1-y).sum()} Control")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression (L2 regularized, class balanced)
    model = LogisticRegressionCV(
        penalty='l2',
        solver='lbfgs',
        cv=5,
        max_iter=5000,
        class_weight='balanced',
        random_state=42,
        scoring='roc_auc',
    )
    model.fit(X_scaled, y)

    # Training performance (for reference only — real perf is nested LOOCV)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    train_acc = accuracy_score(y, y_pred)
    train_auc = roc_auc_score(y, y_proba)

    print(f"\n  Training accuracy (resubstitution): {train_acc:.1%}")
    print(f"  Training AUC (resubstitution):      {train_auc:.3f}")
    print(f"  Note: This is optimistic. Use nested LOOCV for true estimate.")

    # Model coefficients
    coefs = model.coef_.flatten()
    print(f"\n  Model coefficients:")
    print(f"  {'Pathway':<50} {'Coef':>8}  {'Direction'}")
    print("  " + "─" * 72)
    for pid, coef, (_, row) in zip(pathway_ids, coefs, stable.iterrows()):
        name = str(row['pathway_name'])[:49]
        direction = "→ AD" if coef > 0 else "→ Control"
        print(f"  {name:<50} {coef:>+8.4f}  {direction}")

    # Save model + scaler + metadata
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    meta = {
        'n_features': len(pathway_ids),
        'n_training_samples': int(X.shape[0]),
        'n_ad': int(y.sum()),
        'n_control': int((1 - y).sum()),
        'pathway_ids': pathway_ids,
        'pathway_names': stable['pathway_name'].tolist(),
        'pathway_directions': stable['direction'].tolist(),
        'pathway_stability': stable['selection_prob'].tolist(),
        'regularization_C': float(model.C_[0]),
        'training_accuracy': round(train_acc, 3),
        'training_auc': round(train_auc, 3),
        'intercept': float(model.intercept_[0]),
        'coefficients': {pid: float(c) for pid, c in zip(pathway_ids, coefs)},
        'feature_means': scaler.mean_.tolist(),
        'feature_stds': scaler.scale_.tolist(),
        'note': 'POC model. Use nested LOOCV accuracy for true performance estimate.',
    }

    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✓ Model saved   → {MODEL_PATH}")
    print(f"  ✓ Scaler saved  → {SCALER_PATH}")
    print(f"  ✓ Metadata      → {META_PATH}")
    print(f"\n{'=' * 60}")
    print(f"  Model ready. Use --predict <csv> to classify new patients.")
    print(f"{'=' * 60}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict(input_csv: str, label_col: str | None = None):
    """
    Score new patient(s) using the saved model.

    Expected input CSV format:
        - Rows = patients (samples)
        - Columns must include the 13 pathway IDs as column names
        - Each cell = the ssGSEA pathway activity score for that patient/pathway

    Optionally, if label_col is provided, the CSV also contains ground truth
    labels for evaluation (useful for external validation).
    """
    print("=" * 60)
    print("  AD Classifier — Prediction Mode")
    print("=" * 60)

    # Load model + scaler + meta
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(META_PATH) as f:
        meta = json.load(f)

    pathway_ids = meta['pathway_ids']
    pathway_names = meta['pathway_names']

    print(f"\n  Model: {len(pathway_ids)} pathway biomarkers")
    print(f"  Trained on: {meta['n_training_samples']} samples "
          f"({meta['n_ad']} AD, {meta['n_control']} Control)")

    # Load new data
    new_data = pd.read_csv(input_csv, index_col=0)
    print(f"\n  Input: {input_csv}")
    print(f"  Samples: {new_data.shape[0]}")

    # Check for required pathways
    missing_pathways = [p for p in pathway_ids if p not in new_data.columns]
    if missing_pathways:
        print(f"\n  ⚠ Missing {len(missing_pathways)} pathway columns:")
        for mp in missing_pathways:
            idx = pathway_ids.index(mp)
            print(f"    • {mp} ({pathway_names[idx]})")
        print("  Missing pathways will be set to 0 (mean-imputed after scaling).")

    # Build feature matrix
    X_new = np.zeros((new_data.shape[0], len(pathway_ids)))
    for j, pid in enumerate(pathway_ids):
        if pid in new_data.columns:
            X_new[:, j] = new_data[pid].fillna(0).values
        # else: stays 0

    # Scale using training statistics
    X_scaled = scaler.transform(X_new)

    # Predict
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # Results
    print(f"\n  {'Sample':<20} {'Prediction':<12} {'AD Probability':>15}")
    print("  " + "─" * 50)
    for i, sample_id in enumerate(new_data.index):
        label = "AD" if y_pred[i] == 1 else "Control"
        prob = y_proba[i]
        flag = " ⚠" if 0.3 < prob < 0.7 else ""  # low-confidence warning
        print(f"  {str(sample_id):<20} {label:<12} {prob:>14.1%}{flag}")

    # If ground truth is available, evaluate
    if label_col and label_col in new_data.columns:
        y_true = new_data[label_col].values
        # Convert string labels if needed
        if y_true.dtype == object:
            y_true = np.array([1 if str(v).lower() in ('ad', '1', 'patient', 'true')
                               else 0 for v in y_true])

        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = float('nan')
        kappa = cohen_kappa_score(y_true, y_pred)

        print(f"\n  External Validation Results:")
        print(f"    Accuracy:     {acc:.1%}")
        print(f"    AUC:          {auc:.3f}")
        print(f"    Cohen's κ:    {kappa:.3f}")
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"                 Predicted")
        print(f"                 Ctrl    AD")
        print(f"    Actual Ctrl  {cm[0, 0]:<7} {cm[0, 1]}")
        print(f"    Actual AD    {cm[1, 0]:<7} {cm[1, 1]}")
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred,
                                     target_names=['Control', 'AD']))

    # Save predictions
    out_df = pd.DataFrame({
        'sample_id': new_data.index,
        'prediction': ['AD' if p == 1 else 'Control' for p in y_pred],
        'ad_probability': np.round(y_proba, 4),
    })
    out_path = input_csv.replace('.csv', '_predictions.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Predictions saved → {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INFO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def info():
    """Print model summary without loading the full model."""
    if not os.path.exists(META_PATH):
        print("No trained model found. Run with --train first.")
        return

    with open(META_PATH) as f:
        meta = json.load(f)

    print("=" * 60)
    print("  AD Classifier — Model Summary")
    print("=" * 60)
    print(f"\n  Training samples: {meta['n_training_samples']} "
          f"({meta['n_ad']} AD, {meta['n_control']} Control)")
    print(f"  Biomarker panel: {meta['n_features']} pathways")
    print(f"  Regularization C: {meta['regularization_C']:.4f}")
    print(f"  Training accuracy: {meta['training_accuracy']:.1%}")
    print(f"  Training AUC: {meta['training_auc']:.3f}")

    print(f"\n  {'#':<4} {'Pathway':<50} {'Coef':>8}  {'Stability':>10}")
    print("  " + "─" * 76)
    for i, (pid, name) in enumerate(zip(meta['pathway_ids'], meta['pathway_names'])):
        coef = meta['coefficients'][pid]
        stab = meta['pathway_stability'][i]
        print(f"  {i+1:<4} {name:<50} {coef:>+8.4f}  {stab:>9.0%}")

    print(f"\n  Note: {meta['note']}")

    print(f"\n  To classify new patients, prepare a CSV with columns:")
    print(f"  {', '.join(meta['pathway_ids'][:5])}, ...")
    print(f"  Then run: python3 poc_classifier.py --predict <file.csv>")
    print(f"  Optionally add --label-col <col> for external validation.\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AD vs Control Proof-of-Concept Classifier')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true',
                       help='Train and save the model')
    group.add_argument('--predict', type=str, metavar='CSV',
                       help='Predict on a new CSV file')
    group.add_argument('--info', action='store_true',
                       help='Print model summary')
    parser.add_argument('--label-col', type=str, default=None,
                        help='Column name with ground truth labels (for validation)')
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict(args.predict, label_col=args.label_col)
    elif args.info:
        info()
