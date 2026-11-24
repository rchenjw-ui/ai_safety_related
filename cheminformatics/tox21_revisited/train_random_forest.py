#!/usr/bin/env python3
"""
TOX21 - Deep Random Forest baseline
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

FEATURE_DIR = "tox21_features"
TRAIN_FEAT_PATH = os.path.join(FEATURE_DIR, "train_features.parquet")
VAL_FEAT_PATH   = os.path.join(FEATURE_DIR, "val_features.parquet")
OUTPUT_DIR = "tox21_rf_models"

ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

def load_selected(endpoint):
    path = os.path.join(FEATURE_DIR, f"selected_{endpoint}.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train = pd.read_parquet(TRAIN_FEAT_PATH).set_index("smiles")
    X_val   = pd.read_parquet(VAL_FEAT_PATH).set_index("smiles")

    train_lab = pd.read_parquet("tox21_split/train.parquet").set_index("smiles")
    val_lab   = pd.read_parquet("tox21_split/val.parquet").set_index("smiles")

    # Align
    common_tr = X_train.index.intersection(train_lab.index)
    common_va = X_val.index.intersection(val_lab.index)
    X_train, train_lab = X_train.loc[common_tr], train_lab.loc[common_tr]
    X_val, val_lab     = X_val.loc[common_va], val_lab.loc[common_va]

    results = []

    for endpoint in ENDPOINTS:
        print(f"\n=== {endpoint} ===")
        feats = load_selected(endpoint)
        feats = [f for f in feats if f in X_train.columns]
        if not feats:
            print("  No features")
            continue

        # Drop missing labels
        mask_tr = train_lab[endpoint].notna()
        mask_va = val_lab[endpoint].notna()

        X_tr = X_train.loc[mask_tr, feats]
        y_tr = train_lab.loc[mask_tr, endpoint].astype(int)
        X_va = X_val.loc[mask_va, feats]
        y_va = val_lab.loc[mask_va, endpoint].astype(int)

        if len(y_tr) < 50 or y_tr.nunique() < 2:
            print("  Skipping – insufficient data")
            continue

        # Deep + balanced Random Forest
        rf = RandomForestClassifier(
            n_estimators=2000,          # very deep ensemble
            max_depth=None,             # pure deep trees
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",  # handles imbalance
            n_jobs=-1,
            random_state=42,
            oob_score=False,
        )
        rf.fit(X_tr, y_tr)

        # Evaluation
        proba = rf.predict_proba(X_va)[:, 1]
        preds = (proba >= 0.5).astype(int)

        roc = roc_auc_score(y_va, proba)
        pr  = average_precision_score(y_va, proba)
        f1  = f1_score(y_va, preds)
        base_pr = y_va.mean()

        print(f"  ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | F1: {f1:.4f} | Base PR: {base_pr:.4f}")

        results.append({
            "endpoint": endpoint,
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "f1": float(f1),
            "baseline_pr_auc": float(base_pr),
            "positive_fraction_val": float(base_pr),
            "n_features": len(feats),
            "n_train": len(y_tr),
            "n_val": len(y_va),
        })

        # Optional: save model
        # import joblib
        # joblib.dump(rf, os.path.join(OUTPUT_DIR, f"rf_{endpoint}.joblib"))

    # Summary table
    print("\n===== Random Forest Results =====")
    print(f"{'Endpoint':<18} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'Base PR':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['endpoint']:<18} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['baseline_pr_auc']:>8.4f}")

    with open(os.path.join(OUTPUT_DIR, "rf_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()