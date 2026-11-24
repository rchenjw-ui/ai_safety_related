#!/usr/bin/env python3
"""
TOX21 - XGBoost + SHAP
----------------------
- Loads precomputed features (train + val)
- Uses features selected on the training set only
- Trains one XGBoost model per endpoint with early stopping
- Evaluates on validation set (ROC-AUC, PR-AUC)
- Computes SHAP values and saves summary plots
"""

import os
import json
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb


# -----------------------------
# Config
# -----------------------------
FEATURE_DIR = "tox21_features"
TRAIN_FEAT_PATH = os.path.join(FEATURE_DIR, "train_features.parquet")
VAL_FEAT_PATH   = os.path.join(FEATURE_DIR, "val_features.parquet")

OUTPUT_DIR = "tox21_xgb_models"
SHAP_DIR   = os.path.join(OUTPUT_DIR, "shap")

RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": RANDOM_STATE,
}

EARLY_STOPPING_ROUNDS = 50
MAX_SHAP_SAMPLES = 1000          # for speed; increase if you want

ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


# -----------------------------
# Helpers
# -----------------------------
def load_selected_features(endpoint: str) -> List[str]:
    path = os.path.join(FEATURE_DIR, f"selected_{endpoint}.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def prepare_xy(feat_df: pd.DataFrame, labels: pd.Series, feature_list: List[str]):
    """Return X, y after dropping missing labels and keeping only selected features."""
    # Keep only features that actually exist
    feats = [f for f in feature_list if f in feat_df.columns]
    if not feats:
        return None, None, []

    mask = labels.notna()
    X = feat_df.loc[mask, feats].copy()
    y = labels.loc[mask].astype(int).copy()
    return X, y, feats


# -----------------------------
# Training + SHAP for one endpoint
# -----------------------------
def train_and_explain(
    endpoint: str,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_val_full: pd.DataFrame,
    y_val_full: pd.Series,
) -> Dict:

    selected = load_selected_features(endpoint)
    if not selected:
        return {"endpoint": endpoint, "error": "No selected features found"}

    X_tr, y_tr, feats = prepare_xy(X_train_full, y_train_full, selected)
    X_va, y_va, _     = prepare_xy(X_val_full,   y_val_full,   selected)
    # Automatic class imbalance handling
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    if X_tr is None or len(y_tr) < 50 or y_tr.nunique() < 2:
        return {"endpoint": endpoint, "error": "Insufficient training data"}

    # Internal split for early stopping
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=RANDOM_STATE
    )

    model = xgb.XGBClassifier(
    **XGB_PARAMS,
    scale_pos_weight=scale_pos_weight,          # ← add this
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    model.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    # ----- Validation metrics -----
    metrics = {
        "roc_auc": None,
        "pr_auc": None,
        "f1": None,
        "baseline_pr_auc": None,
        "baseline_roc_auc": 0.5,          # random baseline for ROC
        "positive_fraction_train": None,
        "positive_fraction_val": None,
    }

    # Positive fraction (train)
    pos_frac_train = y_tr.mean()
    metrics["positive_fraction_train"] = float(pos_frac_train)

    if X_va is not None and len(y_va) > 10 and y_va.nunique() > 1:
        proba = model.predict_proba(X_va)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics["roc_auc"] = float(roc_auc_score(y_va, proba))
        metrics["pr_auc"]  = float(average_precision_score(y_va, proba))
        metrics["f1"]      = float(f1_score(y_va, preds))

        # Baseline PR-AUC = prevalence of positive class in validation set
        pos_frac_val = y_va.mean()
        metrics["positive_fraction_val"] = float(pos_frac_val)
        metrics["baseline_pr_auc"] = float(pos_frac_val)
    else:
        metrics["positive_fraction_val"] = None
        metrics["baseline_pr_auc"] = None

    # ----- Save model -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, f"xgb_{endpoint}.json")
    model.save_model(model_path)

    # ----- SHAP -----
    os.makedirs(SHAP_DIR, exist_ok=True)
    shap_path = None
    try:
        # Use a subset for speed
        sample_size = min(MAX_SHAP_SAMPLES, len(X_tr))
        X_shap = X_tr.sample(sample_size, random_state=RANDOM_STATE)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        # Summary plot (bar)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance – {endpoint}")
        plt.tight_layout()
        bar_path = os.path.join(SHAP_DIR, f"shap_bar_{endpoint}.png")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Summary plot (beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
        plt.title(f"SHAP Summary – {endpoint}")
        plt.tight_layout()
        beeswarm_path = os.path.join(SHAP_DIR, f"shap_beeswarm_{endpoint}.png")
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()

        shap_path = {
            "bar": bar_path,
            "beeswarm": beeswarm_path,
        }
    except Exception as e:
        warnings.warn(f"SHAP failed for {endpoint}: {e}")

    result = {
    "endpoint": endpoint,
    "n_features": len(feats),
    "n_train": int(len(y_tr)),
    "n_val": int(len(y_va)) if y_va is not None else 0,
    "best_iteration": int(model.best_iteration) if model.best_iteration is not None else None,
    "roc_auc": metrics["roc_auc"],
    "pr_auc": metrics["pr_auc"],
    "f1": metrics["f1"],
    "baseline_pr_auc": metrics["baseline_pr_auc"],
    "baseline_roc_auc": metrics["baseline_roc_auc"],
    "positive_fraction_train": metrics["positive_fraction_train"],
    "positive_fraction_val": metrics["positive_fraction_val"],
    "model_path": model_path,
    "shap": shap_path,
    "scale_pos_weight": float(scale_pos_weight)
    }
    return result


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading feature matrices...")
    X_train = pd.read_parquet(TRAIN_FEAT_PATH)
    X_val   = pd.read_parquet(VAL_FEAT_PATH)

    # We need the original labels. Load the split files again.
    train_labels = pd.read_parquet("tox21_split/train.parquet")
    val_labels   = pd.read_parquet("tox21_split/val.parquet")

    # Align by SMILES (safest)
    train_labels = train_labels.set_index("smiles")
    val_labels   = val_labels.set_index("smiles")

    X_train = X_train.set_index("smiles")
    X_val   = X_val.set_index("smiles")

    # Keep only common molecules (in case some failed featurization)
    common_train = X_train.index.intersection(train_labels.index)
    common_val   = X_val.index.intersection(val_labels.index)

    X_train = X_train.loc[common_train]
    X_val   = X_val.loc[common_val]
    train_labels = train_labels.loc[common_train]
    val_labels   = val_labels.loc[common_val]

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    results = []
    for endpoint in ENDPOINTS:
        print(f"\n=== {endpoint} ===")
        res = train_and_explain(
            endpoint,
            X_train, train_labels[endpoint],
            X_val,   val_labels[endpoint],
        )
        results.append(res)

        if "error" in res:
            print(f"  Error: {res['error']}")
        else:
            print(f"  Features: {res['n_features']} | Train: {res['n_train']} | Val: {res['n_val']}")
            print(f"  Best iteration: {res['best_iteration']}")
            print(f"  Positive fraction (train/val): "
                f"{res['positive_fraction_train']:.3f} / {res['positive_fraction_val']:.3f}")
            if res["roc_auc"] is not None:
                print(f"  ROC-AUC: {res['roc_auc']:.4f}  (baseline = {res['baseline_roc_auc']:.1f})")
                print(f"  PR-AUC : {res['pr_auc']:.4f}  (baseline = {res['baseline_pr_auc']:.4f})")
                print(f"  F1     : {res['f1']:.4f}")
                print(f"  scale_pos_weight: {res['scale_pos_weight']:.2f}")

    # Save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary → {summary_path}")

    # Pretty table
    print("\n===== Final Results =====")
    print(f"{'Endpoint':<18} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'Base PR':>8} {'Pos% Val':>9}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            continue
        roc  = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
        pr   = f"{r['pr_auc']:.4f}"  if r["pr_auc"]  is not None else "N/A"
        f1   = f"{r['f1']:.4f}"      if r["f1"]      is not None else "N/A"
        base = f"{r['baseline_pr_auc']:.4f}" if r["baseline_pr_auc"] is not None else "N/A"
        pos  = f"{r['positive_fraction_val']*100:.1f}%" if r["positive_fraction_val"] is not None else "N/A"
        print(f"{r['endpoint']:<18} {roc:>8} {pr:>8} {f1:>8} {base:>8} {pos:>9}")

if __name__ == "__main__":
    main()