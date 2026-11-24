#!/usr/bin/env python3
"""
TOX21 Feature Pipeline (Train + Val)
------------------------------------
- Feature selection (Mutual Information) is performed ONLY on the training set
- Both train and validation sets are featurized with the same descriptors
- Saves:
    - train_features.parquet
    - val_features.parquet
    - selected_{endpoint}.txt   (chosen on train only)
"""

import os
import warnings
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Fragments
from sklearn.feature_selection import mutual_info_classif

# Optional but recommended
try:
    from skfp.fingerprints import KlekotaRothFingerprint
    HAS_SKFP = True
except ImportError:
    HAS_SKFP = False
    warnings.warn("scikit-fingerprints not installed → Klekota-Roth will be skipped.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    train_path: str = "tox21_split/train.parquet"
    val_path: str   = "tox21_split/val.parquet"
    smiles_col: str = "smiles"

    top_k: int = 700 #raise it to 700
    use_correlation_filter: bool = False
    corr_threshold: float = 0.90

    n_jobs: int = -1
    output_dir: str = "tox21_features"
    random_state: int = 42


ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


# ---------------------------------------------------------------------------
# Featurization helpers
# ---------------------------------------------------------------------------
def get_fragment_names() -> List[str]:
    return [name for name in dir(Fragments) if name.startswith("fr_")]


def featurize_one(smiles: str) -> Optional[Tuple[Dict, np.ndarray]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = {name: getattr(Fragments, name)(mol) for name in get_fragment_names()}
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)
    return frags, maccs


def build_feature_matrix(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (feature_df, valid_df)"""
    smiles_list = df[cfg.smiles_col].tolist()

    print(f"  Computing Fragments + MACCS on {len(smiles_list)} molecules...")
    results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(featurize_one)(smi) for smi in smiles_list
    )

    valid_idx = [i for i, r in enumerate(results) if r is not None]
    results = [results[i] for i in valid_idx]
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    smiles_valid = df_valid[cfg.smiles_col].tolist()

    # Fragments
    frag_names = get_fragment_names()
    frag_matrix = np.array([[r[0][name] for name in frag_names] for r in results], dtype=np.float32)

    # MACCS
    maccs_matrix = np.array([r[1] for r in results], dtype=np.float32)

    feature_blocks = [frag_matrix, maccs_matrix]
    feature_names = frag_names + [f"MACCS_{i}" for i in range(maccs_matrix.shape[1])]

    # Klekota-Roth
    if HAS_SKFP:
        print("  Computing Klekota-Roth...")
        kr_fp = KlekotaRothFingerprint(n_jobs=cfg.n_jobs, verbose=0)
        kr_matrix = kr_fp.transform(smiles_valid).astype(np.float32)
        kr_names = [f"KR_{i}" for i in range(kr_matrix.shape[1])]
        feature_blocks.append(kr_matrix)
        feature_names.extend(kr_names)

    X = np.hstack(feature_blocks)
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df.insert(0, "smiles", smiles_valid)          # keep SMILES for joining later

    print(f"  → Feature matrix: {X_df.shape}")
    return X_df, df_valid


# ---------------------------------------------------------------------------
# Feature selection (ONLY on training data)
# ---------------------------------------------------------------------------
def select_features_for_endpoint(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
) -> List[str]:
    # Drop SMILES column for modeling
    feature_cols = [c for c in X.columns if c != "smiles"]
    X_ = X[feature_cols]

    mask = y.notna()
    X_ = X_.loc[mask]
    y_ = y.loc[mask].astype(int)

    if len(y_) < 50 or y_.nunique() < 2:
        warnings.warn("Endpoint has too few labeled samples or only one class.")
        return []

    mi = mutual_info_classif(
        X_.values, y_.values,
        discrete_features="auto",
        random_state=cfg.random_state,
        n_neighbors=3,
    )
    mi_series = pd.Series(mi, index=X_.columns).sort_values(ascending=False)

    candidates = mi_series.head(min(cfg.top_k * 2, len(mi_series))).index.tolist()

    if not cfg.use_correlation_filter:
        return candidates[:cfg.top_k]

    # Optional correlation pruning
    selected = []
    corr = X_[candidates].corr().abs()
    for feat in candidates:
        if len(selected) >= cfg.top_k:
            break
        if not selected or corr.loc[feat, selected].max() < cfg.corr_threshold:
            selected.append(feat)
    return selected


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(cfg: Config = Config()):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- 1. Load splits ----
    print("Loading training set...")
    train_df = pd.read_parquet(cfg.train_path)
    print(f"  {len(train_df)} compounds")

    print("Loading validation set...")
    val_df = pd.read_parquet(cfg.val_path)
    print(f"  {len(val_df)} compounds")

    # ---- 2. Featurize both ----
    print("\nFeaturizing TRAIN set...")
    X_train, train_valid = build_feature_matrix(train_df, cfg)

    print("\nFeaturizing VALIDATION set...")
    X_val, val_valid = build_feature_matrix(val_df, cfg)

    # Save full feature matrices
    train_feat_path = os.path.join(cfg.output_dir, "train_features.parquet")
    val_feat_path   = os.path.join(cfg.output_dir, "val_features.parquet")
    X_train.to_parquet(train_feat_path, index=False)
    X_val.to_parquet(val_feat_path, index=False)
    print(f"\nSaved:\n  {train_feat_path}\n  {val_feat_path}")

    # ---- 3. Feature selection ONLY on training set ----
    print("\nRunning per-endpoint feature selection (on TRAIN only)...")
    selected_features = {}

    for endpoint in ENDPOINTS:
        if endpoint not in train_valid.columns:
            print(f"  [Skip] {endpoint} not found")
            continue

        print(f"  Selecting features for {endpoint} ...")
        feats = select_features_for_endpoint(X_train, train_valid[endpoint], cfg)
        selected_features[endpoint] = feats

        out_file = os.path.join(cfg.output_dir, f"selected_{endpoint}.txt")
        with open(out_file, "w") as f:
            f.write("\n".join(feats))
        print(f"    → {len(feats)} features → {out_file}")

    # Summary
    print("\n=== Selection Summary (from TRAIN) ===")
    for ep, feats in selected_features.items():
        print(f"{ep:20s}: {len(feats):3d} features")

    return selected_features, X_train, X_val


if __name__ == "__main__":
    cfg = Config()
    # cfg.use_correlation_filter = True
    # cfg.corr_threshold = 0.90
    # cfg.top_k = 200

    selected, X_train, X_val = run_pipeline(cfg)
    print("\nDone.")