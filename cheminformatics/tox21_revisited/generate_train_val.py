#!/usr/bin/env python3
"""
Butina split for TOX21
----------------------
Creates train.parquet and val.parquet using Butina clustering.
"""

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from joblib import Parallel, delayed


# -----------------------------
# Config
# -----------------------------
HF_DATASET = "HUBioDataLab/tox21"
SMILES_COL = "smiles"
OUTPUT_DIR = "tox21_split"
TRAIN_FRAC = 0.8          # approximate
CUTOFF = 0.6              # Tanimoto distance cutoff (0.6 is common)
N_JOBS = -1
RANDOM_SEED = 42

ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


def load_data() -> pd.DataFrame:
    print(f"Loading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET)
    if "train" in ds:
        df = ds["train"].to_pandas()
    else:
        df = ds[list(ds.keys())[0]].to_pandas()

    df = df.dropna(subset=[SMILES_COL]).reset_index(drop=True)
    print(f"Loaded {len(df)} compounds")
    return df


def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def compute_fingerprints(smiles_list: List[str], n_jobs: int = -1):
    print("Computing Morgan fingerprints in parallel...")
    fps = Parallel(n_jobs=n_jobs)(
        delayed(smiles_to_fp)(smi) for smi in smiles_list
    )
    return fps


def butina_split(
    smiles_list: List[str],
    cutoff: float = 0.6,
    train_frac: float = 0.8,
    seed: int = 42,
    n_jobs: int = -1,
) -> Tuple[List[int], List[int]]:
    """
    Returns train_idx, val_idx
    """
    fps = compute_fingerprints(smiles_list, n_jobs=n_jobs)

    # Filter invalid molecules
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    fps = [fps[i] for i in valid_idx]
    print(f"Valid molecules for clustering: {len(fps)}")

    # Pairwise distances (Tanimoto distance = 1 - similarity)
    print("Calculating distance matrix (this can take a bit)...")
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = Chem.DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    # Butina clustering
    print("Running Butina clustering...")
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    clusters = sorted(clusters, key=lambda x: -len(x))  # largest first

    # Assign clusters to train / val
    rng = np.random.default_rng(seed)
    train_idx_local = []
    val_idx_local = []

    for cluster in clusters:
        cluster = list(cluster)
        if len(train_idx_local) / nfps < train_frac:
            train_idx_local.extend(cluster)
        else:
            val_idx_local.extend(cluster)

    # Map back to original indices
    train_idx = [valid_idx[i] for i in train_idx_local]
    val_idx   = [valid_idx[i] for i in val_idx_local]

    print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")
    return train_idx, val_idx


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    smiles_list = df[SMILES_COL].tolist()

    train_idx, val_idx = butina_split(
        smiles_list,
        cutoff=CUTOFF,
        train_frac=TRAIN_FRAC,
        seed=RANDOM_SEED,
        n_jobs=N_JOBS,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path   = os.path.join(OUTPUT_DIR, "val.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nSaved:")
    print(f"  {train_path}  ({len(train_df)} rows)")
    print(f"  {val_path}    ({len(val_df)} rows)")


if __name__ == "__main__":
    main()