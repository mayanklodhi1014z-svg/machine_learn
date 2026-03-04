"""
Task 1: Prepare Dataset
ML Home Assignment - COVID-19 Fake News Detection
Constraint@AAAI-2021 Dataset

Steps:
  1. Load the raw dataset CSV
  2. Encode labels: fake -> 0, real -> 1
  3. Split into train (80%), val (10%), test (10%) using sklearn train_test_split with shuffle=True
  4. Save splits to data/splits/train.csv, val.csv, test.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RAW_DATA_PATH = "data/raw/dataset.csv"   # <-- update path if needed
SPLITS_DIR    = "data/splits"
RANDOM_STATE  = 42

# ────────────────────────────────────���────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}")
    return df

# ─────────────────────────────────────────────
# STEP 2: Encode Labels
# ─────────────────────────────────────────────
def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode text labels to binary integers:
        'fake' -> 0
        'real' -> 1
    """
    label_map = {'fake': 0, 'real': 1}
    df['label'] = df['label'].str.strip().str.lower().map(label_map)
    assert df['label'].isnull().sum() == 0, "[ERROR] Some labels could not be mapped!"
    print(f"[INFO] Labels encoded: fake=0, real=1")
    print(f"[INFO] Encoded distribution:\n{df['label'].value_counts()}")
    return df

# ─────────────────────────────────────────────
# STEP 3: Split Dataset
# ─────────────────────────────────────────────
def split_dataset(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """
    Split dataset into:
        Train : 80%
        Val   : 10%
        Test  : 10%
    Uses shuffle=True for randomness and stratify on label for balanced splits.
    """
    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        shuffle=True,
        stratify=df['label'],
        random_state=random_state
    )

    # Second split: temp -> 50% val, 50% test  => 10% val, 10% test overall
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        shuffle=True,
        stratify=temp_df['label'],
        random_state=random_state
    )

    print(f"\n[INFO] Split sizes:")
    print(f"  Train : {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df

# ─────────────────────────────────────────────
# STEP 4: Save Splits to CSV
# ─────────────────────────────────────────────
def save_splits(train_df, val_df, test_df, output_dir: str = SPLITS_DIR):
    """Save train/val/test splits as CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"\n[INFO] Splits saved:")
    print(f"  -> {train_path}")
    print(f"  -> {val_path}")
    print(f"  -> {test_path}")

# ─────────────────────────────────────────────
# STEP 5: Verify Splits
# ─────────────────────────────────────────────
def verify_splits(train_df, val_df, test_df):
    """Print label distribution for each split to verify balance."""
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df['label'].value_counts()
        print(f"\n[VERIFY] {name} label distribution:")
        print(f"  Fake (0): {dist.get(0, 0)}")
        print(f"  Real (1): {dist.get(1, 0)}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TASK 1: Prepare Dataset")
    print("=" * 60)

    # 1. Load
    df = load_dataset(RAW_DATA_PATH)

    # 2. Encode labels
    df = encode_labels(df)

    # 3. Split
    train_df, val_df, test_df = split_dataset(df)

    # 4. Save
    save_splits(train_df, val_df, test_df)

    # 5. Verify
    verify_splits(train_df, val_df, test_df)

    print("\n[DONE] Task 1 Complete! ✓")
    print("=" * 60)