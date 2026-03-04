"""
Task 3: Obtaining Vector Representations (TF-IDF)
ML Home Assignment - COVID-19 Fake News Detection

Steps:
  1. Load preprocessed train/val/test CSVs (output of Task 2)
  2. Fit TfidfVectorizer ONLY on training data (prevent data leakage)
  3. Transform train, val, and test sets
  4. Save vectorizer and transformed matrices for reuse in Task 4 models

Key Design Choices:
  - Fit ONLY on train to prevent data leakage
  - Use unigrams + bigrams (ngram_range=(1,2)) to capture context
  - sublinear_tf=True applies log normalization to term frequencies
  - max_features=20000 to limit dimensionality
  - min_df=2 to ignore very rare terms (likely noise)
  - max_df=0.95 to ignore terms that appear in almost all docs
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PROCESSED_DIR  = "data/processed"
VECTORS_DIR    = "data/vectors"
TEXT_COLUMN    = "tweet"
LABEL_COLUMN   = "label"

TFIDF_PARAMS = {
    "max_features" : 20000,
    "ngram_range"  : (1, 2),
    "sublinear_tf" : True,
    "min_df"       : 2,
    "max_df"       : 0.95,
    "strip_accents" : "unicode",
    "analyzer"     : "word",
    "token_pattern": r"\b[a-zA-Z_][a-zA-Z_]+\b",
}

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
def load_processed(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df   = pd.read_csv(path)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
    print(f"[INFO] Loaded {name}.csv — {len(df)} rows")
    return df


# -------------------------------------------------------------------
# FIT TFIDF
# -------------------------------------------------------------------
def fit_tfidf(train_texts: pd.Series) -> TfidfVectorizer:
    print("
[INFO] Fitting TF-IDF vectorizer on training data...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    vectorizer.fit(train_texts)
    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_)})
    print(f"[INFO] Feature names sample: {list(vectorizer.vocabulary_.keys())[:10]}")
    return vectorizer


# -------------------------------------------------------------------
# TRANSFORM
# -------------------------------------------------------------------
def transform(vectorizer: TfidfVectorizer, texts: pd.Series, name: str):
    print(f"[INFO] Transforming {name} set...")
    matrix = vectorizer.transform(texts)
    print(f"[INFO] {name} matrix shape: {matrix.shape}")
    return matrix


# -------------------------------------------------------------------
# SAVE / LOAD UTILITIES
# -------------------------------------------------------------------
def save_artifacts(vectorizer, X_train, X_val, X_test, y_train, y_val, y_test):
    os.makedirs(VECTORS_DIR, exist_ok=True)
    vec_path = os.path.join(VECTORS_DIR, "tfidf_vectorizer.pkl")
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Vectorizer saved -> {vec_path}")

    save_npz(os.path.join(VECTORS_DIR, "X_train.npz"), X_train)
    save_npz(os.path.join(VECTORS_DIR, "X_val.npz"),   X_val)
    save_npz(os.path.join(VECTORS_DIR, "X_test.npz"),  X_test)

    np.save(os.path.join(VECTORS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(VECTORS_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(VECTORS_DIR, "y_test.npy"),  y_test)

    print(f"[INFO] All matrices and labels saved to {VECTORS_DIR}/")


def load_artifacts():
    vec_path = os.path.join(VECTORS_DIR, "tfidf_vectorizer.pkl")
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    X_train = load_npz(os.path.join(VECTORS_DIR, "X_train.npz"))
    X_val   = load_npz(os.path.join(VECTORS_DIR, "X_val.npz"))
    X_test  = load_npz(os.path.join(VECTORS_DIR, "X_test.npz"))

    y_train = np.load(os.path.join(VECTORS_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(VECTORS_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(VECTORS_DIR, "y_test.npy"))

    print("[INFO] Artifacts loaded successfully.")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  X_test : {X_test.shape}")
    return vectorizer, X_train, X_val, X_test, y_train, y_val, y_test


# -------------------------------------------------------------------
# ANALYSIS UTILITY
# -------------------------------------------------------------------
def analyze_tfidf(vectorizer: TfidfVectorizer, X_train, y_train: np.ndarray):
    feature_names = np.array(vectorizer.get_feature_names_out())
    X_dense = X_train.toarray()

    for label, class_name in [(0, "FAKE"), (1, "REAL")]:
        mask        = (y_train == label)
        mean_tfidf  = X_dense[mask].mean(axis=0)
        top_indices = mean_tfidf.argsort()[::-1][:20]
        top_words   = feature_names[top_indices]
        print(f"\n[ANALYSIS] Top 20 TF-IDF features for class '{class_name}':")
        print(f"  {list(top_words)}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  TASK 3: TF-IDF Vector Representations")
    print("=" * 60)

    train_df = load_processed("train")
    val_df   = load_processed("val")
    test_df  = load_processed("test")

    X_train_text = train_df[TEXT_COLUMN]
    X_val_text   = val_df[TEXT_COLUMN]
    X_test_text  = test_df[TEXT_COLUMN]

    y_train = train_df[LABEL_COLUMN].values
    y_val   = val_df[LABEL_COLUMN].values
    y_test  = test_df[LABEL_COLUMN].values

    vectorizer = fit_tfidf(X_train_text)

    X_train = transform(vectorizer, X_train_text, "train")
    X_val   = transform(vectorizer, X_val_text,   "val")
    X_test  = transform(vectorizer, X_test_text,  "test")

    save_artifacts(vectorizer, X_train, X_val, X_test, y_train, y_val, y_test)

    analyze_tfidf(vectorizer, X_train, y_train)

    print("\n[DONE] Task 3 Complete! ✓")
    print("=" * 60)
