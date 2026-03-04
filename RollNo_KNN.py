"""
Task 4 - Model 1: K-Nearest Neighbor (KNN)
ML Home Assignment - COVID-19 Fake News Detection

Hyperparameters tuned:
  - n_neighbors : number of neighbors K
  - metric      : distance metric
  - weights     : uniform or distance-weighted
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from scipy.sparse import load_npz

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
VECTORS_DIR  = "data/vectors"
RANDOM_STATE = 42

# -------------------------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------------------------
def load_data():
    X_train = load_npz(f"{VECTORS_DIR}/X_train.npz")
    X_val   = load_npz(f"{VECTORS_DIR}/X_val.npz")
    X_test  = load_npz(f"{VECTORS_DIR}/X_test.npz")
    y_train = np.load(f"{VECTORS_DIR}/y_train.npy")
    y_val   = np.load(f"{VECTORS_DIR}/y_val.npy")
    y_test  = np.load(f"{VECTORS_DIR}/y_test.npy")
    print("[INFO] Data loaded.")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------------------------------------------------
# HYPERPARAMETER TUNING
# -------------------------------------------------------------------
def tune_knn(X_train, y_train):
    param_grid = {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "metric"     : ["euclidean", "manhattan", "cosine"],
        "weights"    : ["uniform", "distance"],
    }
    print("\n[INFO] Starting GridSearchCV for KNN...")
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    # KNN with sparse + cosine works; convert to dense for euclidean/manhattan
    grid.fit(X_train, y_train)
    print(f"\n[INFO] Best params : {grid.best_params_}")
    print(f"[INFO] Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_

# -------------------------------------------------------------------
# EVALUATE
# -------------------------------------------------------------------
def evaluate(model, X, y, split_name="Test"):
    y_pred = model.predict(X)
    print(f"\n{'='*60}")
    print(f"  KNN Evaluation — {split_name} Set")
    print(f"{'='*60}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
    print(f"Accuracy  : {accuracy_score(y, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y, y_pred):.4f}")
    print(f"Precision : {precision_score(y, y_pred):.4f}")
    print(f"Recall    : {recall_score(y, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y, y_pred, target_names=['Fake','Real'])}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("  TASK 4 — Model 1: K-Nearest Neighbor")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Tune on train, evaluate on val
    best_model, best_params = tune_knn(X_train, y_train)

    # Validation performance
    evaluate(best_model, X_val, y_val, split_name="Validation")

    # Final test evaluation
    evaluate(best_model, X_test, y_test, split_name="Test")

    print("\n[DONE] KNN Complete!")
    print(f"[BEST PARAMS] {best_params}")
