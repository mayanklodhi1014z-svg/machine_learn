"""
Task 4 - Model 4: K-Means Clustering
ML Home Assignment - COVID-19 Fake News Detection

Since K-Means is unsupervised, after clustering we:
  - Map cluster IDs to class labels via majority voting
  - Evaluate using the mapped predictions

Hyperparameters tuned:
  - init     : k-means++ or random
  - max_iter : maximum iterations
  - n_init   : number of initializations
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from scipy.sparse import load_npz
from scipy.stats import mode

VECTORS_DIR  = "data/vectors"
RANDOM_STATE = 42

def load_data():
    X_train = load_npz(f"{VECTORS_DIR}/X_train.npz")
    X_val   = load_npz(f"{VECTORS_DIR}/X_val.npz")
    X_test  = load_npz(f"{VECTORS_DIR}/X_test.npz")
    y_train = np.load(f"{VECTORS_DIR}/y_train.npy")
    y_val   = np.load(f"{VECTORS_DIR}/y_val.npy")
    y_test  = np.load(f"{VECTORS_DIR}/y_test.npy")
    print("[INFO] Data loaded.")
    return X_train, X_val, X_test, y_train, y_val, y_test

def map_clusters_to_labels(cluster_labels, true_labels):
    """Map cluster IDs to true class labels via majority voting."""
    mapped = np.zeros_like(cluster_labels)
    for cluster_id in np.unique(cluster_labels):
        mask = (cluster_labels == cluster_id)
        majority = mode(true_labels[mask], keepdims=True).mode[0]
        mapped[mask] = majority
    return mapped

def tune_kmeans(X_train, y_train):
    param_grid = list(ParameterGrid({
        "init"    : ["k-means++", "random"],
        "max_iter": [100, 300, 500],
        "n_init"  : [10, 20],
    }))
    best_f1     = -1
    best_params = None
    best_model  = None

    print("\n[INFO] Starting manual grid search for K-Means...")
    X_dense = X_train.toarray()

    for params in param_grid:
        km = KMeans(n_clusters=2, random_state=RANDOM_STATE, **params)
        km.fit(X_dense)
        mapped = map_clusters_to_labels(km.labels_, y_train)
        score  = f1_score(y_train, mapped)
        if score > best_f1:
            best_f1     = score
            best_params = params
            best_model  = km

    print(f"\n[INFO] Best params : {best_params}")
    print(f"[INFO] Best F1     : {best_f1:.4f}")
    return best_model, best_params

def evaluate(model, X, y, y_train, split_name="Test"):
    X_dense    = X.toarray()
    cluster_ids = model.predict(X_dense)
    y_pred      = map_clusters_to_labels(cluster_ids, y)
    print(f"\n{'='*60}")
    print(f"  K-Means Evaluation — {split_name} Set")
    print(f"{'='*60}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
    print(f"Accuracy  : {accuracy_score(y, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y, y_pred):.4f}")
    print(f"Precision : {precision_score(y, y_pred):.4f}")
    print(f"Recall    : {recall_score(y, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y, y_pred, target_names=['Fake','Real'])}")

if __name__ == "__main__":
    print("="*60)
    print("  TASK 4 — Model 4: K-Means Clustering")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    best_model, best_params = tune_kmeans(X_train, y_train)
    evaluate(best_model, X_val, y_val, y_train, split_name="Validation")
    evaluate(best_model, X_test, y_test, y_train, split_name="Test")

    print("\n[DONE] K-Means Complete!")
    print(f"[BEST PARAMS] {best_params}")