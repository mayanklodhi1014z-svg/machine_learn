"""
Task 4 - Model 6: Gradient Boosting (Ensemble Learning)
ML Home Assignment - COVID-19 Fake News Detection

Hyperparameters tuned:
  - n_estimators  : number of boosting stages
  - learning_rate : shrinkage factor
  - max_depth     : depth of individual trees
  - subsample     : fraction of samples per tree
  - min_samples_split: minimum samples to split a node
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from scipy.sparse import load_npz

VECTORS_DIR  = "data/vectors"
RANDOM_STATE = 42

def load_data():
    X_train = load_npz(f"{VECTORS_DIR}/X_train.npz").toarray()
    X_val   = load_npz(f"{VECTORS_DIR}/X_val.npz").toarray()
    X_test  = load_npz(f"{VECTORS_DIR}/X_test.npz").toarray()
    y_train = np.load(f"{VECTORS_DIR}/y_train.npy")
    y_val   = np.load(f"{VECTORS_DIR}/y_val.npy")
    y_test  = np.load(f"{VECTORS_DIR}/y_test.npy")
    print("[INFO] Data loaded.")
    return X_train, X_val, X_test, y_train, y_val, y_test

def tune_gb(X_train, y_train):
    param_dist = {
        "n_estimators"     : [50, 100, 200, 300],
        "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
        "max_depth"        : [3, 4, 5, 7],
        "subsample"        : [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
    }
    print("\n[INFO] Starting RandomizedSearchCV for Gradient Boosting...")
    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )
    search.fit(X_train, y_train)
    print(f"\n[INFO] Best params : {search.best_params_}")
    print(f"[INFO] Best CV F1  : {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_

def evaluate(model, X, y, split_name="Test"):
    y_pred = model.predict(X)
    print(f"\n{'='*60}")
    print(f"  Gradient Boosting Evaluation — {split_name} Set")
    print(f"{'='*60}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
    print(f"Accuracy  : {accuracy_score(y, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y, y_pred):.4f}")
    print(f"Precision : {precision_score(y, y_pred):.4f}")
    print(f"Recall    : {recall_score(y, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y, y_pred, target_names=['Fake','Real'])}")

if __name__ == "__main__":
    print("="*60)
    print("  TASK 4 — Model 6: Gradient Boosting")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    best_model, best_params = tune_gb(X_train, y_train)
    evaluate(best_model, X_val, y_val, split_name="Validation")
    evaluate(best_model, X_test, y_test, split_name="Test")

    print("\n[DONE] Gradient Boosting Complete!")
    print(f"[BEST PARAMS] {best_params}")