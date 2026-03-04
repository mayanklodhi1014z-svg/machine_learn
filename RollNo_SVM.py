"""
Task 4 - Model 3: Support Vector Machine (Linear Kernel)
ML Home Assignment - COVID-19 Fake News Detection

Hyperparameters tuned:
  - C    : regularization parameter
  - loss : hinge or squared_hinge
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from scipy.sparse import load_npz

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

def tune_svm(X_train, y_train):
    param_grid = {
        "estimator__C"   : [0.001, 0.01, 0.1, 1, 10],
        "estimator__loss": ["hinge", "squared_hinge"],
    }
    base = CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
    )
    print("\n[INFO] Starting GridSearchCV for SVM (Linear Kernel)...")
    grid = GridSearchCV(base, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"\n[INFO] Best params : {grid.best_params_}")
    print(f"[INFO] Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_

def evaluate(model, X, y, split_name="Test"):
    y_pred = model.predict(X)
    print(f"\n{'='*60}")
    print(f"  SVM Evaluation — {split_name} Set")
    print(f"{'='*60}")
    print(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
    print(f"Accuracy  : {accuracy_score(y, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y, y_pred):.4f}")
    print(f"Precision : {precision_score(y, y_pred):.4f}")
    print(f"Recall    : {recall_score(y, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y, y_pred, target_names=['Fake','Real'])}")

if __name__ == "__main__":
    print("="*60)
    print("  TASK 4 — Model 3: SVM Linear Kernel")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    best_model, best_params = tune_svm(X_train, y_train)
    evaluate(best_model, X_val, y_val, split_name="Validation")
    evaluate(best_model, X_test, y_test, split_name="Test")

    print("\n[DONE] SVM Complete!")
    print(f"[BEST PARAMS] {best_params}")