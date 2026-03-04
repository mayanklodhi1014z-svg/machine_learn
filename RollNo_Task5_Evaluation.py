"""
Task 5: Evaluating Machine Learning Models
ML Home Assignment - COVID-19 Fake News Detection

This script:
  1. Loads TF-IDF vectors and labels (output of Task 3)
  2. Trains and tunes all 6 models
  3. Evaluates each on the test split
  4. Prints confusion matrix, accuracy, f1, precision, recall
  5. Summarizes results in a comparison table

Models evaluated:
  1. K-Nearest Neighbor
  2. Logistic Regression
  3. Support Vector Machine (Linear Kernel)
  4. K-Means Clustering
  5. Neural Network (PyTorch)
  6. Gradient Boosting
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import load_npz
from scipy.stats import mode

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
VECTORS_DIR  = "data/vectors"
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
def load_data():
    X_train = load_npz(f"{VECTORS_DIR}/X_train.npz")
    X_val   = load_npz(f"{VECTORS_DIR}/X_val.npz")
    X_test  = load_npz(f"{VECTORS_DIR}/X_test.npz")
    y_train = np.load(f"{VECTORS_DIR}/y_train.npy")
    y_val   = np.load(f"{VECTORS_DIR}/y_val.npy")
    y_test  = np.load(f"{VECTORS_DIR}/y_test.npy")
    print(f"[INFO] Data loaded.")
    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------------------------------------------------
# EVALUATION UTILITY
# -------------------------------------------------------------------
def evaluate_model(y_true, y_pred, model_name):
    """Print full evaluation metrics for a model."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    cm = confusion_matrix(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted")
    rec  = recall_score(y_true, y_pred, average="weighted")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"\nClassification Report:\n"
          f"{classification_report(y_true, y_pred, target_names=['Fake (0)', 'Real (1)'])}")
    return {"Model": model_name, "Accuracy": acc, "F1": f1,
            "Precision": prec, "Recall": rec}

# -------------------------------------------------------------------
# 1. K-NEAREST NEIGHBOR
# -------------------------------------------------------------------
def run_knn(X_train, y_train, X_test, y_test):
    print("\n[MODEL 1] K-Nearest Neighbor — Hyperparameter Tuning...")
    param_grid = {
        "n_neighbors": [3, 5, 7, 11, 15],
        "metric"     : ["euclidean", "manhattan", "cosine"],
        "weights"    : ["uniform", "distance"],
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid,
                        cv=5, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_} | CV F1: {grid.best_score_:.4f}")
    y_pred  = grid.best_estimator_.predict(X_test)
    results = evaluate_model(y_test, y_pred, "1. K-Nearest Neighbor")
    results["Best Params"] = grid.best_params_
    return results

# -------------------------------------------------------------------
# 2. LOGISTIC REGRESSION
# -------------------------------------------------------------------
def run_lr(X_train, y_train, X_test, y_test):
    print("\n[MODEL 2] Logistic Regression — Hyperparameter Tuning...")
    param_grid = {
        "C"       : [0.01, 0.1, 1, 10, 100],
        "penalty" : ["l1", "l2"],
        "solver"  : ["liblinear", "saga"],
        "max_iter": [200, 500],
    }
    grid = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE),
                        param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_} | CV F1: {grid.best_score_:.4f}")
    y_pred  = grid.best_estimator_.predict(X_test)
    results = evaluate_model(y_test, y_pred, "2. Logistic Regression")
    results["Best Params"] = grid.best_params_
    return results

# -------------------------------------------------------------------
# 3. SUPPORT VECTOR MACHINE
# -------------------------------------------------------------------
def run_svm(X_train, y_train, X_test, y_test):
    print("\n[MODEL 3] SVM (Linear Kernel) — Hyperparameter Tuning...")
    param_grid = {
        "estimator__C"   : [0.001, 0.01, 0.1, 1, 10],
        "estimator__loss": ["hinge", "squared_hinge"],
    }
    base = CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
    )
    grid = GridSearchCV(base, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"  Best params: {grid.best_params_} | CV F1: {grid.best_score_:.4f}")
    y_pred  = grid.best_estimator_.predict(X_test)
    results = evaluate_model(y_test, y_pred, "3. SVM (Linear Kernel)")
    results["Best Params"] = grid.best_params_
    return results

# -------------------------------------------------------------------
# 4. K-MEANS CLUSTERING
# -------------------------------------------------------------------
def map_clusters(cluster_labels, true_labels):
    mapped = np.zeros_like(cluster_labels)
    for cid in np.unique(cluster_labels):
        mask     = (cluster_labels == cid)
        majority = mode(true_labels[mask], keepdims=True).mode[0]
        mapped[mask] = majority
    return mapped

def run_kmeans(X_train, y_train, X_test, y_test):
    print("\n[MODEL 4] K-Means Clustering — Hyperparameter Tuning...")
    param_grid = list(ParameterGrid({
        "init"    : ["k-means++", "random"],
        "max_iter": [100, 300],
        "n_init"  : [10, 20],
    }))
    X_train_d = X_train.toarray()
    X_test_d  = X_test.toarray()
    best_f1, best_params, best_model = -1, None, None

    for params in param_grid:
        km     = KMeans(n_clusters=2, random_state=RANDOM_STATE, **params)
        km.fit(X_train_d)
        mapped = map_clusters(km.labels_, y_train)
        score  = f1_score(y_train, mapped)
        if score > best_f1:
            best_f1, best_params, best_model = score, params, km

    print(f"  Best params: {best_params} | Best F1: {best_f1:.4f}")
    cluster_ids = best_model.predict(X_test_d)
    y_pred      = map_clusters(cluster_ids, y_test)
    results     = evaluate_model(y_test, y_pred, "4. K-Means Clustering")
    results["Best Params"] = best_params
    return results

# -------------------------------------------------------------------
# 5. NEURAL NETWORK (PyTorch)
# -------------------------------------------------------------------
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_nn(model, train_loader, val_loader, lr, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_val_f1, best_weights = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                p = (model(X_b.to(DEVICE)) >= 0.5).long().cpu().numpy()
                preds.extend(p)
                labels.extend(y_b.long().numpy())
        val_f1 = f1_score(labels, preds)
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    return model, best_val_f1

def run_nn(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n[MODEL 5] Neural Network (PyTorch) — Hyperparameter Tuning...")
    X_tr = X_train.toarray().astype(np.float32)
    X_v  = X_val.toarray().astype(np.float32)
    X_te = X_test.toarray().astype(np.float32)
    y_tr = y_train.astype(np.float32)
    y_v  = y_val.astype(np.float32)
    input_dim = X_tr.shape[1]

    param_grid = [
        {"hidden_dim": 256, "dropout": 0.3, "lr": 1e-3, "batch_size": 64,  "epochs": 20},
        {"hidden_dim": 512, "dropout": 0.3, "lr": 1e-3, "batch_size": 64,  "epochs": 20},
        {"hidden_dim": 256, "dropout": 0.5, "lr": 1e-4, "batch_size": 32,  "epochs": 30},
        {"hidden_dim": 512, "dropout": 0.2, "lr": 1e-3, "batch_size": 128, "epochs": 20},
    ]
    best_f1, best_params, best_model = 0.0, None, None

    for params in param_grid:
        print(f"  Trying: {params}")
        model = FakeNewsClassifier(input_dim, params["hidden_dim"], params["dropout"]).to(DEVICE)
        t_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                              batch_size=params["batch_size"], shuffle=True)
        v_loader = DataLoader(TensorDataset(torch.tensor(X_v),  torch.tensor(y_v)),
                              batch_size=params["batch_size"], shuffle=False)
        model, val_f1 = train_nn(model, t_loader, v_loader, params["lr"], params["epochs"])
        if val_f1 > best_f1:
            best_f1, best_params, best_model = val_f1, params, model

    print(f"  Best params: {best_params} | Best Val F1: {best_f1:.4f}")
    best_model.eval()
    with torch.no_grad():
        y_pred = (best_model(torch.tensor(X_te).to(DEVICE)) >= 0.5).long().cpu().numpy()
    results = evaluate_model(y_test.astype(int), y_pred, "5. Neural Network (PyTorch)")
    results["Best Params"] = best_params
    return results

# -------------------------------------------------------------------
# 6. GRADIENT BOOSTING
# -------------------------------------------------------------------
def run_gb(X_train, y_train, X_test, y_test):
    print("\n[MODEL 6] Gradient Boosting — Hyperparameter Tuning...")
    param_dist = {
        "n_estimators"     : [50, 100, 200, 300],
        "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
        "max_depth"        : [3, 4, 5, 7],
        "subsample"        : [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=20, cv=5, scoring="f1",
        n_jobs=-1, verbose=0, random_state=RANDOM_STATE
    )
    search.fit(X_train.toarray(), y_train)
    print(f"  Best params: {search.best_params_} | CV F1: {search.best_score_:.4f}")
    y_pred  = search.best_estimator_.predict(X_test.toarray())
    results = evaluate_model(y_test, y_pred, "6. Gradient Boosting")
    results["Best Params"] = search.best_params_
    return results

# -------------------------------------------------------------------
# SUMMARY TABLE
# -------------------------------------------------------------------
def print_summary(all_results):
    print("\n" + "="*80)
    print("  FINAL RESULTS SUMMARY (Test Set)")
    print("="*80)
    print(f"{'Model':<40} {'Accuracy':>9} {'F1':>9} {'Precision':>10} {'Recall':>9}")
    print("-"*80)
    for r in all_results:
        print(f"{r['Model']:<40} {r['Accuracy']:>9.4f} {r['F1']:>9.4f} "
              f"{r['Precision']:>10.4f} {r['Recall']:>9.4f}")
    print("="*80)
    best = max(all_results, key=lambda x: x["F1"])
    print(f"\n[BEST MODEL] {best['Model']} with F1={best['F1']:.4f}")
    print("\n[BEST HYPERPARAMETERS PER MODEL]")
    for r in all_results:
        print(f"  {r['Model']}: {r['Best Params']}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("  TASK 5: Evaluating All ML Models")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    all_results = []
    all_results.append(run_knn(X_train, y_train, X_test, y_test))
    all_results.append(run_lr(X_train,  y_train, X_test, y_test))
    all_results.append(run_svm(X_train, y_train, X_test, y_test))
    all_results.append(run_kmeans(X_train, y_train, X_test, y_test))
    all_results.append(run_nn(X_train, y_train, X_val, y_val, X_test, y_test))
    all_results.append(run_gb(X_train, y_train, X_test, y_test))

    print_summary(all_results)

    print("\n[DONE] Task 5 Complete!")
    print("="*60)