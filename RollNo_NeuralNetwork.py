"""
Task 4 - Model 5: Neural Network (PyTorch)
ML Home Assignment - COVID-19 Fake News Detection

Architecture:
  Input (TF-IDF) -> Linear -> BatchNorm -> ReLU -> Dropout
                 -> Linear -> BatchNorm -> ReLU -> Dropout
                 -> Linear(1) -> Sigmoid

Hyperparameters tuned:
  - hidden_dim   : size of hidden layers
  - dropout      : dropout rate
  - learning_rate: optimizer LR
  - batch_size   : mini-batch size
  - epochs       : training epochs
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from scipy.sparse import load_npz

VECTORS_DIR  = "data/vectors"
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# -------------------------------------------------------------------
# MODEL DEFINITION
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

# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------
def load_data():
    X_train = load_npz(f"{VECTORS_DIR}/X_train.npz").toarray().astype(np.float32)
    X_val   = load_npZ(f"{VECTORS_DIR}/X_val.npz").toarray().astype(np.float32)
    X_test  = load_npZ(f"{VECTORS_DIR}/X_test.npz").toarray().astype(np.float32)
    y_train = np.load(f"{VECTORS_DIR}/y_train.npy").astype(np.float32)
    y_val   = np.load(f"{VECTORS_DIR}/y_val.npy").astype(np.float32)
    y_test  = np.load(f"{VECTORS_DIR}/y_test.npy").astype(np.float32)
    print(f"[INFO] Data loaded. Input dim: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def make_loader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------
def train_model(model, train_loader, val_loader, lr, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_f1  = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                preds   = (model(X_batch) >= 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.long().numpy())

        val_f1 = f1_score(all_labels, all_preds)
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    model.load_state_dict(best_weights)
    print(f"[INFO] Best Val F1: {best_val_f1:.4f}")
    return model

# -------------------------------------------------------------------
# HYPERPARAMETER SEARCH
# -------------------------------------------------------------------
def tune_nn(X_train, y_train, X_val, y_val):
    input_dim = X_train.shape[1]
    param_grid = [
        {"hidden_dim": 256, "dropout": 0.3, "lr": 1e-3, "batch_size": 64,  "epochs": 20},
        {"hidden_dim": 512, "dropout": 0.3, "lr": 1e-3, "batch_size": 64,  "epochs": 20},
        {"hidden_dim": 256, "dropout": 0.5, "lr": 1e-4, "batch_size": 32,  "epochs": 30},
        {"hidden_dim": 512, "dropout": 0.2, "lr": 1e-3, "batch_size": 128, "epochs": 20},
    ]

    best_f1     = 0.0
    best_params = None
    best_model  = None

    for params in param_grid:
        print(f"\n[TUNE] Trying params: {params}")
        model = FakeNewsClassifier(input_dim, params["hidden_dim"], params["dropout"]).to(DEVICE)
        train_loader = make_loader(X_train, y_train, params["batch_size"])
        val_loader   = make_loader(X_val,   y_val,   params["batch_size"], shuffle=False)
        model = train_model(model, train_loader, val_loader, params["lr"], params["epochs"])

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            preds   = (model(X_val_t) >= 0.5).long().cpu().numpy()
        score = f1_score(y_val.astype(int), preds)

        if score > best_f1:
            best_f1     = score
            best_params = params
            best_model  = model

    print(f"\n[INFO] Best params : {best_params}")
    print(f"[INFO] Best Val F1 : {best_f1:.4f}")
    return best_model, best_params

# -------------------------------------------------------------------
# EVALUATE
# -------------------------------------------------------------------
def evaluate(model, X, y, split_name="Test"):
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        preds = (model(X_t) >= 0.5).long().cpu().numpy()
    y_int = y.astype(int)
    print(f"\n{'='*60}")
    print(f"  Neural Network Evaluation — {split_name} Set")
    print(f"{'='*60}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_int, preds)}")
    print(f"Accuracy  : {accuracy_score(y_int, preds):.4f}")
    print(f"F1-Score  : {f1_score(y_int, preds):.4f}")
    print(f"Precision : {precision_score(y_int, preds):.4f}")
    print(f"Recall    : {recall_score(y_int, preds):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_int, preds, target_names=['Fake','Real'])}")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("  TASK 4 — Model 5: Neural Network (PyTorch)")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    best_model, best_params = tune_nn(X_train, y_train, X_val, y_val)

    evaluate(best_model, X_val,  y_val,  split_name="Validation")
    evaluate(best_model, X_test, y_test, split_name="Test")

    print("\n[DONE] Neural Network Complete!")
    print(f"[BEST PARAMS] {best_params}")