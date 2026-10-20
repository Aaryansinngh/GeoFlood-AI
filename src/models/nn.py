"""
src/models/nn.py

PyTorch Feed-Forward Neural Network for Flood Susceptibility Mapping.
Architecture:
- Input (D features)
- Layer 1: Linear(D -> 64) -> BatchNorm1d -> ReLU -> Dropout(0.2)
- Layer 2: Linear(64 -> 32) -> BatchNorm1d -> ReLU -> Dropout(0.2)
- Layer 3: Linear(32 -> 16) -> ReLU
- Output: Linear(16 -> 1) -> Sigmoid
- Loss: BCEWithLogitsLoss with pos_weight for class imbalance
"""

import copy
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score


class FloodSusceptibilityNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16], dropout: float = 0.2):
        super(FloodSusceptibilityNN, self).__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits for numerical stability in BCEWithLogitsLoss."""
        features = self.feature_extractor(x)
        logits = self.output_head(features)
        return logits.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[FloodSusceptibilityNN, Dict[str, List[float]]]:
    """
    Train the PyTorch NN with early stopping on validation loss/PR-AUC.
    """
    # Deterministic seeding
    seed = params.get("random_state", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Class imbalance weight: neg_count / pos_count
    pos_count = np.sum(y_train == 1.0)
    neg_count = np.sum(y_train == 0.0)
    pos_weight_val = float(neg_count / max(1, pos_count))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    input_dim = X_train.shape[1]
    hidden_dims = params.get("hidden_dims", [64, 32, 16])
    dropout = params.get("dropout", 0.2)
    lr = params.get("learning_rate", 0.001)
    weight_decay = params.get("weight_decay", 0.0001)
    batch_size = params.get("batch_size", 2048)
    epochs = params.get("epochs", 40)
    patience = params.get("early_stopping_patience", 7)

    model = FloodSusceptibilityNN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_x_t = torch.tensor(X_val, dtype=torch.float32)
    val_y_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": []
    }

    best_val_loss = float("inf")
    best_weights = None
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train_loss = float(np.mean(batch_losses))

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_t)
            val_loss = float(criterion(val_logits, val_y_t).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            try:
                val_pr = float(average_precision_score(y_val, val_probs))
            except ValueError:
                val_pr = 0.0

        history["train_loss"].append(round(avg_train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_pr_auc"].append(round(val_pr, 5))

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def predict_proba_nn(model: FloodSusceptibilityNN, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Predict probabilities for large arrays in batches."""
    model.eval()
    probs = []
    x_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i:i + batch_size]
            p = model.predict_proba(batch).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)
