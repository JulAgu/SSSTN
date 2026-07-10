import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from models.basic_mlp import MLP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def train_mlp(
    X_tr,
    y_tr,
    n_classes,
    hidden=128,
    n_layers=2,
    epochs=50,
    lr=1e-3,
    batch_size=64,
    device="cpu",
    log_every=2,
):
    """
    Parameters
    ----------
    X_tr : numpy.array
    y_tr : numpy.array
    n_classes : int
    hidden : int
    n_layers : int
    epochs : int
    lr : float
    batch_size : int
    device : str
    log_every: int
        Record a checkpoint (logits + metrics) every ``log_every`` epochs. The
        final epoch is always recorded.
        ``epochs=1000, log_every=2`` yields 500 checkpoints.

    Returns
    -------
    model : torch.nn.Module
        trained MLP
    scaler : sklearn.preprocessing.StandardScaler
        fitted StandardScaler
    trace : dict
        "logits" : list
            [(N, n_classes)] raw logit arrays (float16) on the training set, one per checkpoint
        "epoch_idx" : 0-based epoch index of each recorded checkpoint
        "n_sgd_steps" : total number of SGD steps taken over training
        "loss" : list of train cross-entropy, one per checkpoint
        "acc" : list of train accuracy, one per checkpoint
        "f1" : list of train macro-F1, one per checkpoint
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)

    Xt = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)

    model = MLP(Xt.shape[1], n_classes, hidden=hidden, n_layers=n_layers).to(device)
    print(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    trace = {
        "logits": [],
        "epoch_idx": [],
        "n_sgd_steps": 0,
        "loss": [],
        "acc": [],
        "f1": [],
    }

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            trace["n_sgd_steps"] += 1

        # Only checkpoint every `log_every` epochs (always include the last one).
        is_last = epoch == epochs - 1
        if (epoch + 1) % log_every != 0 and not is_last:
            continue

        # epoch-level metrics on the full training set
        model.eval()
        with torch.no_grad():
            logits = model(Xt)
            epoch_loss = loss_fn(logits, yt).item()
            logits_np = logits.cpu().numpy()
            preds = logits_np.argmax(1)

        # Store logits as float16 to keep the trace files small
        # (e.g 500 checkpoints * 30 runs for mnist_784 ~18Go)
        trace["logits"].append(logits_np.astype("float16").copy())
        trace["epoch_idx"].append(epoch)
        trace["loss"].append(epoch_loss)
        trace["acc"].append(accuracy_score(y_tr, preds))
        trace["f1"].append(f1_score(y_tr, preds, average="macro", zero_division=0))

    return model, scaler, trace


def train_mlp_regression(
    X_tr,
    y_tr,
    hidden=128,
    n_layers=2,
    epochs=50,
    lr=1e-3,
    batch_size=64,
    device="cpu",
    log_every=2,
):
    """
    Regression twin of ``train_mlp`` (single output, MSE loss).

    Parameters
    ----------
    X_tr : numpy.array
    y_tr : numpy.array
    hidden : int
    n_layers : int
    epochs : int
    lr : float
    batch_size : int
    device : str
    log_every : int
        Record a checkpoint (predictions + metrics) every ``log_every`` epochs.
        The final epoch is always recorded.

    Returns
    -------
    model : torch.nn.Module
        trained MLP
    scaler : sklearn.preprocessing.StandardScaler
        fitted StandardScaler
    trace : dict
        "logits" : list
            [(N, 1)] predictions (float16) on the training set, one per checkpoint
        "epoch_idx" : 0-based epoch index of each recorded checkpoint
        "n_sgd_steps" : total number of SGD steps taken over training
        "loss" : list of train MSE, one per checkpoint
        "mse" : list of train MSE, one per checkpoint
        "r2" : list of train R^2, one per checkpoint
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)

    Xt = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device).reshape(-1, 1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)

    model = MLP(Xt.shape[1], 1, hidden=hidden, n_layers=n_layers).to(device)
    print(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    trace = {
        "logits": [],
        "epoch_idx": [],
        "n_sgd_steps": 0,
        "loss": [],
        "mse": [],
        "r2": [],
    }

    y_np = np.asarray(y_tr, dtype="float32").reshape(-1)

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            trace["n_sgd_steps"] += 1

        # Only checkpoint every `log_every` epochs (always include the last one).
        is_last = epoch == epochs - 1
        if (epoch + 1) % log_every != 0 and not is_last:
            continue

        # epoch-level metrics on the full training set
        model.eval()
        with torch.no_grad():
            preds = model(Xt)
            epoch_loss = loss_fn(preds, yt).item()
            preds_np = preds.cpu().numpy().reshape(-1)

        # Store predictions as float16 to keep the trace files small; downstream
        # loss computation re-casts to float32.
        trace["logits"].append(preds_np.astype("float16").reshape(-1, 1).copy())
        trace["epoch_idx"].append(epoch)
        trace["loss"].append(epoch_loss)
        trace["mse"].append(mean_squared_error(y_np, preds_np))
        trace["r2"].append(r2_score(y_np, preds_np))

    return model, scaler, trace
