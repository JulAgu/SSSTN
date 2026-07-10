"""
For gradient boosting the natural trajectory is the **sequence of boosting
rounds**: round ``k`` adds trees on top of round ``k-1``, so the prediction over
the training set after ``k`` rounds is the boosting analog of the MLP's
prediction after ``k`` epochs. We train the booster once and then snapshot its
prediction after each round via ``iteration_range=(0, k)``.

A booster is deterministic given the data and params
*except* for the stochasticity introduced by ``subsample`` / ``colsample_bytree``

We vary ``seed`` while using row/column subsampling < 1,
this makes each run a distinct boosting trajectory.
With ``subsample == colsample_bytree == 1`` every run
would be identical and the STN degenerate.
#TODO: Clean all this script
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def _checkpoint_rounds(n_estimators: int, log_every: int) -> list[int]:
    rounds = [k for k in range(1, n_estimators + 1) if k % log_every == 0]
    if not rounds or rounds[-1] != n_estimators:
        rounds.append(n_estimators)
    return rounds


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_xgb(
    X_tr,
    y_tr,
    n_classes,
    n_estimators=1000,
    max_depth=6,
    lr=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    log_every=2,
    seed=0,
    device="cpu",
):
    """
    Gradient-boosting analog of ``train_mlp``, one checkpoint per boosting round.

    Uses ``multi:softprob`` for every class count (binary included, where it
    yields a 2-column margin) so the descriptor is uniformly ``(N, n_classes)``.

    Parameters
    ----------
    X_tr : numpy.array
    y_tr : numpy.array
    n_classes : int
    n_estimators : int
    max_depth : int
    lr : float
    subsample : float
    colsample_bytree : float
    log_every : int
        Record a checkpoint every ``log_every`` rounds. The final round is
        always recorded.
    seed : int
    device : str

    Returns
    -------
    booster : xgboost.Booster
        trained booster
    trace : dict
        "logits" : list
            [(N, n_classes)] raw margin arrays (float16) on the training set, one per checkpoint
        "epoch_idx" : 0-based round index of each recorded checkpoint
        "n_sgd_steps" : total number of boosting rounds (= ``n_estimators``)
        "loss" : list of train multiclass log-loss, one per checkpoint
        "acc" : list of train accuracy, one per checkpoint
        "f1" : list of train macro-F1, one per checkpoint
    """
    y_np = np.asarray(y_tr).reshape(-1)
    dtrain = xgb.DMatrix(X_tr, label=y_np)
    params = {
        "objective": "multi:softprob",
        "num_class": int(n_classes),
        "eta": lr,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "tree_method": "hist",
        "device": device,
        "seed": seed,
        "eval_metric": "mlogloss",
    }
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators)

    trace = {
        "logits": [],
        "epoch_idx": [],
        "n_sgd_steps": n_estimators,
        "loss": [],
        "acc": [],
        "f1": [],
    }

    rows = np.arange(len(y_np))
    for k in _checkpoint_rounds(n_estimators, log_every):
        margin = booster.predict(dtrain, iteration_range=(0, k), output_margin=True)
        margin = margin.reshape(len(y_np), -1)
        preds = margin.argmax(1)
        probs = _softmax(margin.astype(np.float64))
        epoch_loss = float(-np.log(np.clip(probs[rows, y_np], 1e-12, None)).mean())

        trace["logits"].append(margin.astype("float16").copy())
        trace["epoch_idx"].append(k - 1)
        trace["loss"].append(epoch_loss)
        trace["acc"].append(accuracy_score(y_np, preds))
        trace["f1"].append(f1_score(y_np, preds, average="macro", zero_division=0))

    return booster, trace


def train_xgb_regression(
    X_tr,
    y_tr,
    n_estimators=1000,
    max_depth=6,
    lr=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    log_every=2,
    seed=0,
    device="cpu",
):
    """
    Regression twin of ``train_xgb`` (single output, squared-error loss).

    Parameters
    ----------
    X_tr : numpy.array
    y_tr : numpy.array
    n_estimators : int
    max_depth : int
    lr : float
    subsample : float
    colsample_bytree : float
    log_every : int
        Record a checkpoint every ``log_every`` rounds. The final round is
        always recorded.
    seed : int
    device : str

    Returns
    -------
    booster : xgboost.Booster
        trained booster
    trace : dict
        "logits" : list
            [(N, 1)] predictions (float16) on the training set, one per checkpoint
        "epoch_idx" : 0-based round index of each recorded checkpoint
        "n_sgd_steps" : total number of boosting rounds (= ``n_estimators``)
        "loss" : list of train MSE, one per checkpoint
        "mse" : list of train MSE, one per checkpoint
        "r2" : list of train R^2, one per checkpoint
    """
    y_np = np.asarray(y_tr, dtype="float32").reshape(-1)
    dtrain = xgb.DMatrix(X_tr, label=y_np)
    params = {
        "objective": "reg:squarederror",
        "eta": lr,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "tree_method": "hist",
        "device": device,
        "seed": seed,
        "eval_metric": "rmse",
    }
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators)

    trace = {
        "logits": [],
        "epoch_idx": [],
        "n_sgd_steps": n_estimators,
        "loss": [],
        "mse": [],
        "r2": [],
    }

    for k in _checkpoint_rounds(n_estimators, log_every):
        preds = booster.predict(dtrain, iteration_range=(0, k)).reshape(-1)
        epoch_mse = mean_squared_error(y_np, preds)

        trace["logits"].append(preds.astype("float16").reshape(-1, 1).copy())
        trace["epoch_idx"].append(k - 1)
        trace["loss"].append(epoch_mse)
        trace["mse"].append(epoch_mse)
        trace["r2"].append(r2_score(y_np, preds))

    return booster, trace
