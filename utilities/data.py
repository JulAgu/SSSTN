"""
Shared OpenML data access for the main experiments (A*, B*, C*, D*).

Every experiment draws continuous-only tasks from one of two OpenML suites --
CC18 (classification, suite 99) and CTR23 (regression, suite 353) -- downloads
them, and cuts the official train/test fold. These helpers mutualize that
fetch-and-prepare boilerplate so each experiment script keeps only what makes it
distinct (label corruption, multi-init, model type, ...).

Fetch and prepare are kept separate so both usage patterns fit: scripts that
pre-download every dataset before training (A*), and scripts that download one
dataset per loop iteration (B*, C*, D*).
"""

from __future__ import annotations

import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# suite id and human-readable label per task kind
SUITES = {
    "classification": (99, "CC18"),
    "regression": (353, "CTR23"),
}


def add_datasets_arg(parser, kind: str = "classification") -> None:
    """
    Add the shared ``--datasets`` filter to an argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    kind : {"classification", "regression"}
    """
    label = SUITES[kind][1]
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Dataset names to run. If omitted, all continuous-only {label} datasets are tested.",
    )


def select_tasks(
    kind: str = "classification",
    datasets: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Continuous-only tasks of the suite matching ``kind``, sorted by size.

    Parameters
    ----------
    kind : {"classification", "regression"}
        Selects CC18 (classification) or CTR23 (regression).
    datasets : list of str, optional
        If given, keep only these dataset names (a warning lists any that are
        not continuous-only members of the suite).
    verbose : bool
        Print suite counts and the selection.

    Returns
    -------
    selected : pandas.DataFrame
        Columns ``tid``, ``did``, ``name``, ``NumberOfInstances``,
        ``NumberOfFeatures`` (plus ``NumberOfClasses`` for classification),
        sorted ascending by ``NumberOfInstances``.
    """
    suite_id, label = SUITES[kind]
    suite = openml.study.get_suite(suite_id)
    tasks = openml.tasks.list_tasks(output_format="dataframe", task_id=suite.tasks)

    cols = ["tid", "did", "name", "NumberOfInstances", "NumberOfFeatures"]
    if kind == "classification":
        # NumberOfSymbolicFeatures counts the target too, so <= 1 means all covariates are continuous.
        mask = tasks["NumberOfSymbolicFeatures"] <= 1
        cols.append("NumberOfClasses")
    else:
        # The regression target is numeric, so it is NOT counted in
        # NumberOfSymbolicFeatures; == 0 therefore means all covariates are continuous.
        mask = tasks["NumberOfSymbolicFeatures"] == 0
    selected = tasks.loc[mask, cols]

    if verbose:
        print(f"{label} total: {len(tasks)}  |  continuous-only: {len(selected)}")

    if datasets is not None:
        missing = sorted(set(datasets) - set(selected["name"]))
        if missing:
            print(
                f"WARNING: requested datasets not in continuous-only {label}: {missing}"
            )
        selected = selected.loc[selected["name"].isin(datasets)]
        if verbose:
            print(f"Selected {len(selected)} dataset(s): {sorted(selected['name'])}")

    return selected.sort_values(by="NumberOfInstances").reset_index(drop=True)


def load_xy(did: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Download one OpenML dataset and return its (features, target) frame.

    Parameters
    ----------
    did : int
        OpenML dataset id.

    Returns
    -------
    X : pandas.DataFrame
    y : pandas.Series
    """
    ds = openml.datasets.get_dataset(
        int(did),
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X, y, _, _ = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )
    return X, y


def fold_indices(tid: int, fold: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Train/test row indices of the task's official split.

    Parameters
    ----------
    tid : int
        OpenML task id.
    fold : int

    Returns
    -------
    train_idx : numpy.array
    test_idx : numpy.array
    """
    task = openml.tasks.get_task(int(tid), download_splits=True)
    return task.get_train_test_split_indices(fold=fold, repeat=0)


def numeric_features(X: pd.DataFrame) -> np.ndarray:
    """
    Continuous feature matrix: numeric columns only, NaNs filled with 0.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    X_vals : numpy.array
        float32 array of shape (N, n_numeric_features)
    """
    return X.select_dtypes(include="number").fillna(0).values.astype("float32")


def prepare_classification(X, y, tid, fold=0):
    """
    Split and prepare a classification dataset for training.

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series
    tid : int
        OpenML task id (for the official fold split).
    fold : int

    Returns
    -------
    X_tr, y_tr : numpy.array
        train features (float32) and integer-encoded labels
    X_te, y_te : numpy.array
        test features (float32) and integer-encoded labels
    le : sklearn.preprocessing.LabelEncoder
        fitted label encoder
    n_classes : int
    """
    train_idx, test_idx = fold_indices(tid, fold)
    le = LabelEncoder()
    y_enc = np.asarray(le.fit_transform(y.values))
    X_vals = numeric_features(X)
    return (
        X_vals[train_idx],
        y_enc[train_idx],
        X_vals[test_idx],
        y_enc[test_idx],
        le,
        len(le.classes_),
    )


def prepare_regression(X, y, tid, fold=0):
    """
    Split and prepare a regression dataset for training.

    The continuous target is returned as a ``(N, 1)`` float32 array (unscaled);
    callers standardize ``y`` themselves so the MSE is in unit-variance units.

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series
    tid : int
        OpenML task id (for the official fold split).
    fold : int

    Returns
    -------
    X_tr, y_tr : numpy.array
        train features (float32) and target of shape (n_train, 1)
    X_te, y_te : numpy.array
        test features (float32) and target of shape (n_test, 1)
    """
    train_idx, test_idx = fold_indices(tid, fold)
    y_all = np.asarray(y.values, dtype="float32").reshape(-1, 1)
    X_vals = numeric_features(X)
    return X_vals[train_idx], y_all[train_idx], X_vals[test_idx], y_all[test_idx]
