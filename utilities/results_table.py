"""Aggregate a folder of multirun ``.pkl`` results into a mean +- std table.

A *condition* is one ``(dataset, label_type)`` pair, where ``label_type`` is
``"random"`` for files carrying the ``randlabels_`` prefix (permuted
targets/labels) and ``"real"`` otherwise. Each ``.pkl`` in the folder is one run
(one weight init / seed); this module groups them by condition and reports the
train and test score as ``mean`` and ``std`` over the runs.

Per task the reported score is:

* **classification** -- accuracy (``final_train_acc`` / ``final_test_acc``);
* **regression**     -- ``R^2``.

Regression R^2, and the label asymmetry between conditions
------------------------------------------------------------
The targets are standardized on the train split, so train variance is exactly 1
and ``train R^2 = 1 - final_train_mse`` is exact.  In the **random** condition the
*train* targets are permuted, so this train R^2 measures the fit to the permuted
(noise) targets -- the memorization signal -- not a fit to real labels.

The **test** split keeps its TRUE targets in *both* conditions (the permutation
is applied to train only).  The true test R^2 needs the test set's own variance,
which is not ``1`` (the test split is scaled by *train* stats), so ``1 -
test_mse`` is only a pseudo-R^2.  We therefore recompute the exact value::

    true_test_R^2 = 1 - final_test_mse / Var(y_te_s)

reloading the true test targets from OpenML and applying the stored ``y_scaler``
(identical across all runs of a dataset).  Because the stored ``final_test_mse``
is already ``mean((pred - y_te_s)**2)`` in standardized units, this is
exact-consistent with the stored MSE -- no model rebuild, no recompute drift.
The true test targets are used for both conditions; the stored ``perm`` is never
applied to the test split.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def _label_type(filename: str) -> str:
    """``"random"`` for permuted-target/label runs, else ``"real"``."""
    return "random" if filename.startswith("randlabels_") else "real"


def _test_target_variance(exp: dict, cache: dict) -> float:
    """Population variance of the standardized TRUE test targets for this run.

    Reloads the dataset's true targets from OpenML, selects the same fold's test
    split, and applies the run's stored ``y_scaler``.  The permutation (random
    condition) is *not* applied -- it only affects train.  Cached per
    ``(did, fold)`` so the seeds of a condition don't re-download, and shared
    across real/random of the same dataset (their scaler fit is identical).
    """
    key = (int(exp["did"]), int(exp["fold"]))
    if key in cache:
        return cache[key]

    import openml  # lazy: only needed when a regression test R^2 is recomputed

    ds = openml.datasets.get_dataset(
        int(exp["did"]),
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    _, y, _, _ = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )
    task = openml.tasks.get_task(int(exp["tid"]), download_splits=True)
    _, test_idx = task.get_train_test_split_indices(fold=int(exp["fold"]), repeat=0)

    y_all = np.asarray(y.values, dtype="float32").reshape(-1, 1)
    y_te_s = exp["y_scaler"].transform(y_all[test_idx]).reshape(-1)
    var = float(np.var(y_te_s))  # ddof=0 to match the MSE / r2_score denominator
    cache[key] = var
    return var


def _run_scores(exp: dict, var_cache: dict) -> tuple[str, float, float]:
    """Return ``(metric_name, train_score, test_score)`` for one loaded run."""
    if exp.get("task") == "regression":
        train_r2 = 1.0 - exp["final_train_mse"]  # exact: train variance == 1
        var_te = _test_target_variance(exp, var_cache)
        test_r2 = 1.0 - exp["final_test_mse"] / var_te
        return "r2", train_r2, test_r2
    return "accuracy", exp["final_train_acc"], exp["final_test_acc"]


def summarize_folder(
    results_dir: str | Path,
    out_path: str | Path | None = None,
    *,
    ddof: int = 1,
    decimals: int = 4,
) -> dict:
    """Summarize every ``.pkl`` run in ``results_dir`` as mean +- std by dataset.

    Parameters
    ----------
    results_dir : folder of run ``.pkl`` files (non-recursive).
    out_path : if given, the JSON summary is also written here.
    ddof : delta-degrees-of-freedom for the std (1 = sample std; 0 = population).
        A single-run condition reports ``std = 0.0`` regardless.
    decimals : rounding for the reported numbers.

    Regression test R^2 is recomputed exactly (see module docstring), which
    downloads each dataset's test split from OpenML once.

    Returns
    -------
    dict keyed by dataset name; each value maps ``label_type`` ("real"/"random")
    to a record::

        {
          "task": "regression" | "classification",
          "metric": "r2" | "accuracy",
          "n_runs": int,
          "train": {"mean": float, "std": float},
          "test":  {"mean": float, "std": float},
          "train_pm_std": "0.9912 +- 0.0007",   # convenience string
          "test_pm_std":  "0.8543 +- 0.0211"
        }
    """
    results_dir = Path(results_dir)

    # Accumulate per (dataset, label_type): metric name + lists of train/test scores.
    grouped: dict[tuple[str, str], dict] = {}
    var_cache: dict[tuple[int, int], float] = {}
    for pkl_path in sorted(results_dir.glob("*.pkl")):
        try:
            with open(pkl_path, "rb") as f:
                exp = pickle.load(f)
            dataset = exp["name"]
            label_type = _label_type(pkl_path.name)
            metric, train_s, test_s = _run_scores(exp, var_cache)
        except Exception as e:  # skip unreadable / unexpected payloads, keep going
            print(f"skipping {pkl_path.name}: {e}")
            continue

        key = (dataset, label_type)
        acc = grouped.setdefault(
            key,
            {
                "task": exp.get("task", "classification"),
                "metric": metric,
                "train": [],
                "test": [],
            },
        )
        acc["train"].append(train_s)
        acc["test"].append(test_s)

    summary: dict[str, dict] = {}
    for (dataset, label_type), acc in sorted(grouped.items()):
        train = np.asarray(acc["train"], dtype=float)
        test = np.asarray(acc["test"], dtype=float)
        n = len(train)
        std = lambda a: round(float(a.std(ddof=ddof)) if n > 1 else 0.0, decimals)
        mean = lambda a: round(float(a.mean()), decimals)

        summary.setdefault(dataset, {})[label_type] = {
            "task": acc["task"],
            "metric": acc["metric"],
            "n_runs": n,
            "train": {"mean": mean(train), "std": std(train)},
            "test": {"mean": mean(test), "std": std(test)},
            "train_pm_std": f"{mean(train)} +- {std(train)}",
            "test_pm_std": f"{mean(test)} +- {std(test)}",
        }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_path}")

    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", help="folder of run *.pkl files")
    p.add_argument(
        "-o", "--out", default=None, help="write JSON summary here (else stdout)"
    )
    args = p.parse_args()

    result = summarize_folder(args.results_dir, args.out)
    if args.out is None:
        print(json.dumps(result, indent=2))
