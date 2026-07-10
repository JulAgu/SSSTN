"""
For each generation we preserve the *best individual*

To expose the itretive process and create traces as for the MLPs and XGBs, we use
PySR's ``warm_start``, that allows to do the search one iteration at a time
(``niterations=1`` per ``fit`` call, continuing the same population).

The B/C scripts vary the MLP weight init across runs.
The XGBoost twin varies the subsampling ``seed``.
Here the analog is the genetic algorithm's ``random_state``.
We default to ``deterministic`` and use serial evolution,
so that each ``seed`` maps reproducibly to a distinct.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from pysr import PySRRegressor


def _best_index(equations) -> int:
    """
    Row index of the *best individual*: the lowest-loss row of the current hall
    of fame, regardless of complexity. Picking a single individual avoids having
    to track a whole solution set at each step; tracking several would also be a
    valid (and interesting) use of STN semantic spaces.

    Parameters
    ----------
    equations : pandas.DataFrame
        A ``PySRRegressor.equations_`` table (one row per equation).

    Returns
    -------
    best_index : int
        row index of the lowest-loss equation
    """
    best_index = int(equations["loss"].idxmin())
    return best_index


def train_sr_regression(
    X_tr,
    y_tr,
    n_generations=40,
    log_every=1,
    seed=0,
    deterministic=True,
    parallelism="serial",
    **pysr_kwargs,
):
    """
    Symbolic-regression analog of ``train_mlp``, one checkpoint per generation.

    Parameters
    ----------
    X_tr : numpy.array
    y_tr : numpy.array
    n_generations : int
        Number of evolutionary generations to run (one ``warm_start`` iteration each).
    log_every : int
        Record a checkpoint every ``log_every`` generations. The final
        generation is always recorded.
    seed : int
        Genetic algorithm's ``random_state``.
    deterministic : bool
    parallelism : str
    pysr_kwargs :
        Extra keyword arguments forwarded to ``PySRRegressor`` (omitting them
        uses PySR's defaults).

    Returns
    -------
    model : pysr.PySRRegressor
        the "trained" PySRRegressor object
    best_index : int
        row index of the best individual in ``model.equations_``
    trace : dict
        "logits" : list
            [(N, 1)] predictions (float16) of the best individual, one per checkpoint
        "epoch_idx" : 0-based generation index of each recorded checkpoint
        "n_sgd_steps" : total number of generations (= ``n_generations``)
        "loss" : list of train MSE, one per checkpoint
        "mse" : list of train MSE, one per checkpoint
        "r2" : list of train R^2, one per checkpoint
    """
    X_np = np.asarray(X_tr, dtype="float64")
    y_np = np.asarray(y_tr, dtype="float64").reshape(-1)

    model = PySRRegressor(
        niterations=1,  # one generation per fit() call; the loop accumulates them
        warm_start=True,  # continue the same population across calls
        random_state=seed,
        deterministic=deterministic,
        parallelism=parallelism,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
        **pysr_kwargs,
    )

    trace = {
        "logits": [],
        "epoch_idx": [],
        "n_sgd_steps": n_generations,
        "loss": [],
        "mse": [],
        "r2": [],
    }

    for gen in range(n_generations):
        # One generation of evolution, continuing from the previous population.
        model.fit(X_np, y_np)

        is_last = gen == n_generations - 1
        if (gen + 1) % log_every != 0 and not is_last:
            continue

        best_idx = _best_index(model.equations_)
        preds = np.asarray(
            model.predict(X_np, index=best_idx), dtype="float64"
        ).reshape(-1)
        gen_mse = mean_squared_error(y_np, preds)

        # Store predictions as float16 to keep the trace small; downstream loss
        trace["logits"].append(preds.astype("float16").reshape(-1, 1).copy())
        trace["epoch_idx"].append(gen)
        trace["loss"].append(gen_mse)
        trace["mse"].append(gen_mse)
        trace["r2"].append(r2_score(y_np, preds))

    return model, _best_index(model.equations_), trace
