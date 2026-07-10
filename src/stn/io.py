"""Discover, group and load NN multiruns into a long trajectory table.

A *condition* is one ``(dataset, label_type)`` pair (e.g. ``("madelon", "real")``).
Its ``.pkl`` files -- one per seed/init -- are the euroGP "runs".  This is the
analog of euroGP reading a folder of run traces (``create.R`` ``fread`` loop).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .build import STN

# Filename: [randlabels_]fold0_layers2_hidden512_epc1000_ninits20__seed{N}_{dataset}.pkl
_FNAME_RE = re.compile(
    r"^(?P<rand>randlabels_)?.*__seed(?P<seed>\d+)_(?P<dataset>.+)\.pkl$"
)


def parse_condition(path: Path) -> tuple[str, str, int]:
    """Return (dataset, label_type, seed) parsed from a result filename."""
    m = _FNAME_RE.match(path.name)
    if m is None:
        raise ValueError(f"Unrecognised result filename: {path.name}")
    label_type = "random" if m.group("rand") else "real"
    return m.group("dataset"), label_type, int(m.group("seed"))


def discover_conditions(
    results_dir: str | Path,
) -> dict[tuple[str, str], list[Path]]:
    """Group ``results/*.pkl`` files by ``(dataset, label_type)``.

    Returns a dict mapping each condition to its list of run files, sorted by
    seed so run numbering is deterministic.
    """
    results_dir = Path(results_dir)
    conditions: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for path in results_dir.glob("*.pkl"):
        dataset, label_type, seed = parse_condition(path)
        conditions.setdefault((dataset, label_type), []).append((seed, path))
    return {
        key: [p for _, p in sorted(seeded)]
        for key, seeded in sorted(conditions.items())
    }


def _load_run(
    path: Path, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load one run.

    Parameters
    ----------
    path : pathlib.Path
    stride : int
        keep every ``stride``-th snapshot (coarsens the trajectory, preserving
        order and endpoints); 1 keeps the full trace

    Returns
    -------
    descriptors : numpy.array
        (n_snapshots, n_train * n_classes) flattened logits (n_classes == 1 for regression)
    fitness : numpy.array
        (n_snapshots,) per-snapshot fitness, maximised: train accuracy
        (classification) or train R^2 (regression, scale-free so "Best" is
        comparable across datasets)
    iters : numpy.array
        (n_snapshots,) epoch index of each snapshot
    n_classes : int
        number of output categories (1 for regression)
    """
    with open(path, "rb") as f:
        run = pickle.load(f)
    trace = run["trace"]
    # Keep the native float16 the trainers store: the hamming/quantize locations
    # only argmax/bin these values (identical on float16) and the PCA locations
    # upcast locally, so materialising float32 here just doubles the (large)
    # descriptor matrix and its concatenate copy for no gain.
    logits = np.asarray(trace["logits"], dtype=np.float16)  # (T, n_train, C)
    if stride > 1:
        logits = logits[::stride]
    n_classes = logits.shape[2]
    descriptors = logits.reshape(logits.shape[0], -1)  # (T, n_train*C)
    if run.get("task") == "regression":
        fitness = np.asarray(trace["r2"], dtype=np.float64)  # R^2, maximised
    else:
        fitness = np.asarray(trace["acc"], dtype=np.float64)  # accuracy, maximised
    iters = np.asarray(trace["epoch_idx"], dtype=np.int64)
    if stride > 1:
        fitness = fitness[::stride]
        iters = iters[::stride]
    return descriptors, fitness, iters, n_classes


def load_condition(
    paths: list[Path], stride: int = 1
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """
    Stack every run of a condition into one descriptor matrix + long table.

    Parameters
    ----------
    paths : list of pathlib.Path
        one file per run/seed
    stride : int
        forwarded to :func:`_load_run`

    Returns
    -------
    descriptors : numpy.array
        (n_rows, n_train * n_classes) flattened logits, row-aligned to ``df``
    df : pandas.DataFrame
        long trace with columns ``Run`` (1..nruns), ``Iter``, ``Fitness``
    n_classes : int
        shared by all runs; needed to un-flatten the logits
    """
    desc_blocks: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    n_classes = 0
    for run_id, path in enumerate(paths, start=1):
        descriptors, fitness, iters, n_classes = _load_run(path, stride=stride)
        desc_blocks.append(descriptors)
        frames.append(
            pd.DataFrame(
                {
                    "Run": run_id,
                    "Iter": iters,
                    "Fitness": fitness,
                }
            )
        )
    descriptors = np.concatenate(desc_blocks, axis=0)
    df = pd.concat(frames, ignore_index=True)
    return descriptors, df, n_classes


def save_edgelist(
    stn: "STN", path: str | Path, *, weight: bool = True
) -> None:
    """
    Write the STN graph as an edge-list CSV (one directed ``From,To`` per line).

    Parameters
    ----------
    stn : STN
    path : str or pathlib.Path
        destination .csv (parent dirs created)
    weight : bool
        include a third ``Count`` column (edge multiplicity)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    g = stn.graph
    names = g.vs["name"]
    rows = [(names[s], names[t]) for s, t in g.get_edgelist()]
    edges = pd.DataFrame(rows, columns=["From", "To"])
    if weight and "Count" in g.es.attributes():
        edges["Count"] = list(g.es["Count"])
    edges.to_csv(path, index=False)


def save_membership(located: pd.DataFrame, path: str | Path) -> None:
    """
    Write the per-step node-assignment table (which snapshot landed in which node).

    Parameters
    ----------
    located : pandas.DataFrame
        long trajectory table after a location added its ``name`` column;
        written columns are ``Run``, ``Iter``, ``Fitness``, ``name``
    path : str or pathlib.Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    located[["Run", "Iter", "Fitness", "name"]].to_csv(path, index=False)
