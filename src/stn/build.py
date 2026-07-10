"""
Build the igraph STN from a located trajectory table.

Operates on a long table that already carries a ``name`` column produced by a location function.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd


@dataclass
class STN:
    """
    An STN model plus the metadata stored alongside the graph.
    """

    graph: ig.Graph
    best: float
    condition: tuple[str, str]
    location: str
    nruns: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "STN":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_stn(
    df: pd.DataFrame,
    condition: tuple[str, str],
    location: str,
    fitness_agg: str = "max",
) -> STN:
    """
    Construct an STN from a located trajectory table.

    Parameters
    ----------
    df : DataFrame with columns ``Run``, ``Fitness``, ``name``.  Rows are ordered
        within each run by trajectory step.
    condition : (dataset, label_type)
    location : name of the location model ("hamming" / "quantize").
    fitness_agg : how to aggregate a node's fitness over its visits -- ``"max"``
        (best = highest accuracy) or ``"median"``.  Fitness is training accuracy,
        so it is *maximised*.

    Returns
    -------
    stn : STN
        the built graph plus its metadata (condition, location, nruns, best)
    """
    nruns = int(df["Run"].max())

    # --- Edges per run, de-duplicating revisits keeping first occurrence ------
    # (euroGP applies distinct(Run, descriptor) before edge building.)
    edge_rows: list[tuple[str, str]] = []
    start_nodes: list[str] = []
    end_nodes: list[str] = []
    for run_id, run_df in df.groupby("Run", sort=True):
        # Old: keep="first" -> seq[-1] is the last *newly-discovered* node, which
        # mislabels the End when a run converges back onto an earlier state.
        # seq = run_df.drop_duplicates(subset="name", keep="first")["name"].tolist()
        # keep="last" makes seq[-1] the run's true terminal node.
        seq = run_df.drop_duplicates(subset="name", keep="last")["name"].tolist()
        start_nodes.append(seq[0])
        end_nodes.append(seq[-1])
        edge_rows.extend(zip(seq[:-1], seq[1:]))

    start_nodes = set(start_nodes)
    end_nodes = set(end_nodes)

    # --- Nodes: aggregate over all (deduplicated) visits ----------------------
    dedup = df.groupby(["Run", "name"], sort=False, as_index=False).first()
    nodes = dedup.groupby("name", sort=False).agg({"Fitness": fitness_agg})
    nodes["Count"] = dedup.groupby("name", sort=False).size()
    nodes = nodes.reset_index()

    best = float(nodes["Fitness"].max())
    nodes["Node"] = "Medium"
    nodes.loc[nodes["name"].isin(end_nodes), "Node"] = "End"
    nodes.loc[nodes["name"].isin(start_nodes), "Node"] = "Start"
    nodes.loc[nodes["Fitness"] == best, "Node"] = "Best"

    # --- Edges: aggregate duplicates with a Count weight ----------------------
    edges = (
        pd.DataFrame(edge_rows, columns=["From", "To"])
        .groupby(["From", "To"], sort=False)
        .size()
        .reset_index(name="Count")
    )

    # --- Build graph (vertices first col = name; edges first two = endpoints) -
    vert_df = nodes[["name", "Fitness", "Count", "Node"]]
    # The third edge column ("Count") is attached automatically as an edge attr.
    graph = ig.Graph.DataFrame(edges, directed=True, vertices=vert_df, use_vids=False)
    graph.simplify(multiple=False, loops=True)  # drop self-loops

    # --- Edge Type by endpoint fitness delta (maximisation: accuracy) ---------
    fit = np.asarray(graph.vs["Fitness"], dtype=float)
    types = []
    for e in graph.es:
        f1, f2 = fit[e.source], fit[e.target]
        if f2 > f1:
            types.append("Improving")
        elif f2 == f1:
            types.append("Equal")
        else:
            types.append("Worsening")
    graph.es["Type"] = types

    # Incoming strength (used by the cluster model / plotting), +1 like the R.
    graph.vs["Strength"] = [s + 1 for s in graph.strength(mode="in", weights="Count")]

    return STN(
        graph=graph, best=best, condition=condition, location=location, nruns=nruns
    )
