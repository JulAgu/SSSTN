"""Network metrics per STN -- port of ``metrics.R``.

Computes the 11 columns of the euroGP metrics table from an :class:`~src.stn.build.STN`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .build import STN

ROUND = 2  # rounding factor for numeric metrics (metrics.R `rf`)


def stn_metrics(stn: STN) -> dict:
    """Return the metric row for a single STN (one dict, ordered like metrics.R)."""
    g = stn.graph
    nodes = np.asarray(g.vs["Node"])

    best_ids = np.where(nodes == "Best")[0].tolist()
    end_ids = np.where(nodes == "End")[0].tolist()
    start_ids = np.where(nodes == "Start")[0].tolist()

    ne = g.ecount()
    impr = sum(1 for t in g.es["Type"] if t == "Improving")

    row = {
        "dataset": stn.condition[0],
        "label_type": stn.condition[1],
        "model": stn.location,
        "num_nodes": g.vcount(),
        "num_best": len(best_ids),
        "num_ends": len(end_ids),
        "num_edges": ne,
        "impr_edges": round(impr / ne, ROUND) if ne else np.nan,
        "num_comp": len(g.connected_components(mode="weak")),
    }

    if best_ids:
        # Incoming strength of best nodes, normalised by number of runs.
        best_str = sum(g.strength(best_ids, mode="in", weights="Count"))
        row["strength"] = round(best_str / stn.nruns, ROUND)
        # Shortest path lengths from start nodes to best nodes (unweighted).
        dist = np.asarray(g.distances(source=start_ids, target=best_ids, mode="out"))
        finite = dist[np.isfinite(dist)]
        row["plength"] = round(float(finite.mean()), ROUND) if finite.size else np.nan
        row["npaths"] = int(finite.size)
    else:
        row["strength"] = np.nan
        row["plength"] = np.nan
        row["npaths"] = 0

    return row


def terminal_connectivity(stn: STN, normalize: bool = False) -> dict:
    """
    How connected the end-of-optimization (End/Best) nodes are, undirected.

    Parameters
    ----------
    stn : STN
    normalize : bool
        also report ``mean_dist_norm`` = ``mean_dist`` divided by the graph's
        overall undirected mean path length (removes the graph-size confound;
        < 1 means terminals sit closer than a typical node pair)

    Returns
    -------
    dict
        "dataset", "label_type", "n_terminal", "mean_dist" (mean undirected
        shortest path over reachable terminal pairs), "frac_reachable" (fraction
        of terminal pairs connected), plus "mean_dist_norm" when ``normalize``
    """
    g = stn.graph
    roles = np.asarray(g.vs["Node"])
    term = np.where((roles == "End") | (roles == "Best"))[0].tolist()
    k = len(term)
    out = {
        "dataset": stn.condition[0],
        "label_type": stn.condition[1],
        "n_terminal": k,
    }
    if k < 2:
        out.update(mean_dist=np.nan, frac_reachable=np.nan)
        if normalize:
            out["mean_dist_norm"] = np.nan
        return out

    # mode="all" -> distances ignore edge direction.
    D = np.asarray(g.distances(source=term, target=term, mode="all"), dtype=float)
    d = D[np.triu_indices(k, k=1)]
    finite = d[np.isfinite(d)]
    mean_dist = float(finite.mean()) if finite.size else np.nan
    out["mean_dist"] = round(mean_dist, ROUND)
    out["frac_reachable"] = round(float(np.mean(np.isfinite(d))), ROUND)

    if normalize:
        # overall undirected mean path length (averaged over connected pairs).
        apl = g.average_path_length(directed=False, unconn=True)
        out["mean_dist_norm"] = (
            round(mean_dist / apl, ROUND) if apl and np.isfinite(mean_dist) else np.nan
        )
    return out


def terminal_connectivity_table(stns: list[STN], **kw) -> pd.DataFrame:
    """Stack :func:`terminal_connectivity` rows for several STNs.

    Extra keyword arguments (e.g. ``normalize=True``) are forwarded per STN.
    """
    return pd.DataFrame([terminal_connectivity(s, **kw) for s in stns])


def metrics_table(stns: list[STN]) -> pd.DataFrame:
    """Stack metric rows for several STNs into one DataFrame."""
    return pd.DataFrame([stn_metrics(s) for s in stns])


def write_metrics(stns: list[STN], out_csv: str | Path) -> pd.DataFrame:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table = metrics_table(stns)
    table.to_csv(out_csv, index=False)
    return table
