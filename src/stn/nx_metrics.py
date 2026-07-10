"""Classic NetworkX topological measures for STNs + a paired real-vs-random study.

Converts each STN (igraph) to networkx and computes standard graph-level scalars
plus mean/max aggregates of the node centralities. ``paired_study`` contrasts
each measure across the generalizing (real) vs memorizing (random) regime per
dataset. Every measure is wrapped so a failed computation yields ``NaN`` rather
than aborting the whole row.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .build import STN


def stn_to_nx(stn: STN) -> nx.DiGraph:
    """
    Convert an STN's igraph graph to a weighted ``networkx.DiGraph``.

    Node ``name`` becomes the node id; ``Fitness``, ``Count`` and role ride along
    as node attributes; edge ``Count`` becomes the ``weight`` attribute.

    Parameters
    ----------
    stn : STN

    Returns
    -------
    G : networkx.DiGraph
    """
    g = stn.graph
    names = g.vs["name"]
    G = nx.DiGraph()
    for v in g.vs:
        G.add_node(
            v["name"],
            Fitness=float(v["Fitness"]),
            Count=int(v["Count"]),
            role=v["Node"],
        )
    for e in g.es:
        G.add_edge(names[e.source], names[e.target], weight=int(e["Count"]))
    return G


def _try(fn, default=np.nan):
    """Evaluate ``fn`` returning ``default`` (NaN) on any error or warning-as-error."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn()
    except Exception:
        return default


def _agg(out: dict, name: str, d: dict) -> None:
    """Store mean and max of a node-keyed centrality dict under ``name_{mean,max}``."""
    if not d:
        out[f"{name}_mean"] = np.nan
        out[f"{name}_max"] = np.nan
        return
    v = np.fromiter(d.values(), dtype=float)
    out[f"{name}_mean"] = float(v.mean())
    out[f"{name}_max"] = float(v.max())


def nx_measures(stn: STN) -> dict:
    """Compute the classic NetworkX topological measures for one STN."""
    G = stn_to_nx(stn)
    U = G.to_undirected()
    n, m = G.number_of_nodes(), G.number_of_edges()

    # Largest undirected connected component: distance metrics need a connected graph.
    if n:
        Ulcc = U.subgraph(max(nx.connected_components(U), key=len)).copy()
    else:
        Ulcc = U

    out: dict = {"dataset": stn.condition[0], "label_type": stn.condition[1]}

    # --- size & degree ----------------------------------------------------------
    out["n_nodes"] = n
    out["n_edges"] = m
    out["density"] = _try(lambda: nx.density(G))
    degs = np.array([d for _, d in G.degree()], dtype=float)
    out["mean_degree"] = float(degs.mean()) if degs.size else np.nan
    out["max_degree"] = float(degs.max()) if degs.size else np.nan
    out["std_degree"] = float(degs.std()) if degs.size else np.nan
    out["mean_in_degree"] = _try(lambda: float(np.mean([d for _, d in G.in_degree()])))
    out["mean_out_degree"] = _try(lambda: float(np.mean([d for _, d in G.out_degree()])))

    # --- components & connectivity ---------------------------------------------
    out["n_weakly_conn"] = _try(lambda: nx.number_weakly_connected_components(G))
    out["n_strongly_conn"] = _try(lambda: nx.number_strongly_connected_components(G))
    out["largest_wcc_frac"] = _try(
        lambda: len(max(nx.weakly_connected_components(G), key=len)) / n
    )
    out["largest_scc_frac"] = _try(
        lambda: len(max(nx.strongly_connected_components(G), key=len)) / n
    )
    out["reciprocity"] = _try(lambda: nx.reciprocity(G))
    out["node_connectivity_lcc"] = _try(lambda: nx.node_connectivity(Ulcc))
    out["edge_connectivity_lcc"] = _try(lambda: nx.edge_connectivity(Ulcc))
    out["n_isolates"] = _try(lambda: nx.number_of_isolates(G))

    # --- distance (on the largest undirected component) -------------------------
    out["avg_shortest_path_lcc"] = _try(lambda: nx.average_shortest_path_length(Ulcc))
    out["diameter_lcc"] = _try(lambda: nx.diameter(Ulcc))
    out["radius_lcc"] = _try(lambda: nx.radius(Ulcc))
    out["wiener_index_lcc"] = _try(lambda: nx.wiener_index(Ulcc))
    out["global_efficiency"] = _try(lambda: nx.global_efficiency(U))
    out["local_efficiency"] = _try(lambda: nx.local_efficiency(U))

    # --- clustering & triangles (undirected) -----------------------------------
    out["transitivity"] = _try(lambda: nx.transitivity(U))
    out["avg_clustering"] = _try(lambda: nx.average_clustering(U))
    out["n_triangles"] = _try(lambda: sum(nx.triangles(U).values()) // 3)
    out["max_core_number"] = _try(lambda: max(nx.core_number(U).values()))

    # --- mixing / global shape -------------------------------------------------
    out["degree_assortativity"] = _try(lambda: nx.degree_assortativity_coefficient(G))
    out["is_dag"] = _try(lambda: int(nx.is_directed_acyclic_graph(G)), default=np.nan)
    out["flow_hierarchy"] = _try(lambda: nx.flow_hierarchy(G))
    out["s_metric"] = _try(lambda: nx.s_metric(G))

    # --- node centralities, aggregated to mean & max ---------------------------
    _agg(out, "deg_centrality", _try(lambda: nx.degree_centrality(G), default={}))
    _agg(out, "in_deg_centrality", _try(lambda: nx.in_degree_centrality(G), default={}))
    _agg(out, "out_deg_centrality", _try(lambda: nx.out_degree_centrality(G), default={}))
    _agg(out, "betweenness", _try(lambda: nx.betweenness_centrality(G, weight=None), default={}))
    _agg(out, "closeness", _try(lambda: nx.closeness_centrality(G), default={}))
    _agg(out, "pagerank", _try(lambda: nx.pagerank(G), default={}))
    # eigenvector centrality is ill-defined on a directed graph with sinks; use the
    # undirected graph (the standard, well-posed choice).
    _agg(out, "eigenvector", _try(lambda: nx.eigenvector_centrality_numpy(U), default={}))
    _agg(out, "harmonic", _try(lambda: nx.harmonic_centrality(G), default={}))
    _agg(out, "clustering", _try(lambda: nx.clustering(U), default={}))
    _agg(out, "avg_neighbor_degree", _try(lambda: nx.average_neighbor_degree(G), default={}))
    return out


def nx_measures_table(stns: list[STN]) -> pd.DataFrame:
    """Stack :func:`nx_measures` rows for several STNs."""
    return pd.DataFrame([nx_measures(s) for s in stns])


def paired_study(
    table: pd.DataFrame,
    id_col: str = "dataset",
    label_col: str = "label_type",
    pos: str = "real",
    neg: str = "random",
) -> pd.DataFrame:
    """
    Paired per-dataset contrast of every measure across the real/random regime.

    For each metric and dataset with both regimes, ``Delta = value(pos) -
    value(neg)``. Rows are sorted by ``sign_consistency`` then ``|effect_dz|``.

    Parameters
    ----------
    table : pandas.DataFrame
        one row per STN (from :func:`nx_measures_table`)
    id_col : str
    label_col : str
    pos, neg : str
        the two regime labels to contrast

    Returns
    -------
    pandas.DataFrame
        one row per metric: "n_pairs", "mean_diff", "direction",
        "sign_consistency" (fraction of datasets agreeing on the sign),
        "effect_dz" (``mean(Delta)/std(Delta)``, Cohen's d_z; NaN if < 2 pairs)
    """
    metrics = [c for c in table.columns if c not in (id_col, label_col)]
    real = table[table[label_col] == pos].set_index(id_col)
    rand = table[table[label_col] == neg].set_index(id_col)
    common = real.index.intersection(rand.index)

    rows = []
    for mcol in metrics:
        diff = (real.loc[common, mcol] - rand.loc[common, mcol]).dropna()
        if diff.empty:
            continue
        npos, nneg, npair = int((diff > 0).sum()), int((diff < 0).sum()), len(diff)
        sd = diff.std(ddof=1)
        rows.append(
            {
                "metric": mcol,
                "n_pairs": npair,
                "mean_diff": diff.mean(),
                "direction": "real>random" if npos >= nneg else "real<random",
                "sign_consistency": max(npos, nneg) / npair,
                "effect_dz": diff.mean() / sd if sd and sd > 0 else np.nan,
            }
        )
    res = pd.DataFrame(rows)
    res["abs_dz"] = res["effect_dz"].abs()
    res = res.sort_values(
        ["sign_consistency", "abs_dz"], ascending=False
    ).drop(columns="abs_dz")
    return res.reset_index(drop=True)


def effect_correlation(
    table: pd.DataFrame,
    method: str = "pearson",
    label_col: str = "label_type",
    pos: str = "real",
) -> pd.DataFrame:
    """
    Correlation matrix among the measures and the generalizing/memorizing effect.

    A binary ``generalizing`` target (1 for ``pos``, else 0) is appended and the
    full correlation matrix over all numeric measures + target is returned. Pools
    all STNs and ignores dataset pairing (read alongside :func:`paired_study`).

    Parameters
    ----------
    table : pandas.DataFrame
    method : str
        correlation method for :meth:`pandas.DataFrame.corr`
    label_col : str
    pos : str
        label counted as generalizing

    Returns
    -------
    pandas.DataFrame
        correlation matrix; the ``generalizing`` row is each measure's
        (point-biserial) correlation with the effect
    """
    feat = table.copy()
    feat["generalizing"] = (feat[label_col] == pos).astype(int)
    num = feat.select_dtypes("number")
    return num.corr(method=method)


def effect_correlation_ranked(table: pd.DataFrame, **kw) -> pd.DataFrame:
    """Each measure's correlation with the effect, sorted by absolute strength."""
    corr = effect_correlation(table, **kw)
    s = corr["generalizing"].drop("generalizing")
    return (
        s.rename("corr_with_generalizing")
        .to_frame()
        .assign(abs_corr=lambda d: d["corr_with_generalizing"].abs())
        .sort_values("abs_corr", ascending=False)
        .drop(columns="abs_corr")
        .reset_index(names="metric")
    )


def main() -> None:
    """CLI: load STNs from a folder and write/print the topological-measures table.

    Examples
    --------
        uv run stn-measures
        uv run stn-measures --stn-dir results/stn/hamming --paired
        uv run python -m src.stn.nx_metrics --out my_measures.csv
    """
    p = argparse.ArgumentParser(
        description="Classic NetworkX topological measures per STN."
    )
    p.add_argument(
        "--stn-dir",
        default="results/stn/hamming",
        help="folder of STN *.pkl files (default: results/stn/hamming)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: <stn-dir>/../metrics/<loc>_nx-measures.csv)",
    )
    p.add_argument(
        "--paired",
        action="store_true",
        help="also compute the paired real-vs-random study (written alongside)",
    )
    args = p.parse_args()

    stn_dir = Path(args.stn_dir)
    paths = sorted(stn_dir.glob("*.pkl"))
    if not paths:
        raise SystemExit(f"No STN .pkl files found in {stn_dir}")

    stns = [STN.load(pth) for pth in paths]
    table = nx_measures_table(stns)

    out = (
        Path(args.out)
        if args.out
        else stn_dir.parent / "metrics" / f"{stn_dir.name}_nx-measures.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))
    print(f"\nwrote {out}  ({table.shape[0]} STNs x {table.shape[1] - 2} measures)")

    # Correlation matrix between measures and the generalizing/memorizing effect.
    corr = effect_correlation(table)
    corr_out = out.with_name(out.stem + "_corr.csv")
    corr.to_csv(corr_out)
    ranked = effect_correlation_ranked(table)
    print("\n=== correlation with generalizing effect (real=1, random=0), ranked ===")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(ranked.to_string(index=False))
    print(f"\nwrote full correlation matrix to {corr_out}")

    if args.paired:
        study = paired_study(table)
        study_out = out.with_name(out.stem + "_paired.csv")
        study.to_csv(study_out, index=False)
        print("\n=== paired study (real - random), ranked ===")
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(study.to_string(index=False))
        print(f"\nwrote {study_out}")


if __name__ == "__main__":
    main()
