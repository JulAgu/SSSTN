"""Label-aware (attribute-aware) topological measures for STNs.

Unlike the purely structural battery in :mod:`src.stn.nx_metrics`, these use the
node value (``Fitness`` = accuracy), node role (Start/Medium/End/Best) and edge
type (Improving/Worsening/Equal) -- the descriptors that respond to *what* the
search optimises. Families: attribute assortativity, spatial autocorrelation of
accuracy (Moran's I, Geary's C, Dirichlet energy), fitness-distance correlation,
group separation (conductance, modularity) and edge-type mix.

Exposes a CLI that writes the measures table, its correlation matrix with the
generalizing/memorizing effect, and the paired real-vs-random study::

    uv run python -m src.stn.label_metrics --stn-dir results/stn/hamming
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .build import STN
from .metrics import terminal_connectivity
from .nx_metrics import (
    _try,
    effect_correlation,
    effect_correlation_ranked,
    paired_study,
    stn_to_nx,
)


def _morans_i(x: np.ndarray, A: np.ndarray) -> float:
    """Moran's I of node values ``x`` under binary adjacency ``A`` (undirected)."""
    n = x.size
    xc = x - x.mean()
    s0 = A.sum()
    den = (xc**2).sum()
    if s0 <= 0 or den <= 0:
        return np.nan
    return float((n / s0) * (xc @ A @ xc) / den)


def _gearys_c(x: np.ndarray, A: np.ndarray) -> float:
    """Geary's C of node values ``x`` under binary adjacency ``A`` (undirected)."""
    n = x.size
    xc = x - x.mean()
    s0 = A.sum()
    den = (xc**2).sum()
    if s0 <= 0 or den <= 0:
        return np.nan
    D = (x[:, None] - x[None, :]) ** 2
    return float(((n - 1) * (A * D).sum()) / (2 * s0 * den))


def label_measures(stn: STN) -> dict:
    """Label-aware topological measures for one STN."""
    G = stn.graph  # igraph, for edge Type / roles
    Gx = stn_to_nx(stn)  # networkx DiGraph with Fitness / role / weight
    U = Gx.to_undirected()

    nodes = list(Gx.nodes())
    x = np.array([Gx.nodes[v]["Fitness"] for v in nodes], dtype=float)
    A = nx.to_numpy_array(U, nodelist=nodes, weight=None)
    np.fill_diagonal(A, 0.0)
    var = float(x.var())

    # Best and End are the same end-of-optimization class -> collapse to "Terminal".
    role_c = {
        v: ("Terminal" if Gx.nodes[v]["role"] in ("End", "Best") else Gx.nodes[v]["role"])
        for v in nodes
    }
    nx.set_node_attributes(Gx, role_c, "role_c")
    term = {v for v in nodes if role_c[v] == "Terminal"}

    out: dict = {"dataset": stn.condition[0], "label_type": stn.condition[1]}

    # --- attribute assortativity ----------------------------------------------
    out["assort_fitness"] = _try(
        lambda: nx.numeric_assortativity_coefficient(Gx, "Fitness")
    )
    out["assort_role"] = _try(
        lambda: nx.attribute_assortativity_coefficient(Gx, "role_c")
    )

    # --- spatial autocorrelation of accuracy ----------------------------------
    out["moran_I"] = _morans_i(x, A) if var > 0 else np.nan
    out["geary_C"] = _gearys_c(x, A) if var > 0 else np.nan
    # Dirichlet energy = sum over undirected edges of squared accuracy difference.
    de = float(sum((Gx.nodes[u]["Fitness"] - Gx.nodes[v]["Fitness"]) ** 2
                   for u, v in U.edges()))
    out["dirichlet_energy"] = de
    m_u = U.number_of_edges()
    out["dirichlet_energy_norm"] = de / (m_u * var) if m_u and var > 0 else np.nan

    # --- fitness-distance correlation to the nearest terminal (End/Best) node -
    if term:
        # Undirected distance from each node to the nearest Terminal node (BFS per source).
        dist: dict = {}
        for b in term:
            for v, d in nx.single_source_shortest_path_length(U, b).items():
                if v not in dist or d < dist[v]:
                    dist[v] = d
        reach = [(Gx.nodes[v]["Fitness"], dist[v]) for v in nodes if v in dist]
        if len(reach) >= 2:
            fv = np.array([f for f, _ in reach], float)
            dv = np.array([d for _, d in reach], float)
            out["fdc_terminal"] = (
                float(np.corrcoef(fv, dv)[0, 1])
                if fv.std() > 0 and dv.std() > 0
                else np.nan
            )
        else:
            out["fdc_terminal"] = np.nan
    else:
        out["fdc_terminal"] = np.nan

    # --- group separation ------------------------------------------------------
    if 0 < len(term) < len(nodes):
        out["conductance_terminal"] = _try(lambda: nx.conductance(U, term))
    else:
        out["conductance_terminal"] = np.nan
    communities = [
        {v for v in nodes if role_c[v] == r} for r in sorted(set(role_c.values()))
    ]
    out["modularity_role"] = _try(
        lambda: nx.community.modularity(U, communities, weight=None)
    )

    # --- edge-type mix ---------------------------------------------------------
    types = pd.Series(G.es["Type"]) if G.ecount() else pd.Series(dtype=object)
    frac = types.value_counts(normalize=True) if not types.empty else pd.Series(dtype=float)
    out["frac_improving"] = float(frac.get("Improving", 0.0))
    out["frac_worsening"] = float(frac.get("Worsening", 0.0))
    out["frac_equal"] = float(frac.get("Equal", 0.0))

    # --- best-node / terminal summaries ---------------------------------------
    out["num_terminal"] = len(term)
    out["best_fitness"] = float(stn.best)
    tc = terminal_connectivity(stn, normalize=True)
    out["term_mean_dist"] = tc.get("mean_dist", np.nan)
    out["term_mean_dist_norm"] = tc.get("mean_dist_norm", np.nan)
    out["term_frac_reachable"] = tc.get("frac_reachable", np.nan)
    return out


def label_measures_table(stns: list[STN]) -> pd.DataFrame:
    """Stack :func:`label_measures` rows for several STNs."""
    return pd.DataFrame([label_measures(s) for s in stns])


def main() -> None:
    """CLI: write the label-aware measures table, correlation matrix and paired study."""
    p = argparse.ArgumentParser(
        description="Label-aware (attribute) topological measures per STN."
    )
    p.add_argument(
        "--stn-dir",
        default="results/stn/hamming",
        help="folder of STN *.pkl files (default: results/stn/hamming)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="CSV output path (default: <stn-dir>/../metrics/<loc>_label-measures.csv)",
    )
    args = p.parse_args()

    stn_dir = Path(args.stn_dir)
    paths = sorted(stn_dir.glob("*.pkl"))
    if not paths:
        raise SystemExit(f"No STN .pkl files found in {stn_dir}")

    stns = [STN.load(pth) for pth in paths]
    table = label_measures_table(stns)

    out = (
        Path(args.out)
        if args.out
        else stn_dir.parent / "metrics" / f"{stn_dir.name}_label-measures.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))
    print(f"\nwrote {out}  ({table.shape[0]} STNs x {table.shape[1] - 2} measures)")

    # Correlation matrix with the generalizing/memorizing effect.
    corr = effect_correlation(table)
    corr_out = out.with_name(out.stem + "_corr.csv")
    corr.to_csv(corr_out)
    print("\n=== correlation with generalizing effect (real=1, random=0), ranked ===")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(effect_correlation_ranked(table).to_string(index=False))
    print(f"\nwrote full correlation matrix to {corr_out}")

    # Paired real-vs-random study.
    study = paired_study(table)
    study_out = out.with_name(out.stem + "_paired.csv")
    study.to_csv(study_out, index=False)
    print("\n=== paired study (real - random), ranked ===")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(study.to_string(index=False))
    print(f"\nwrote {study_out}")


if __name__ == "__main__":
    main()
