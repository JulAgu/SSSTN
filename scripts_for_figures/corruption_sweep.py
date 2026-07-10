"""
Pavé of STNs across the label-corruption level (columns = increasing corruption).

Each cell reads one STN.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.stn.build import STN
from src.stn.nx_metrics import stn_to_nx
from src.stn.plot import EDGE_TYPE_COLOR, ROLE_STYLE, plot_stn

LAYOUT = "fitness"
FIG_DIR = Path("results/figures")

CORRUPTIONS = [0, 20, 40, 60, 80, 100]

GRID = (2, 3)

# The single task shown across the sweep:
TASK_LOCATION = "hamming"
TASK_STEM = "mnist_784"


def cell_path(pct: int, location: str, stem: str) -> Path:
    """
    Helper function for building real and fully destroyed information trains
    STN path for one (corruption, task) cell.
    """
    if pct == 0:
        print(Path("results/stn") / location / f"{stem}_real.pkl")
        return Path("results/stn") / location / f"{stem}_real.pkl"
    if pct == 100:
        return Path("results/stn") / location / f"{stem}_random.pkl"
    return Path(f"results/stn_corrupt{pct}") / location / f"{stem}_real.pkl"


def _cell_stats(stn: STN) -> str:
    """
    Corner-box label with topological descriptors.

    - global efficiency -- mean inverse shortest-path length
    - max closeness -- the most central node's closeness centrality
    - max PageRank -- the most-visited node's share of the stationary flow
    """
    G = stn_to_nx(stn)
    U = G.to_undirected()
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)
    return "\n".join(
        [
            rf"Global eff. $= {nx.global_efficiency(U):.3f}$",
            rf"Closeness$_{{\max}} = {max(closeness.values()):.3f}$",
            rf"PageRank$_{{\max}} = {max(pagerank.values()):.3f}$",
        ]
    )


def _shared_legend(fig) -> None:
    """
    One horizontal legend under the whole panel.
    """
    node_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="w",
            markerfacecolor=c,
            markeredgecolor="black",
            markersize=10,
            linestyle="none",
            label=role,
        )
        for role, (c, m) in ROLE_STYLE.items()
    ]
    edge_handles = [
        Line2D([0], [0], color=c, lw=2.5, label=etype)
        for etype, c in EDGE_TYPE_COLOR.items()
    ]
    leg_nodes = fig.legend(
        handles=node_handles,
        loc="lower center",
        bbox_to_anchor=(0.30, 0.0),
        ncols=len(node_handles),
        title="Node role",
        frameon=False,
    )
    fig.add_artist(leg_nodes)
    fig.legend(
        handles=edge_handles,
        loc="lower center",
        bbox_to_anchor=(0.74, 0.0),
        ncols=len(edge_handles),
        title="Move",
        frameon=False,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )

    nrows, ncols = GRID
    fig_h = 3.3 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, fig_h), squeeze=False)
    flat_axes = axes.ravel()

    for idx, ax in enumerate(flat_axes):
        if idx >= len(CORRUPTIONS):
            ax.axis("off")
            continue
        pct = CORRUPTIONS[idx]
        path = cell_path(pct, TASK_LOCATION, TASK_STEM)
        if not path.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            stn = STN.load(path)
            plot_stn(stn, ax, layout=LAYOUT)
            ax.text(
                0.97,
                0.04,
                _cell_stats(stn),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                linespacing=1.4,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="0.7",
                    alpha=0.85,
                ),
            )

        ax.set_ylabel("")  # drop any layout-injected y-label
        for spine in ax.spines.values():
            spine.set_edgecolor("0.6")  # gray subplot border
        ax.set_title(rf"${pct}\%$ of random labels", fontsize=15, pad=10)

    _shared_legend(fig)
    fig.tight_layout(rect=(0, 0.5 / fig_h, 1, 1))

    out_path = FIG_DIR / f"corruption_sweep_{LAYOUT}_{nrows}_{ncols}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
