"""
Pavé of STNs across the number of runs (columns = increasing run count).

    python -m scripts_for_figures.runs_sweep

Reads one STN per cell from ``<base>/stn_nor_<N>/<location>/<stem>_real.pkl``;
missing cells render as a "missing" placeholder.
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
OUT_PATH = FIG_DIR / f"runs_sweep_{LAYOUT}_2_3.pdf"

# Run-count columns (also the stn_nor_<N> suffix), increasing left -> right.
RUN_COUNTS = ["10", "20", "30"]


def stn_root(base: str, n_runs: str) -> Path:
    return Path(base) / f"stn_nor_{n_runs}"


# (row label, results base, location subfolder, dataset stem) per task row.
TASKS: list[tuple[str, str, str, str]] = [
    ("CIFAR 10\nMLP", "results", "hamming", "CIFAR_10"),
    ("cpu activity\nXGBoost", "results_xgb", "quantize", "cpu_activity"),
]


def _cell_stats(stn: STN) -> str:
    """Corner-box label: graph size next to one size-invariant descriptor.

    The node and edge counts grow with the number of runs (the network gets
    bigger), while the global efficiency -- mean inverse shortest-path length on
    the undirected graph, ``nx_metrics``' definition -- stays roughly flat: size
    grows, shape does not.
    """
    G = stn_to_nx(stn)
    U = G.to_undirected()
    return "\n".join(
        [
            rf"glob. eff. $= {nx.global_efficiency(U):.3f}$",
        ]
    )


def _shared_legend(fig) -> None:
    """
    One horizontal legend (node roles + move types) under the whole panel.
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
            "pdf.fonttype": 42,  # editable text in the vector PDF
        }
    )

    nrows, ncols = len(TASKS), len(RUN_COUNTS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), squeeze=False)

    for r, (row_label, base, location, stem) in enumerate(TASKS):
        for c, n_runs in enumerate(RUN_COUNTS):
            ax = axes[r][c]
            path = stn_root(base, n_runs) / location / f"{stem}_real.pkl"
            if not path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                stn = STN.load(path)
                plot_stn(stn, ax, layout=LAYOUT)
                # Node/edge counts grow across columns while global efficiency
                # stays flat -> the graph gets bigger but its shape is stable.
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
            ax.set_title("")
            ax.set_ylabel("")  # drop any layout-injected y-label
            for spine in ax.spines.values():
                spine.set_edgecolor("0.6")  # gray subplot border
            if r == 0:
                ax.set_title(rf"${n_runs}$ runs", fontsize=15, pad=10)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=14, labelpad=10)

    _shared_legend(fig)
    # Reserve a bottom band for the legend and a top band for the arrow/caption.
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
