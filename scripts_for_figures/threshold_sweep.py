"""
Pavé of STNs across the location threshold tau (columns = increasing tau).

    python -m scripts_for_figures.threshold_sweep

Reads one STN per cell from ``results/stn_hamming_<tau>/<location>/<stem>_real.pkl``;
missing cells render as a "missing" placeholder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from src.stn.build import STN
from src.stn.plot import EDGE_TYPE_COLOR, ROLE_STYLE, plot_stn

LAYOUT = "fitness"
FIG_DIR = Path("results/figures")
OUT_PATH = FIG_DIR / f"threshold_sweep_{LAYOUT}_2_4.pdf"

# Threshold columns (also the STN-root suffix), increasing left -> right.
THRESHOLDS = ["0.1", "0.2", "0.3", "0.4"]


def stn_root(tau: str) -> Path:
    return Path(f"results/stn_hamming_{tau}")


# (row label, location subfolder, dataset stem) per task row.
TASKS: list[tuple[str, str, str]] = [
    ("Bioresponse\nMLP", "hamming", "Bioresponse"),
    ("cars\nMLP", "quantize", "cars"),
]


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

    nrows, ncols = len(TASKS), len(THRESHOLDS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), squeeze=False
    )

    for r, (row_label, location, stem) in enumerate(TASKS):
        for c, tau in enumerate(THRESHOLDS):
            ax = axes[r][c]
            path = stn_root(tau) / location / f"{stem}_real.pkl"
            if not path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                stn = STN.load(path)
                plot_stn(stn, ax, layout=LAYOUT)
                # Node count makes "more nodes at smaller tau" quantitative.
                ax.text(
                    0.97,
                    0.04,
                    rf"{stn.graph.vcount()} nodes",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=12,
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
                ax.set_title(rf"$\tau = {tau}$", fontsize=15, pad=10)
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
