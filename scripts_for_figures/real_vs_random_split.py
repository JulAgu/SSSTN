"""
Pavé of STNs split by task: one classification figure, one regression figure.

- classification (``hamming`` location) -- 4 datasets, a 4x2 pavé
-d regression (``quantize`` location)    -- 3 datasets, a 3x2 pavé

    python -m scripts_for_figures.real_vs_random_split
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.stn.build import STN
from src.stn.plot import EDGE_TYPE_COLOR, ROLE_STYLE, plot_stn

# STN root for the 20-run neural-network multiruns.
STN_ROOT = Path("results/stn_20")
LAYOUT = "fitness"
FIG_DIR = Path("results/figures")

# (dataset stem, location subfolder, pretty row label) per task.
# For now, tau is a parameter that we set manually based on the particle size distribution we want in the STN;
# we transcribe these values manually into the figure.
CLASSIFICATION: list[tuple[str, str, str]] = [
    ("Bioresponse", "hamming", r"Bioresponse ($\tau = 0.1$)"),
    ("CIFAR_10", "hamming", r"CIFAR 10 ($\tau = 0.3$)"),
    ("Fashion-MNIST", "hamming", r"Fashion-MNIST ($\tau = 0.1$)"),
    ("mnist_784", "hamming", r"MNIST ($\tau = 0.1$)"),
]
REGRESSION: list[tuple[str, str, str]] = [
    ("cars", "quantize", r"cars ($\tau = 0.2$)"),
    ("cpu_activity", "quantize", r"cpu activity ($\tau = 0.4$)"),
    ("energy_efficiency", "quantize", r"energy efficiency ($\tau = 0.2$)"),
]
LABEL_TYPES = ("real", "random")
COL_TITLES = {"real": "Real labels", "random": "Shuffled labels"}


def _shared_legend(fig) -> None:
    """One horizontal legend (node roles + move types) under the whole panel."""
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


def make_pave(datasets: list[tuple[str, str, str]], out_path: Path) -> None:
    """
    Render one task's datasets as a rows=datasets x columns=label-type pavé.
    """
    nrows, ncols = len(datasets), len(LABEL_TYPES)
    fig_h = 3.3 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, fig_h),
        squeeze=False,
    )

    for r, (stem, loc, row_label) in enumerate(datasets):
        for c, label_type in enumerate(LABEL_TYPES):
            ax = axes[r][c]
            path = STN_ROOT / loc / f"{stem}_{label_type}.pkl"
            if not path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                stn = STN.load(path)
                plot_stn(stn, ax, layout=LAYOUT)
            ax.set_title("")  # row/column headers carry the labels instead
            for spine in ax.spines.values():
                spine.set_edgecolor("0.6")  # drop the subplot border
            if r == 0:
                ax.set_title(COL_TITLES[label_type], fontsize=14, pad=10)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=14, labelpad=10)

    _shared_legend(fig)
    # rect = (left, bottom, right, top) in figure fractions: keep the grid full
    # width/height and only lift `bottom` to free a ~0.8in strip for the legend
    # (constant inches => larger fraction for the shorter regression panel).
    fig.tight_layout(rect=(0, 0.5 / fig_h, 1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


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

    make_pave(CLASSIFICATION, FIG_DIR / f"real_vs_random_classification_{LAYOUT}_2_4.pdf")
    make_pave(REGRESSION, FIG_DIR / f"real_vs_random_regression_{LAYOUT}_2_3.pdf")


if __name__ == "__main__":
    main()
