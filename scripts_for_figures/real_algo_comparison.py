"""
Pavé of real-label STNs compared across algorithms (columns = algorithm).

- regression panel    -- 3 datasets x {MLP, XGB, SR}  (3x3, quantize location)
- classification panel -- 4 datasets x {MLP, XGB}     (4x2, hamming location)

SR is regression-only, so it only appears in the regression panel.

    python -m scripts_for_figures.real_algo_comparison

NOTE: each column reads from that algorithm's STN root below. Any STN file that
does not exist yet is drawn as a "missing" placeholder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.stn.build import STN
from src.stn.plot import EDGE_TYPE_COLOR, ROLE_STYLE, plot_stn

LAYOUT = "fitness"
FIG_DIR = Path("results/figures")

# (column label, STN root) per algorithm. MLP keeps the 30-run root; XGB/SR
# roots already represent their full 30-init runs.
MLP = ("MLP", Path("results/stn_30"))
XGB = ("XGBoost", Path("results_xgb/stn_30"))
SR = ("SR", Path("results_sr/stn_30"))

# (dataset stem, pretty row label) per task.
CLASSIFICATION = [
    ("Bioresponse", "Bioresponse"),
    ("CIFAR_10", "CIFAR 10"),
    ("Fashion-MNIST", "Fashion-MNIST"),
    ("mnist_784", "MNIST"),
]
REGRESSION = [
    ("cars", "cars"),
    ("cpu_activity", "cpu activity"),
    ("energy_efficiency", "energy efficiency"),
]

# TODO: fill in the XGB and SR thresholds (None values below are placeholders).
THRESHOLDS: dict[str, dict[str, float | None]] = {
    "MLP": {
        "Bioresponse": 0.1,
        "CIFAR_10": 0.3,
        "Fashion-MNIST": 0.1,
        "mnist_784": 0.1,
        "cars": 0.2,
        "cpu_activity": 0.4,
        "energy_efficiency": 0.2,
    },
    "XGBoost": {
        "Bioresponse": 0.05,
        "CIFAR_10": 0.3,
        "Fashion-MNIST": 0.05,
        "mnist_784": 0.05,
        "cars": 0.15,
        "cpu_activity": 0.2,
        "energy_efficiency": 0.15,
    },
    "SR": {
        "cars": 0.2,
        "cpu_activity": 0.4,
        "energy_efficiency": 0.2,
    },
}


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


def make_pave(
    datasets: list[tuple[str, str]],
    location: str,
    algorithms: list[tuple[str, Path]],
    out_path: Path,
) -> None:
    """Render real-label STNs as rows=datasets x columns=algorithms."""
    nrows, ncols = len(datasets), len(algorithms)
    fig_h = 3.3 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, fig_h),
        squeeze=False,
    )

    for r, (stem, row_label) in enumerate(datasets):
        for c, (algo_name, root) in enumerate(algorithms):
            ax = axes[r][c]
            path = root / location / f"{stem}_real.pkl"
            if not path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                stn = STN.load(path)
                plot_stn(stn, ax, layout=LAYOUT)
                tau = THRESHOLDS.get(algo_name, {}).get(stem)
                if tau is not None:
                    ax.text(
                        0.97,
                        0.04,
                        rf"$\tau = {tau}$",
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
            ax.set_title("")  # row/column headers carry the labels instead
            # Drop plot_stn's "Accuracy (better = higher)" y-label (set by the
            # fitness layout); the left column carries the dataset name instead.
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_edgecolor("0.6")  # gray subplot border
            if r == 0:
                ax.set_title(algo_name, fontsize=15, pad=10)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=14, labelpad=10)

    _shared_legend(fig)
    # Lift only `bottom` to free a constant ~0.8in strip for the legend.
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

    # Regression: 3 datasets x {MLP, XGB, SR}.
    make_pave(
        REGRESSION,
        "quantize",
        [MLP, XGB, SR],
        FIG_DIR / f"real_algo_regression_{LAYOUT}_3_3.pdf",
    )
    # Classification: 4 datasets x {MLP, XGB} (SR is regression-only).
    make_pave(
        CLASSIFICATION,
        "hamming",
        [MLP, XGB],
        FIG_DIR / f"real_algo_classification_{LAYOUT}_4_2.pdf",
    )


if __name__ == "__main__":
    main()
