"""Render STNs to PNG.

Drawn with matplotlib from an igraph layout. Node role sets colour/shape, node
size encodes visit ``Count`` and edge width encodes edge ``Count``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from .build import STN

# Role -> (colour, marker), ordered Start, Medium, End, Best (plot.R NSHAPES/NCOLORS).
ROLE_STYLE = {
    "Start": ("#4daf4a", "s"),
    "Medium": ((0.2, 0.2, 0.2, 0.5), "o"),
    "End": ("#377eb8", "^"),
    "Best": ("#e41a1c", "o"),
}
# Edge colour by fitness-delta Type (build.py); makes trajectory direction readable.
EDGE_TYPE_COLOR = {
    "Improving": "#1a9850",  # moves to better fitness
    "Worsening": "#d73027",  # moves to worse fitness
    "Equal": "0.45",         # neutral move
}
NSIZE_RANGE = (15, 220)  # marker area range (pt^2)
EWIDTH_RANGE = (0.3, 2.0)


def _scale(values, lo, hi):
    values = np.asarray(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.full_like(values, (lo + hi) / 2)
    return lo + (values - vmin) / (vmax - vmin) * (hi - lo)


def _layout(stn: STN, layout: str):
    g = stn.graph
    coords = np.asarray(g.layout_kamada_kawai().coords, dtype=float)
    if layout == "fitness":
        coords[:, 1] = np.asarray(
            g.vs["Fitness"], dtype=float
        )  # higher accuracy higher
    return coords


def plot_stn(stn: STN, ax: plt.Axes, layout: str = "kk") -> None:
    """Draw one STN onto a matplotlib axis."""
    g = stn.graph
    coords = _layout(stn, layout)
    sizes = _scale(g.vs["Count"], *NSIZE_RANGE)
    ewidths = _scale(g.es["Count"], *EWIDTH_RANGE) if g.ecount() else []
    # Marker radius (pt) per node, so arrowheads land on the node boundary.
    radii = np.sqrt(sizes / np.pi)
    types = g.es["Type"] if "Type" in g.es.attributes() else None

    # Edges first (under nodes). Real arrowheads encode trajectory direction;
    # a slight curve separates the two arrows of a 2-cycle so both stay visible.
    for i, (e, w) in enumerate(zip(g.es, ewidths)):
        color = EDGE_TYPE_COLOR.get(types[i], "0.35") if types is not None else "0.35"
        arrow = FancyArrowPatch(
            (coords[e.source, 0], coords[e.source, 1]),
            (coords[e.target, 0], coords[e.target, 1]),
            arrowstyle="-|>",
            mutation_scale=8 + 4 * w,
            connectionstyle="arc3,rad=0.05",
            linewidth=w,
            color=color,
            alpha=0.6,
            shrinkA=radii[e.source] + 1.0,
            shrinkB=radii[e.target] + 1.5,
            zorder=1,
        )
        ax.add_patch(arrow)

    roles = np.asarray(g.vs["Node"])
    for role, (color, marker) in ROLE_STYLE.items():
        idx = np.where(roles == role)[0]
        if idx.size == 0:
            continue
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=sizes[idx],
            c=[color],
            marker=marker,
            edgecolors="none" if role == "Medium" else "black",
            linewidths=0.3,
            zorder=2,
        )

    ax.set_title(f"{stn.condition[0]} {stn.condition[1]} ({stn.location})", fontsize=11)
    if layout == "fitness":
        ax.set_ylabel("Accuracy (better = higher)")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)


def _legend(fig):
    node_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="w",
            markerfacecolor=(c if isinstance(c, str) else c),
            markeredgecolor="black",
            markersize=9,
            label=role,
        )
        for role, (c, m) in ROLE_STYLE.items()
    ]
    edge_handles = [
        Line2D([0], [0], color=c, lw=2, label=etype)
        for etype, c in EDGE_TYPE_COLOR.items()
    ]
    leg_nodes = fig.legend(handles=node_handles, loc="upper right", title="Node")
    fig.add_artist(leg_nodes)
    fig.legend(handles=edge_handles, loc="lower right", title="Move")


def save_plot(
    stn: STN, out_path: str | Path, layout: str = "kk", legend: bool = False
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_stn(stn, ax, layout=layout)
    # Legend off by default so figures drop straight into multi-panel layouts
    # where the legend is supplied once for the whole panel.
    if legend:
        _legend(fig)
        fig.tight_layout(rect=(0, 0, 0.85, 1))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_grid(
    stns: list[STN],
    out_path: str | Path,
    layout: str = "kk",
    ncols: int = 2,
    legend: bool = False,
) -> None:
    """Grid of STNs (e.g. real vs random side by side), echoing combined_plot_* in plot.R."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(stns)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    for ax, stn in zip(axes.ravel(), stns):
        plot_stn(stn, ax, layout=layout)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if legend:
        _legend(fig)
        fig.tight_layout(rect=(0, 0, 0.9, 1))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
