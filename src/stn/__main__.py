"""
CLI orchestrator: build STNs, metrics and plots over every multirun condition.

Usage
-----
    python -m src.stn --results results --out results/stn
    python -m src.stn --only madelon:real --location hamming
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .build import build_stn
from .io import discover_conditions, load_condition, save_edgelist, save_membership
from .location import hamming_label, regression_hamming
from .metrics import metrics_table
from .plot import save_grid, save_plot

LOCATIONS = ("hamming", "quantize")


def _apply_location(
    name: str, descriptors: np.ndarray, df: pd.DataFrame, args, n_classes: int
) -> tuple[pd.DataFrame, str]:
    """
    Locate every snapshot: return ``df`` with a ``name`` column + fitness agg.

    Parameters
    ----------
    name : str
        location model ("hamming" / "quantize")
    descriptors : numpy.array
    df : pandas.DataFrame
    args : argparse.Namespace
    n_classes : int

    Returns
    -------
    df : pandas.DataFrame
        copy with a ``name`` column added
    fitness_agg : str
    """
    df = df.copy()
    if name == "hamming":
        df["name"] = hamming_label(
            descriptors, n_classes, threshold=args.hamming_threshold
        )
        return df, "max"
    if name == "quantize":
        df["name"] = regression_hamming(
            descriptors, n_bins=args.quant_bins, threshold=args.quant_threshold
        )
        return df, "max"
    raise ValueError(f"unknown location model: {name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build STNs from NN training multiruns.")
    p.add_argument("--results", default="results", help="folder of *.pkl multiruns")
    p.add_argument("--out", default="results/stn", help="output root")
    p.add_argument("--location", choices=(*LOCATIONS, "both"), default="both")
    p.add_argument(
        "--only", default=None, help="restrict to dataset:label_type, e.g. madelon:real"
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        dest="max_runs",
        help="include only the first MAX_RUNS runs (lowest seeds) of each "
        "condition when building the STN; default uses every available run.",
    )
    p.add_argument(
        "--hamming-threshold",
        type=float,
        default=0.1,
        dest="hamming_threshold",
        help="hamming location: max normalised Hamming distance (fraction of "
        "samples allowed to disagree, 0-1) for two snapshots to share a node",
    )
    p.add_argument(
        "--quant-bins",
        type=int,
        default=10,
        dest="quant_bins",
        help="quantize location (regression): number of global quantile value "
        "categories the predictions are binned into (10 = deciles).",
    )
    p.add_argument(
        "--quant-threshold",
        type=float,
        default=0.2,
        dest="quant_threshold",
        help="quantize location (regression): max normalised Hamming distance over "
        "the quantized prediction vectors for two snapshots to share a node (0-1).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=5,
        help="keep every STRIDE-th snapshot of each run's trajectory before "
        "building (post-hoc 'larger log_every'). Coarsens the path uniformly to "
        "shrink the distinct-phenotype count the hamming/quantize locations "
        "cluster; e.g. --stride 5 turns a 2500-snapshot regression run into 500.",
    )
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    locations = LOCATIONS if args.location == "both" else (args.location,)

    conditions = discover_conditions(args.results)
    if args.only:
        ds, lab = args.only.split(":")
        conditions = {k: v for k, v in conditions.items() if k == (ds, lab)}
    if not conditions:
        raise SystemExit("No matching conditions found.")

    stns_by_loc: dict[str, list] = {loc: [] for loc in locations}

    for (dataset, label_type), paths in conditions.items():
        if args.max_runs is not None:
            paths = paths[: args.max_runs]
        print(f"[{dataset} / {label_type}] {len(paths)} runs", flush=True)
        descriptors, df, n_classes = load_condition(paths, stride=args.stride)
        for loc in locations:
            located, agg = _apply_location(loc, descriptors, df, args, n_classes)
            stn = build_stn(located, (dataset, label_type), loc, fitness_agg=agg)
            print(
                f"    {loc:9s} nodes={stn.graph.vcount():6d} edges={stn.graph.ecount():6d}",
                flush=True,
            )
            stn.save(out / loc / f"{dataset}_{label_type}.pkl")
            save_edgelist(stn, out / loc / f"{dataset}_{label_type}.csv")
            save_membership(
                located, out / loc / f"{dataset}_{label_type}_membership.csv"
            )
            stns_by_loc[loc].append(stn)
            if not args.no_plots:
                for layout in ("kk", "fitness"):
                    save_plot(
                        stn,
                        out / "plots" / loc / f"{dataset}_{label_type}_{layout}.png",
                        layout=layout,
                    )

    # Metrics CSV + comparison grid per location model.
    for loc, stns in stns_by_loc.items():
        if not stns:
            continue
        table = metrics_table(stns)
        csv_path = out / "metrics" / f"{loc}_{args.only}_stn-metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
        if not args.no_plots and len(stns) > 1:
            save_grid(stns, out / "plots" / f"{loc}_grid_kk.png", layout="kk", ncols=2)

    print("done.")



if __name__ == "__main__":
    main()
