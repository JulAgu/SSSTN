"""Pool the raw per-STN measures for stn_30 across the three algorithm families
(MLP / XGB / SR), on the **real**-label STNs only.

This is the model-contrast sibling of
``notebooks/selecting_metrics_from_randomization.ipynb`` (which pools real vs
random within a single algorithm).  Here the rows are still 7 datasets — 4
classification (``hamming`` location) + 3 regression (``quantize`` location) —
but the separating column is ``model`` instead of ``memorizing``:

    results      -> MLP   (hamming + quantize)
    results_xgb  -> XGB   (hamming + quantize)
    results_sr   -> SR    (quantize only)

For each (base, model) we load the real STN ``.pkl`` files, compute the
structural (nx) and label-aware measure tables, merge them on
``[dataset, label_type]``, and concatenate.  The result is written to
``results/stn_30/metrics/pooled_all-measures_by-model.csv``.

Run from the repo root or from notebooks/:

    uv run python notebooks/pooled_measures_by_model.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

# Resolve to the repo root so `src` imports and `results/` paths work whether the
# script is launched from the repo root or from notebooks/.
ROOT = Path(__file__).resolve().parent
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.stn.build import STN  # noqa: E402
from src.stn.nx_metrics import nx_measures_table  # noqa: E402
from src.stn.label_metrics import label_measures_table  # noqa: E402

KEYS = ["dataset", "label_type"]
LOCS = ("hamming", "quantize")  # classification + regression location families
BASES = {                       # results directory -> algorithm label
    "results": "MLP",
    "results_xgb": "XGB",
    "results_sr": "SR",
}
OUT = ROOT / "results" / "stn_30" / "metrics" / "pooled_all-measures_by-model.csv"


def real_stns(base: str, loc: str) -> list[STN]:
    """Load every real-label STN for one (base, location) family ([] if absent)."""
    d = ROOT / base / "stn_30" / loc
    if not d.is_dir():
        return []
    return [STN.load(p) for p in sorted(d.glob("*_real.pkl"))]


def pool_model(base: str, model: str) -> pd.DataFrame | None:
    """Structural + label-aware measures for one algorithm, hamming + quantize."""
    stns = [s for loc in LOCS for s in real_stns(base, loc)]
    if not stns:
        return None
    combined = nx_measures_table(stns).merge(
        label_measures_table(stns), on=KEYS, how="inner"
    )
    combined.insert(2, "model", model)  # contrast column (replaces `memorizing`)
    return combined


def main() -> None:
    parts = []
    for base, model in BASES.items():
        part = pool_model(base, model)
        if part is None:
            print(f"{model:4s}: no real STNs under {base}/stn_30 — skipped")
            continue
        parts.append(part)
        print(f"{model:4s}: {part.shape[0]} STNs x {part.shape[1] - 3} measures")

    pooled = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}  ({pooled.shape[0]} rows x {pooled.shape[1]} cols)")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(
            pooled[KEYS + ["model", "n_nodes", "n_edges", "best_fitness"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()