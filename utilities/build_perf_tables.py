"""Build real-vs-shuffled performance tables for the NN experiments.

For each task (regression / classification) we summarize four columns aggregated
over the weight-init seeds of a single dataset:

    real_train, real_test, shuffled_train, shuffled_test

- Real runs are the plain ``fold0_*`` pickles (true targets).
- Shuffled runs are the ``randlabels_fold0_*`` pickles (permuted train targets,
  Zhang-et-al. randomization test). The test split keeps its TRUE targets.

Metric depends on the task:
- regression  -> R^2.  Targets are standardized (unit variance on train), so
  R^2 = 1 - MSE in standardized units. train R^2 == stored trace["r2"][-1].
- classification -> accuracy (final_train_acc / final_test_acc, stored directly).

We pick, per (dir, label-kind), the n_inits batch with the most pickles among the
requested {17, 20} sizes -- regression real -> ninits17, everything else -> ninits20.
"""

import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

NINITS_CANDIDATES = (20, 17)  # preference order when several batches exist


def dataset_of(path: Path) -> str:
    """seed<N>_<dataset>.pkl -> <dataset>."""
    return re.sub(r".*_seed\d+_", "", path.stem)


def ninits_of(path: Path) -> int | None:
    m = re.search(r"ninits(\d+)", path.stem)
    return int(m.group(1)) if m else None


def pick_ninits(paths: list[Path]) -> int:
    """Among the candidate batch sizes, pick the one with the most pickles."""
    counts = defaultdict(int)
    for p in paths:
        n = ninits_of(p)
        if n in NINITS_CANDIDATES:
            counts[n] += 1
    if not counts:
        raise ValueError("no pickles with ninits in {17, 20}")
    # most files first, then prefer the larger batch on a tie
    return sorted(counts, key=lambda n: (counts[n], n), reverse=True)[0]


def metric(payload: dict, task: str) -> tuple[float, float]:
    """(train, test) performance for one run."""
    if task == "regression":
        return (
            1.0 - payload["final_train_mse"],  # train R^2 (== trace['r2'][-1])
            1.0 - payload["final_test_mse"],   # test  R^2 (normalized MSE)
        )
    return payload["final_train_acc"], payload["final_test_acc"]


def collect(directory: Path, task: str) -> pd.DataFrame:
    real_files = sorted(directory.glob("fold0_*.pkl"))
    shuf_files = sorted(directory.glob("randlabels_fold0_*.pkl"))

    real_n = pick_ninits(real_files)
    shuf_n = pick_ninits(shuf_files)
    real_files = [p for p in real_files if ninits_of(p) == real_n]
    shuf_files = [p for p in shuf_files if ninits_of(p) == shuf_n]

    print(f"\n[{directory.name}] task={task}")
    print(f"  real     : ninits{real_n}  ({len(real_files)} pickles)")
    print(f"  shuffled : ninits{shuf_n}  ({len(shuf_files)} pickles)")

    # dataset -> {"real": [(tr,te),...], "shuffled": [...]}
    acc = defaultdict(lambda: {"real": [], "shuffled": []})
    for kind, files in (("real", real_files), ("shuffled", shuf_files)):
        for p in files:
            with open(p, "rb") as f:
                payload = pickle.load(f)
            acc[dataset_of(p)][kind].append(metric(payload, task))

    rows = []
    for ds in sorted(acc):
        out = {"dataset": ds}
        for kind in ("real", "shuffled"):
            vals = np.array(acc[ds][kind], dtype=float)  # (n_seeds, 2)
            if len(vals) == 0:
                out[f"{kind}_train"] = out[f"{kind}_test"] = np.nan
                out[f"n_{kind}"] = 0
                continue
            tr_m, te_m = vals.mean(0)
            tr_s, te_s = vals.std(0)
            out[f"{kind}_train"] = f"{tr_m:.3f} ± {tr_s:.3f}"
            out[f"{kind}_test"] = f"{te_m:.3f} ± {te_s:.3f}"
            out[f"n_{kind}"] = len(vals)
        rows.append(out)

    cols = [
        "dataset",
        "real_train", "real_test",
        "shuffled_train", "shuffled_test",
        "n_real", "n_shuffled",
    ]
    return pd.DataFrame(rows)[cols]


def main():
    base = Path("results")
    out_dir = base / "summarizing"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("regression_NN", "regression", "R^2"),
        ("classification_NN", "classification", "accuracy"),
    ]

    for sub, task, label in specs:
        df = collect(base / sub, task)
        print(f"\n===== {sub}  (metric: {label}) =====")
        print(df.to_string(index=False))
        csv_path = out_dir / f"perf_table_{sub}.csv"
        df.to_csv(csv_path, index=False)
        print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
