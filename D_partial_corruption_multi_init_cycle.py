import os
import argparse
import pickle
import numpy as np
import torch
from engine.trainer import train_mlp
from utilities.data import add_datasets_arg, load_xy, prepare_classification, select_tasks
from tqdm import tqdm

BASE_SEED = 42
N_INITS = 30
FOLD = 0
N_LAYERS = 3
HIDDEN = 512
EPOCHS = 1000
# Fraction of training labels to corrupt with a uniform random label, in [0, 1].
CORRUPTION = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def corrupt_labels(y, n_classes, corruption, rng):
    """Replace a fraction `corruption` of labels by labels drawn from a
    uniform distribution over the classes (Zhang et al. partially corrupted
    labels). The corruption is applied once and fixed so every initialization
    memorizes the same partially corrupted labels.

    Returns the corrupted labels and the boolean mask of corrupted positions.
    """
    y = y.copy()
    n = len(y)
    n_corrupt = int(round(corruption * n))
    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)
    # Uniform over all classes (a corrupted label may by chance equal the true one,
    # matching the standard partially corrupted labels protocol).
    y[corrupt_idx] = rng.integers(0, n_classes, size=n_corrupt)
    mask = np.zeros(n, dtype=bool)
    mask[corrupt_idx] = True
    return y, mask


if __name__ == "__main__":
    print(f"cuda device = {DEVICE}")
    parser = argparse.ArgumentParser(
        description="Overfitting exploration with partially corrupted labels over "
        "multiple weight initializations on continuous-only CC18 datasets."
    )
    add_datasets_arg(parser)
    parser.add_argument(
        "--corruption",
        type=float,
        default=CORRUPTION * 100,
        help="Percentage of train labels to corrupt, e.g. 20 40 60 80. Default: %(default)g.",
    )
    parser.add_argument(
        "--n-inits",
        type=int,
        default=N_INITS,
        help="Number of weight initializations (runs) per dataset. Default: %(default)d.",
    )
    args = parser.parse_args()

    CORRUPTION = args.corruption / 100.0
    N_INITS = args.n_inits

    if not 0.0 <= CORRUPTION <= 1.0:
        raise ValueError(f"corruption must be in [0, 100], got {args.corruption}")

    # Percentage as an integer-ish string for filenames, e.g. 0.5 -> "50", 0.25 -> "25"
    pct_str = f"{CORRUPTION * 100:g}".replace(".", "p")

    selected = select_tasks("classification", args.datasets)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    for _, row in selected.iterrows():
        dataset_name = row["name"]
        out_path = os.path.join(
            out_dir,
            f"corrupt{pct_str}pct_fold{FOLD}_layers{N_LAYERS}_hidden{HIDDEN}_epc{EPOCHS}_ninits{N_INITS}_",
        )

        try:
            X, y = load_xy(row["did"])
            print(
                f"Loaded {dataset_name}  (did={int(row['did'])}, tid={int(row['tid'])})"
            )

            # Held-out test split keeps its TRUE labels: a corrupted-label net's test
            # accuracy on real labels is the Zhang et al. memorization control.
            X_tr, y_tr, X_te, y_te, le, n_classes = prepare_classification(
                X, y, row["tid"], fold=FOLD
            )

            # Corrupt a fraction of y_tr with uniform random labels to test for
            # overfitting. The corruption is fixed per dataset so every initialization
            # memorizes the same partially corrupted labels. Only the weight init seed varies.
            rng = np.random.default_rng(BASE_SEED)
            y_tr, corrupt_mask = corrupt_labels(y_tr, n_classes, CORRUPTION, rng)
            print(
                f"Corrupted {int(corrupt_mask.sum())}/{len(y_tr)} train labels "
                f"({CORRUPTION:.2%})"
            )

            for i in tqdm(range(N_INITS), desc=f"{dataset_name} inits"):
                seed = BASE_SEED + i
                torch.manual_seed(seed)

                model, scaler, trace = train_mlp(
                    X_tr,
                    y_tr,
                    n_classes,
                    hidden=HIDDEN,
                    n_layers=N_LAYERS,
                    device=DEVICE,
                    epochs=EPOCHS,
                )

                model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(
                        scaler.transform(X_te), dtype=torch.float32, device=DEVICE
                    )
                    preds = model(Xv).argmax(1).cpu().numpy()
                    final_test_acc = float((preds == y_te).mean())

                    Xt = torch.tensor(
                        scaler.transform(X_tr), dtype=torch.float32, device=DEVICE
                    )
                    preds = model(Xt).argmax(1).cpu().numpy()
                    final_train_acc = float((preds == y_tr).mean())

                print(
                    f"init_{i} | seed={seed} | train_acc={final_train_acc:.4f} "
                    f"| test_acc={final_test_acc:.4f}"
                )

                payload = {
                    "name": dataset_name,
                    "did": int(row["did"]),
                    "tid": int(row["tid"]),
                    "fold": FOLD,
                    "n_classes": n_classes,
                    "in_dim": X_tr.shape[1],
                    "n_train": X_tr.shape[0],
                    "n_test": X_te.shape[0],
                    "le": le,
                    "n_inits": N_INITS,
                    "seed": seed,
                    "corruption": CORRUPTION,
                    "corrupt_mask": corrupt_mask,
                    "y_tr": y_tr,
                    "final_train_acc": final_train_acc,
                    "final_test_acc": final_test_acc,
                    "scaler": scaler,
                    "model": model.cpu().state_dict(),
                    "trace": trace,
                }

                with open(f"{out_path}_seed{seed}_{dataset_name}.pkl", "wb") as f:
                    pickle.dump(payload, f)
                print(f"Saved to {out_path}_seed{seed}_{dataset_name}.pkl")

        except Exception as e:
            print(f"Dataset {dataset_name} was impossible to process: {e}")
            continue
