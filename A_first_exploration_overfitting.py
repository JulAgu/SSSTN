import os
import argparse
import pickle
import numpy as np
import torch
from engine.trainer import train_mlp
from utilities.data import (
    add_datasets_arg,
    load_xy,
    prepare_classification,
    select_tasks,
)
from tqdm import tqdm

np.random.seed(42)
FOLD = 0
N_LAYERS = 1
HIDDEN = 512
EPOCHS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overfitting exploration on continuous-only CC18 datasets."
    )
    add_datasets_arg(parser)
    args = parser.parse_args()

    selected = select_tasks("classification", args.datasets)

    # Download all continuous-only CC18 datasets and store as (X, y) pairs
    datasets = {}

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Downloading"):
        X, y = load_xy(row["did"])
        datasets[row["name"]] = {
            "X": X,
            "y": y,
            "did": int(row["did"]),
            "tid": int(row["tid"]),
        }

    print(f"Loaded {len(datasets)} datasets.")

    out_dir = f"./results"
    os.makedirs(out_dir, exist_ok=True)

    n_done = 0
    for name, entry in datasets.items():
        out_path = os.path.join(
            out_dir,
            f"fold{FOLD}_layers{N_LAYERS}_hidden{HIDDEN}_epc{EPOCHS}_{name}.pkl",
        )
        if os.path.exists(out_path):
            print(f"Skipping {name} (already done)")
            n_done += 1
            continue

        try:
            X_tr, y_tr, X_te, y_te, le, n_classes = prepare_classification(
                entry["X"], entry["y"], entry["tid"], fold=FOLD
            )
            # permute y_tr to break any correlation with X_tr and test for overfitting
            perm = np.random.permutation(len(y_tr))
            y_tr = y_tr[perm]

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
                Xtr = torch.tensor(
                    scaler.transform(X_tr), dtype=torch.float32, device=DEVICE
                )
                preds_tr = model(Xtr).argmax(1).cpu().numpy()
                Xte = torch.tensor(
                    scaler.transform(X_te), dtype=torch.float32, device=DEVICE
                )
                preds_te = model(Xte).argmax(1).cpu().numpy()
            final_train_acc = (preds_tr == y_tr).mean()
            final_test_acc = (preds_te == y_te).mean()

            payload = {
                "name": name,
                "did": entry["did"],
                "tid": entry["tid"],
                "fold": FOLD,
                "n_classes": n_classes,
                "in_dim": X_tr.shape[1],
                "n_train": X_tr.shape[0],
                "perm": perm,
                "y_tr": y_tr,
                "final_train_acc": float(final_train_acc),
                "final_test_acc": float(final_test_acc),
                "scaler": scaler,
                "le": le,
                "model": model.cpu().state_dict(),
                "trace": trace,
            }

            with open(out_path, "wb") as f:
                pickle.dump(payload, f)
            n_done += 1
        except Exception as e:
            print(f"ERROR on {name}: {e}")
            continue

    print(f"Done — {n_done} experiments saved to {out_dir}/")
