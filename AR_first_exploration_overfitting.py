import os
import argparse
import pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from engine.trainer import train_mlp_regression
from utilities.data import add_datasets_arg, load_xy, prepare_regression, select_tasks
from tqdm import tqdm

np.random.seed(42)
FOLD = 0
N_LAYERS = 2
HIDDEN = 512
EPOCHS = 5000
LR = 5e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overfitting exploration on continuous-only CTR23 regression datasets."
    )
    add_datasets_arg(parser, "regression")
    args = parser.parse_args()

    selected = select_tasks("regression", args.datasets)

    # Download all continuous-only CTR23 datasets and store as (X, y) pairs
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
            X_tr, y_tr, X_te, y_te = prepare_regression(
                entry["X"], entry["y"], entry["tid"], fold=FOLD
            )
            # permute y_tr to break any relation with X_tr and test for overfitting
            perm = np.random.permutation(len(y_tr))
            y_tr = y_tr[perm]

            # z-score normalization over the targets, so the MSE is in unit-variance
            # units, comparable across datasets. out_dim is always 1.
            y_scaler = StandardScaler()
            y_tr_s = y_scaler.fit_transform(y_tr).astype("float32").reshape(-1)
            y_te_s = y_scaler.transform(y_te).astype("float32").reshape(-1)

            model, scaler, trace = train_mlp_regression(
                X_tr,
                y_tr_s,
                hidden=HIDDEN,
                n_layers=N_LAYERS,
                device=DEVICE,
                epochs=EPOCHS,
                lr=LR,
            )

            model.eval()
            with torch.no_grad():
                Xtr = torch.tensor(
                    scaler.transform(X_tr), dtype=torch.float32, device=DEVICE
                )
                preds_tr = model(Xtr).cpu().numpy()
                Xte = torch.tensor(
                    scaler.transform(X_te), dtype=torch.float32, device=DEVICE
                )
                preds_te = model(Xte).cpu().numpy()
            # MSE in standardized-target units (the "MSE accuracy"); lower=better
            final_train_mse = float(np.mean((preds_tr - y_tr_s) ** 2))
            final_test_mse = float(np.mean((preds_te - y_te_s) ** 2))

            payload = {
                "name": name,
                "did": entry["did"],
                "tid": entry["tid"],
                "fold": FOLD,
                "task": "regression",
                "n_classes": 1,  # single regression output
                "in_dim": X_tr.shape[1],
                "n_train": X_tr.shape[0],
                "perm": perm,
                "y_tr": y_tr_s,
                "final_train_mse": final_train_mse,
                "final_test_mse": final_test_mse,
                "scaler": scaler,
                "y_scaler": y_scaler,
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
