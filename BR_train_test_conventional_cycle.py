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
BASE_SEED = 42
N_INITS = 30
FOLD = 0
N_LAYERS = 2
HIDDEN = 512
EPOCHS = 5000
LR = 5e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(f"cuda device = {DEVICE}")
    parser = argparse.ArgumentParser(
        description="Train/test cycle over continuous-only CTR23 regression datasets "
        "across multiple weight initializations (real targets)."
    )
    add_datasets_arg(parser, "regression")
    args = parser.parse_args()

    selected = select_tasks("regression", args.datasets)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    for _, row in selected.iterrows():
        dataset_name = row["name"]
        out_path = os.path.join(
            out_dir,
            f"fold{FOLD}_layers{N_LAYERS}_hidden{HIDDEN}_epc{EPOCHS}_ninits{N_INITS}_",
        )

        try:
            X, y = load_xy(row["did"])
            print(
                f"Loaded {dataset_name}  (did={int(row['did'])}, tid={int(row['tid'])})"
            )

            X_tr, y_tr, X_te, y_te = prepare_regression(X, y, row["tid"], fold=FOLD)

            # Standardize the target so the MSE is in unit-variance units, comparable
            # across datasets. out_dim is always 1.
            y_scaler = StandardScaler()
            y_tr_s = y_scaler.fit_transform(y_tr).astype("float32").reshape(-1)
            y_te_s = y_scaler.transform(y_te).astype("float32").reshape(-1)

            for i in tqdm(range(N_INITS), desc=f"{dataset_name} inits"):
                seed = BASE_SEED + i
                torch.manual_seed(seed)

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
                    Xt = torch.tensor(
                        scaler.transform(X_tr), dtype=torch.float32, device=DEVICE
                    )
                    preds_tr = model(Xt).cpu().numpy().reshape(-1)
                    Xv = torch.tensor(
                        scaler.transform(X_te), dtype=torch.float32, device=DEVICE
                    )
                    preds_te = model(Xv).cpu().numpy().reshape(-1)
                # MSE in standardized-target units (the "MSE accuracy"); lower=better
                final_train_mse = float(np.mean((preds_tr - y_tr_s) ** 2))
                final_test_mse = float(np.mean((preds_te - y_te_s) ** 2))

                print(
                    f"init_{i} | seed={seed} | train_mse={final_train_mse:.4f} "
                    f"| test_mse={final_test_mse:.4f}"
                )

                payload = {
                    "name": dataset_name,
                    "did": int(row["did"]),
                    "tid": int(row["tid"]),
                    "fold": FOLD,
                    "task": "regression",
                    "n_classes": 1,  # single regression output
                    "in_dim": X_tr.shape[1],
                    "n_train": X_tr.shape[0],
                    "n_test": X_te.shape[0],
                    "y_scaler": y_scaler,
                    "n_inits": N_INITS,
                    "seed": seed,
                    "final_test_mse": final_test_mse,
                    "final_train_mse": final_train_mse,
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
