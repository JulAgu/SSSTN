import os
import argparse
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from engine.sr_trainer import train_sr_regression
from utilities.data import add_datasets_arg, load_xy, prepare_regression, select_tasks
from tqdm import tqdm

np.random.seed(42)
BASE_SEED = 42
N_INITS = 30
FOLD = 0
N_GENERATIONS = 40  # PySR default niterations; one trajectory point per generation
DEVICE = "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Symbolic-regression train/test cycle over continuous-only "
        "CTR23 regression datasets across multiple GA seeds (real targets)."
    )
    add_datasets_arg(parser, "regression")
    args = parser.parse_args()

    selected = select_tasks("regression", args.datasets)

    out_dir = "results_sr"
    os.makedirs(out_dir, exist_ok=True)

    for _, row in selected.iterrows():
        dataset_name = row["name"]
        out_path = os.path.join(
            out_dir,
            f"sr_fold{FOLD}_ngens{N_GENERATIONS}_ninits{N_INITS}_",
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

                model, best_idx, trace = train_sr_regression(
                    X_tr,
                    y_tr_s,
                    n_generations=N_GENERATIONS,
                    seed=seed,
                )

                # Best individual (lowest-loss equation) predictions for the
                # final reported MSE, consistent with the trajectory snapshots.
                preds_tr = np.asarray(
                    model.predict(X_tr, index=best_idx), dtype="float64"
                ).reshape(-1)
                preds_te = np.asarray(
                    model.predict(X_te, index=best_idx), dtype="float64"
                ).reshape(-1)
                # MSE in standardized-target units (the "MSE accuracy"); lower=better
                final_train_mse = float(np.mean((preds_tr - y_tr_s) ** 2))
                final_test_mse = float(np.mean((preds_te - y_te_s) ** 2))

                best_equation = str(model.equations_.loc[best_idx, "equation"])
                print(
                    f"init_{i} | seed={seed} | train_mse={final_train_mse:.4f} "
                    f"| test_mse={final_test_mse:.4f} | eq={best_equation}"
                )

                payload = {
                    "name": dataset_name,
                    "did": int(row["did"]),
                    "tid": int(row["tid"]),
                    "fold": FOLD,
                    "task": "regression",
                    "model_type": "symbolic_regression",
                    "n_classes": 1,  # single regression output
                    "in_dim": X_tr.shape[1],
                    "n_train": X_tr.shape[0],
                    "n_test": X_te.shape[0],
                    "y_scaler": y_scaler,
                    "n_inits": N_INITS,
                    "seed": seed,
                    "n_generations": N_GENERATIONS,
                    "final_test_mse": final_test_mse,
                    "final_train_mse": final_train_mse,
                    # Store the discovered equations (picklable DataFrame) and the best one,
                    # rather than the live PySR/Julia object.
                    "equations": model.equations_.copy(),
                    "best_index": int(best_idx),
                    "best_equation": best_equation,
                    "trace": trace,
                }

                with open(f"{out_path}_seed{seed}_{dataset_name}.pkl", "wb") as f:
                    pickle.dump(payload, f)
                print(f"Saved to {out_path}_seed{seed}_{dataset_name}.pkl")

        except Exception as e:
            print(f"Dataset {dataset_name} was impossible to process: {e}")
            continue
