import os
import argparse
import pickle
import numpy as np
import torch
import xgboost as xgb
from engine.xgb_trainer import train_xgb
from utilities.data import add_datasets_arg, load_xy, prepare_classification, select_tasks
from tqdm import tqdm

np.random.seed(42)
BASE_SEED = 42
N_INITS = 30
FOLD = 0
N_ESTIMATORS = 1000
MAX_DEPTH = 6
ETA = 0.1
SUBSAMPLE = 0.8
COLSAMPLE = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(f"xgboost device = {DEVICE}")
    parser = argparse.ArgumentParser(
        description="XGBoost train/test cycle over continuous-only CC18 datasets "
        "across multiple subsampling seeds (real labels)."
    )
    add_datasets_arg(parser)
    args = parser.parse_args()

    selected = select_tasks("classification", args.datasets)

    out_dir = "results_xgb"
    os.makedirs(out_dir, exist_ok=True)

    for _, row in selected.iterrows():
        dataset_name = row["name"]
        out_path = os.path.join(
            out_dir,
            f"xgb_fold{FOLD}_depth{MAX_DEPTH}_nest{N_ESTIMATORS}_ninits{N_INITS}_",
        )

        try:
            X, y = load_xy(row["did"])
            print(
                f"Loaded {dataset_name}  (did={int(row['did'])}, tid={int(row['tid'])})"
            )

            X_tr, y_tr, X_te, y_te, le, n_classes = prepare_classification(
                X, y, row["tid"], fold=FOLD
            )

            dtest = xgb.DMatrix(X_te)
            dtrain = xgb.DMatrix(X_tr)
            for i in tqdm(range(N_INITS), desc=f"{dataset_name} inits"):
                seed = BASE_SEED + i

                booster, trace = train_xgb(
                    X_tr,
                    y_tr,
                    n_classes,
                    n_estimators=N_ESTIMATORS,
                    max_depth=MAX_DEPTH,
                    lr=ETA,
                    subsample=SUBSAMPLE,
                    colsample_bytree=COLSAMPLE,
                    seed=seed,
                    device=DEVICE,
                )

                preds_te = booster.predict(dtest).argmax(1)
                final_test_acc = float((preds_te == y_te).mean())
                preds_tr = booster.predict(dtrain).argmax(1)
                final_train_acc = float((preds_tr == y_tr).mean())

                print(f"init_{i} | seed={seed} | test_acc={final_test_acc:.4f}")

                payload = {
                    "name": dataset_name,
                    "did": int(row["did"]),
                    "tid": int(row["tid"]),
                    "fold": FOLD,
                    "model_type": "xgboost",
                    "n_classes": n_classes,
                    "in_dim": X_tr.shape[1],
                    "n_train": X_tr.shape[0],
                    "n_test": X_te.shape[0],
                    "le": le,
                    "n_inits": N_INITS,
                    "seed": seed,
                    "n_estimators": N_ESTIMATORS,
                    "max_depth": MAX_DEPTH,
                    "eta": ETA,
                    "final_test_acc": final_test_acc,
                    "final_train_acc": final_train_acc,
                    "model": booster,
                    "trace": trace,
                }

                with open(f"{out_path}_seed{seed}_{dataset_name}.pkl", "wb") as f:
                    pickle.dump(payload, f)
                print(f"Saved to {out_path}_seed{seed}_{dataset_name}.pkl")

        except Exception as e:
            print(f"Dataset {dataset_name} was impossible to process: {e}")
            continue
