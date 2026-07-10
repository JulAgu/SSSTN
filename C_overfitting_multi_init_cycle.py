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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(f"cuda device = {DEVICE}")
    parser = argparse.ArgumentParser(
        description="Overfitting exploration with randomized labels over multiple "
        "weight initializations on continuous-only CC18 datasets."
    )
    add_datasets_arg(parser)
    args = parser.parse_args()

    selected = select_tasks("classification", args.datasets)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    for _, row in selected.iterrows():
        dataset_name = row["name"]
        out_path = os.path.join(
            out_dir,
            f"randlabels_fold{FOLD}_layers{N_LAYERS}_hidden{HIDDEN}_epc{EPOCHS}_ninits{N_INITS}_",
        )

        try:
            X, y = load_xy(row["did"])
            print(
                f"Loaded {dataset_name}  (did={int(row['did'])}, tid={int(row['tid'])})"
            )

            # Held-out test split keeps its TRUE labels: a random-label net's test
            # accuracy on real labels is the Zhang et al. memorization control (~chance).
            X_tr, y_tr, X_te, y_te, le, n_classes = prepare_classification(
                X, y, row["tid"], fold=FOLD
            )

            # permute y_tr to break any correlation with X_tr and test for overfitting.
            # The permutation is fixed per dataset so every initialization memorizes the same randomized labels. Only the weight init seed varies.
            perm = np.random.permutation(len(y_tr))
            y_tr = y_tr[perm]

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
                    "perm": perm,
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
