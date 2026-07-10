import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

rows = []
for pkl_path in sorted(Path("results/randomization_test").glob("*.pkl")):
    print(pkl_path)
    exp = trace = y_tr = None
    try:
        with open(pkl_path, "rb") as f:
            exp = pickle.load(f)

        trace = exp["trace"]
        is_regression = exp.get("task") == "regression"

        # Columns shared by both task types.
        row = {
            "dataset": exp["name"],
            "task": exp.get("task", "classification"),
            "n_classes": exp["n_classes"],
            "in_dim": exp["in_dim"],
            "n_train": exp["n_train"],
            "fold": exp["fold"],
            "final_loss": round(trace["loss"][-1], 4),
            "best_loss": round(min(trace["loss"]), 4),
            "n_sgd_steps": trace["n_sgd_steps"],
            "epochs": len(trace["loss"]),
        }

        if is_regression:
            # Regression: MSE is the fitness (lower = better), R^2 the secondary
            # metric. There is no majority-class baseline, so chance_acc is N/A.
            row.update(
                {
                    "chance_acc": None,
                    "final_train_mse": round(exp["final_train_mse"], 4),
                    "final_test_mse": round(exp["final_test_mse"], 4),
                    "final_train_mse_from_trace": round(trace["mse"][-1], 4),
                    "final_r2": round(trace["r2"][-1], 4),
                    "best_mse": round(min(trace["mse"]), 4),
                    "best_r2": round(max(trace["r2"]), 4),
                }
            )
        else:
            # Classification: accuracy is the fitness, F1 the secondary metric;
            # chance_acc is the majority-class frequency of the train labels.
            if "y_tr" in exp:
                y_tr = exp["y_tr"]
                _, counts = np.unique(y_tr, return_counts=True)
                chance_acc = round(float(counts.max() / counts.sum()), 4)
            else:
                chance_acc = None
            row.update(
                {
                    "chance_acc": chance_acc,
                    "final_train_acc": round(exp["final_train_acc"], 4),
                    "final_test_acc": round(exp["final_test_acc"], 4),
                    "final_train_acc_from_trace": round(trace["acc"][-1], 4),
                    "final_f1": round(trace["f1"][-1], 4),
                    "best_acc": round(max(trace["acc"]), 4),
                    "best_f1": round(max(trace["f1"]), 4),
                }
            )
        rows.append(row)
    except Exception as e:
        print(f" Error loading {pkl_path}: {e}")
        continue
    finally:
        del exp, trace
        gc.collect()


df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.4f}".format)
print(df.to_string(index=False))

df.to_csv("results/summarizing/summary.csv", index=False)
print(f"\nSaved summary.csv")
