import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from common import (CST_COLUMNS, INPUT_COLUMNS, OUTPUT_COLUMNS, AirfoilMLP, Scaler,
                    count_parameters, make_features)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAINING_RUN_COLUMNS = ["run_id", "date", "learning_rate", "batch_size", "max_epochs", "epochs_run",
                        "n_params", "n_train_airfoils", "n_val_airfoils", "n_test_airfoils", "n_train_rows",
                        "best_val_loss", "train_time_s", "test_CL_MAE", "test_CL_R2", "test_CD_MAPE_pct", "notes"]

def load_clean_data(verbose=True):
    for path, script in [(config.AIRFOILS_CSV, "01_prepare_airfoils.py"), (config.XFOIL_CSV, "02_generate_xfoil_data.py")]:
        if not path.exists():
            raise SystemExit(f"{path.name} not found. Run `python {script}` first.")
    airfoils = pd.read_csv(config.AIRFOILS_CSV)
    results = pd.read_csv(config.XFOIL_CSV)

    bad_fit = (airfoils[CST_COLUMNS].abs() > config.CST_MAX_ABS).any(axis=1)
    if verbose and bad_fit.any():
        print(f"Leaving out {int(bad_fit.sum())} airfoils with a failed CST fit: {', '.join(airfoils.loc[bad_fit, 'name'])}")
    airfoils = airfoils[~bad_fit]
    data = results.merge(airfoils[["airfoil_id", "name"] + CST_COLUMNS], on="airfoil_id", how="inner")
    n_before = len(data)

    data = data.dropna(subset=CST_COLUMNS + ["alpha", "Re", "CL", "CD"])
    data = data[data["CD"] > 0]
    data = data[data["CL"].abs() <= config.CL_MAX_ABS]
    data = data.drop_duplicates(subset=["airfoil_id", "Re", "alpha"])
    data = data.reset_index(drop=True)
    data["log10_CD"] = np.log10(data["CD"])

    if verbose:
        print(f"Rows: {n_before} before cleaning, {len(data)} after ({n_before - len(data)} removed)")
    return data

def split_by_airfoil(airfoil_ids, seed=config.RANDOM_SEED):
    ids = np.array(sorted(set(airfoil_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(round(config.TRAIN_FRACTION * len(ids)))
    n_val = int(round(config.VAL_FRACTION * len(ids)))
    groups = {}
    for i, airfoil_id in enumerate(ids):
        groups[airfoil_id] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return groups

def add_split(data, verbose=True):
    groups = split_by_airfoil(data["airfoil_id"])
    data = data.copy()
    data["split"] = data["airfoil_id"].map(groups)
    pd.DataFrame(sorted(groups.items()), columns=["airfoil_id", "split"]).to_csv(config.SPLIT_CSV, index=False)

    counts = {s: sum(1 for g in groups.values() if g == s) for s in ("train", "val", "test")}
    if verbose:
        sets = {s: set(data.loc[data["split"] == s, "airfoil_id"]) for s in counts}
        overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
        print(f"Airfoils: train {counts['train']}, val {counts['val']}, test {counts['test']} "
              f"(total {len(groups)}); shared between groups: {len(overlap)}")
    return data, counts

def inspect(data):
    print("\nValue ranges:")
    print(data[["alpha", "Re", "CL", "CD", "CM"]].describe().round(4).to_string())
    print(f"\nRows per Reynolds number:\n{data['Re'].value_counts().sort_index().to_string()}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].hist(data["CL"], bins=60); axes[0, 0].set_title("CL")
    axes[0, 1].hist(data["log10_CD"], bins=60); axes[0, 1].set_title("log10(CD)")
    axes[0, 2].hist(data["alpha"], bins=len(config.ALPHAS)); axes[0, 2].set_title("alpha (deg)")
    re_counts = data["Re"].value_counts().sort_index()
    axes[1, 0].bar([f"{r:.0e}" for r in re_counts.index], re_counts.values); axes[1, 0].set_title("rows per Re")

    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_ids = rng.choice(data["airfoil_id"].unique(), size=min(5, data["airfoil_id"].nunique()), replace=False)
    re_plot = 1e6 if 1e6 in set(data["Re"]) else data["Re"].mode()[0]
    for airfoil_id in sample_ids:
        rows = data[(data["airfoil_id"] == airfoil_id) & (data["Re"] == re_plot)].sort_values("alpha")
        label = rows["name"].iloc[0] if len(rows) else airfoil_id
        axes[1, 1].plot(rows["alpha"], rows["CL"], ".-", label=label)
        axes[1, 2].plot(rows["CD"], rows["CL"], ".-", label=label)
    axes[1, 1].set_title(f"CL vs alpha, 5 random airfoils (Re={re_plot:.0e})"); axes[1, 1].set_xlabel("alpha (deg)")
    axes[1, 2].set_title("CL vs CD (drag polar)"); axes[1, 2].set_xlabel("CD")
    axes[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.DATA_OVERVIEW_PNG, dpi=130)
    print(f"\nSaved {config.DATA_OVERVIEW_PNG}")

def train(data, counts, epochs, lr, batch_size):
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    train_df = data[data["split"] == "train"]
    val_df = data[data["split"] == "val"]
    if train_df.empty or val_df.empty:
        raise SystemExit("Not enough airfoils to make train and validation groups. Generate more XFOIL data.")

    x_train, x_val = make_features(train_df), make_features(val_df)
    y_train = train_df[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_df[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)

    x_scaler, y_scaler = Scaler().fit(x_train), Scaler().fit(y_train)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_scaler.transform(x_train)), torch.from_numpy(y_scaler.transform(y_train))),
        batch_size=batch_size, shuffle=True)
    x_val_t = torch.from_numpy(x_scaler.transform(x_val))
    y_val_t = torch.from_numpy(y_scaler.transform(y_val))

    model = AirfoilMLP()
    n_params = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    print(f"\nTraining {n_params:,}-parameter network on {len(train_df)} rows "
          f"(validating on {len(val_df)}) | lr={lr} batch={batch_size} max_epochs={epochs}")

    history, best_val, best_state, best_epoch, waited = [], float("inf"), None, 0, 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val_t), y_val_t).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}  train loss {train_loss:.5f}  val loss {val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config.PATIENCE:
                print(f"No improvement for {config.PATIENCE} epochs - stopping early.")
                break
    train_time = time.time() - start
    model.load_state_dict(best_state)
    print(f"\nBest validation loss {best_val:.5f} at epoch {best_epoch} | training time {train_time:.1f}s")

    torch.save({"state_dict": model.state_dict(), "hidden_sizes": list(config.HIDDEN_SIZES),
                "input_columns": INPUT_COLUMNS, "output_columns": OUTPUT_COLUMNS}, config.MODEL_PATH)
    ranges = {
        "alpha": [float(train_df["alpha"].min()), float(train_df["alpha"].max())],
        "Re": [float(train_df["Re"].min()), float(train_df["Re"].max())],
        "cst_min": train_df[CST_COLUMNS].min().round(6).tolist(),
        "cst_max": train_df[CST_COLUMNS].max().round(6).tolist(),
    }
    config.SCALERS_PATH.write_text(json.dumps(
        {"x_scaler": x_scaler.to_dict(), "y_scaler": y_scaler.to_dict(), "training_ranges": ranges}, indent=2))

    hist = pd.DataFrame(history)
    hist.to_csv(config.HISTORY_CSV, index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["epoch"], hist["train_loss"], label="training")
    ax.plot(hist["epoch"], hist["val_loss"], label="validation")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss (scaled MSE)")
    ax.set_title("Training progress"); ax.legend()
    fig.tight_layout(); fig.savefig(config.LOSS_CURVE_PNG, dpi=130)

    now = datetime.now()
    run = {"run_id": now.strftime("run_%Y%m%d_%H%M%S"), "date": now.strftime("%Y-%m-%d"),
           "learning_rate": lr, "batch_size": batch_size, "max_epochs": epochs, "epochs_run": len(history),
           "n_params": n_params, "n_train_airfoils": counts["train"], "n_val_airfoils": counts["val"],
           "n_test_airfoils": counts["test"], "n_train_rows": len(train_df), "best_val_loss": round(best_val, 6),
           "train_time_s": round(train_time, 1), "test_CL_MAE": "", "test_CL_R2": "", "test_CD_MAPE_pct": "",
           "notes": ""}
    pd.DataFrame([run], columns=TRAINING_RUN_COLUMNS).to_csv(
        config.TRAINING_RUNS_CSV, mode="a", header=not config.TRAINING_RUNS_CSV.exists(), index=False)

    print(f"Saved {config.MODEL_PATH.name}, {config.SCALERS_PATH.name}, {config.HISTORY_CSV.name}, "
          f"{config.LOSS_CURVE_PNG.name}; logged {run['run_id']} in {config.TRAINING_RUNS_CSV.name}")

def main():
    parser = argparse.ArgumentParser(description="Train the airfoil surrogate network.")
    parser.add_argument("--inspect", action="store_true", help="only summarize the data, no training")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    config.ensure_dirs()
    data = load_clean_data()
    data, counts = add_split(data)
    if args.inspect:
        inspect(data)
    else:
        train(data, counts, args.epochs, args.lr, args.batch_size)

if __name__ == "__main__":
    main()
