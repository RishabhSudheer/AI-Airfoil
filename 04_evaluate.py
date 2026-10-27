import json

import numpy as np
import pandas as pd

import config
from common import CST_COLUMNS, load_trained_model, make_features, predict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib
train_module = importlib.import_module("03_train")

def r2_score(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def metrics_for(frame):
    if frame.empty:
        return {"n_points": 0}
    return {
        "n_points": int(len(frame)),
        "CL_MAE": round(float(np.mean(np.abs(frame["CL_pred"] - frame["CL_xfoil"]))), 5),
        "CL_R2": round(r2_score(frame["CL_xfoil"].to_numpy(), frame["CL_pred"].to_numpy()), 5),
        "CD_MAPE_pct": round(float(np.mean(np.abs(frame["CD_pct_error"]))), 3),
    }

def parity_plot(true, pred, label, path, log=False):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true, pred, s=3, alpha=0.35)
    low, high = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="perfect")
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"{label} from XFOIL"); ax.set_ylabel(f"{label} from model")
    ax.set_title(f"Test airfoils: {label}"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def polar_plots(test_df, model, x_scaler, y_scaler):
    for old in config.POLARS.glob("*.png"):
        old.unlink()
    counts = test_df.groupby("airfoil_id").size().sort_values(ascending=False)
    rng = np.random.default_rng(config.RANDOM_SEED)
    pool = counts.index[: max(config.N_POLAR_AIRFOILS * 5, config.N_POLAR_AIRFOILS)]
    chosen = rng.choice(pool, size=min(config.N_POLAR_AIRFOILS, len(pool)), replace=False)
    fine_alpha = np.arange(min(config.ALPHAS), max(config.ALPHAS) + 0.01, 0.25)

    for airfoil_id in chosen:
        rows = test_df[test_df["airfoil_id"] == airfoil_id]
        name = rows["name"].iloc[0]
        fig, (ax_cl, ax_cd) = plt.subplots(1, 2, figsize=(11, 4.2))
        for i, reynolds in enumerate(sorted(rows["Re"].unique())):
            color = f"C{i}"
            pts = rows[rows["Re"] == reynolds].sort_values("alpha")

            curve = pd.DataFrame({c: pts.iloc[0][c] for c in CST_COLUMNS}, index=range(len(fine_alpha)))
            curve["alpha"], curve["Re"] = fine_alpha, reynolds
            cl, cd = predict(model, x_scaler, y_scaler, make_features(curve))
            ax_cl.plot(pts["alpha"], pts["CL_xfoil"], "o", color=color, markersize=3)
            ax_cl.plot(fine_alpha, cl, "-", color=color, label=f"Re={reynolds:.0e}")
            ax_cd.plot(pts["CD_xfoil"], pts["CL_xfoil"], "o", color=color, markersize=3)
            ax_cd.plot(cd, cl, "-", color=color)
        ax_cl.set_xlabel("alpha (deg)"); ax_cl.set_ylabel("CL"); ax_cl.set_title(f"{name}: CL vs alpha")
        ax_cd.set_xlabel("CD"); ax_cd.set_ylabel("CL"); ax_cd.set_title(f"{name}: drag polar")
        ax_cl.legend(fontsize=8)
        fig.suptitle("dots = XFOIL, lines = neural network", fontsize=9)
        fig.tight_layout()
        fig.savefig(config.POLARS / f"{name}.png", dpi=120); plt.close(fig)
    print(f"Saved {len(chosen)} polar plots in {config.POLARS}")

def update_training_runs(metrics):
    if not config.TRAINING_RUNS_CSV.exists():
        return
    runs = pd.read_csv(config.TRAINING_RUNS_CSV)
    if runs.empty:
        return
    for column in ("test_CL_MAE", "test_CL_R2", "test_CD_MAPE_pct"):
        runs[column] = runs[column].astype("object")
    runs.loc[runs.index[-1], "test_CL_MAE"] = metrics["CL_MAE"]
    runs.loc[runs.index[-1], "test_CL_R2"] = metrics["CL_R2"]
    runs.loc[runs.index[-1], "test_CD_MAPE_pct"] = metrics["CD_MAPE_pct"]
    runs.to_csv(config.TRAINING_RUNS_CSV, index=False)
    print(f"Added test scores to run {runs.iloc[-1]['run_id']} in {config.TRAINING_RUNS_CSV.name}")

def main():
    config.ensure_dirs()
    model, x_scaler, y_scaler, _ = load_trained_model()
    data = train_module.load_clean_data(verbose=False)
    data, _ = train_module.add_split(data, verbose=False)
    test = data[data["split"] == "test"].copy()
    if test.empty:
        raise SystemExit("No test airfoils found. Generate more XFOIL data and retrain.")

    cl_pred, cd_pred = predict(model, x_scaler, y_scaler, make_features(test))
    test["CL_xfoil"], test["CL_pred"] = test["CL"], cl_pred
    test["CL_error"] = test["CL_pred"] - test["CL_xfoil"]
    test["CD_xfoil"], test["CD_pred"] = test["CD"], cd_pred
    test["CD_pct_error"] = 100 * (test["CD_pred"] - test["CD_xfoil"]) / test["CD_xfoil"]
    test["regime"] = np.where(test["alpha"] <= config.STALL_ALPHA, "attached", "near_stall")

    metrics = {
        "overall": metrics_for(test),
        "attached (alpha <= %g)" % config.STALL_ALPHA: metrics_for(test[test["regime"] == "attached"]),
        "near_stall (alpha > %g)" % config.STALL_ALPHA: metrics_for(test[test["regime"] == "near_stall"]),
        "n_test_airfoils": int(test["airfoil_id"].nunique()),
    }
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))

    print(f"\nTest set: {metrics['n_test_airfoils']} airfoils the model never saw, {len(test)} points\n")
    print(f"{'group':<26}{'points':>8}{'CL MAE':>10}{'CL R^2':>10}{'CD error %':>12}")
    for group, m in metrics.items():
        if isinstance(m, dict) and m.get("n_points"):
            print(f"{group:<26}{m['n_points']:>8}{m['CL_MAE']:>10.4f}{m['CL_R2']:>10.4f}{m['CD_MAPE_pct']:>12.2f}")
    print(f"\nSaved {config.METRICS_JSON}")

    columns = ["airfoil_id", "name", "Re", "alpha", "regime", "CL_xfoil", "CL_pred", "CL_error",
               "CD_xfoil", "CD_pred", "CD_pct_error"]
    test[columns].round(6).to_csv(config.PREDICTIONS_CSV, index=False)
    print(f"Saved {config.PREDICTIONS_CSV}")

    parity_plot(test["CL_xfoil"], test["CL_pred"], "CL", config.PARITY_CL_PNG)
    parity_plot(test["CD_xfoil"], test["CD_pred"], "CD", config.PARITY_CD_PNG, log=True)
    print(f"Saved {config.PARITY_CL_PNG.name} and {config.PARITY_CD_PNG.name}")
    polar_plots(test, model, x_scaler, y_scaler)
    update_training_runs(metrics["overall"])

if __name__ == "__main__":
    main()
