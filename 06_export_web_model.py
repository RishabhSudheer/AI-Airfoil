import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import config
from common import CST_COLUMNS

def main():
    parser = argparse.ArgumentParser(description="Export the trained network for the web interface.")
    parser.add_argument("--xfoil-csv", default=str(config.XFOIL_CSV),
                        help="XFOIL results to include as comparison dots (default: the normal results file)")
    parser.add_argument("--out", default=str(config.MODELS / "web_model.json"))
    args = parser.parse_args()

    saved = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
    state = saved["state_dict"]

    layer_keys = sorted((k for k in state if k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    layers = [{"W": np.round(state[k].numpy().astype(float), 7).tolist(),
               "b": np.round(state[k[:-7] + ".bias"].numpy().astype(float), 7).tolist()} for k in layer_keys]

    scalers = json.loads(config.SCALERS_PATH.read_text())

    airfoils = pd.read_csv(config.AIRFOILS_CSV)
    split = pd.read_csv(config.SPLIT_CSV) if config.SPLIT_CSV.exists() else pd.DataFrame(columns=["airfoil_id", "split"])
    airfoils = airfoils.merge(split, on="airfoil_id", how="left")
    airfoils["split"] = airfoils["split"].fillna("unused")

    airfoils = airfoils[~(airfoils[CST_COLUMNS].abs() > config.CST_MAX_ABS).any(axis=1)]
    airfoil_list = []
    for _, row in airfoils.iterrows():
        airfoil_list.append({
            "id": row["airfoil_id"], "name": str(row["name"]), "split": row["split"],
            "t": round(float(row["max_thickness"]), 4),
            "cst": [round(float(row[c]), 5) for c in CST_COLUMNS],
        })

    xfoil = {}
    try:
        res = pd.read_csv(args.xfoil_csv)
        for (aid, re), g in res.groupby(["airfoil_id", "Re"]):
            g = g.sort_values("alpha")
            xfoil.setdefault(aid, {})[str(int(re))] = [[round(a, 2), round(cl, 4), round(cd, 6)]
                                                      for a, cl, cd in zip(g.alpha, g.CL, g.CD)]
    except FileNotFoundError:
        print("No XFOIL results found - the web page will show model curves only.")

    metrics = json.loads(config.METRICS_JSON.read_text()) if config.METRICS_JSON.exists() else {}

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "input_columns": saved["input_columns"], "output_columns": saved["output_columns"],
        "layers": layers, "x_scaler": scalers["x_scaler"], "y_scaler": scalers["y_scaler"],
        "training_ranges": scalers.get("training_ranges", {}),
        "reynolds_numbers": list(config.REYNOLDS_NUMBERS),
        "test_metrics": metrics, "airfoils": airfoil_list, "xfoil": xfoil,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    n_params = sum(np.size(l["W"]) + np.size(l["b"]) for l in layers)
    print(f"Saved {args.out}: {n_params:,} parameters, {len(airfoil_list)} airfoils, "
          f"XFOIL dots for {len(xfoil)} airfoils")

if __name__ == "__main__":
    main()
