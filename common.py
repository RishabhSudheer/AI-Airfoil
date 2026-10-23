import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

import config

UPPER_COLUMNS = [f"upper_w{i}" for i in range(1, config.N_CST + 1)]
LOWER_COLUMNS = [f"lower_w{i}" for i in range(1, config.N_CST + 1)]
CST_COLUMNS = UPPER_COLUMNS + LOWER_COLUMNS + ["le_weight", "te_thickness"]

INPUT_COLUMNS = CST_COLUMNS + ["alpha", "log10_Re"]
OUTPUT_COLUMNS = ["CL", "log10_CD"]

def read_dat_file(path):
    text = Path(path).read_text(encoding="latin-1")
    title = None
    rows = []
    blocks = [[]]

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        if not line:
            if blocks[-1]:
                blocks.append([])
            continue
        if title is None:
            title = line
            continue
        parts = line.replace(",", " ").split()
        try:
            numbers = [float(p) for p in parts[:2]]
        except ValueError:
            continue
        if len(numbers) == 2 and all(math.isfinite(v) for v in numbers):
            rows.append(numbers)
            blocks[-1].append(numbers)

    if not rows:
        raise ValueError("no coordinate numbers found")

    if rows[0][0] > 1.5 and rows[0][1] > 1.5:
        n_upper, n_lower = int(rows[0][0]), int(rows[0][1])
        points = rows[1:]
        if len(points) < n_upper + n_lower:
            raise ValueError("Lednicer file is shorter than its point counts")
        upper = np.array(points[:n_upper])
        lower = np.array(points[n_upper:n_upper + n_lower])
        coords = np.vstack([upper[::-1], lower[1:]])
    else:
        coords = np.array(rows)

    if len(coords) < config.MIN_POINTS:
        raise ValueError(f"only {len(coords)} points (need at least {config.MIN_POINTS})")
    x_span = coords[:, 0].max() - coords[:, 0].min()
    if x_span <= 0:
        raise ValueError("x coordinates do not span a chord")
    return (title or Path(path).stem), coords

def cst_from_coordinates(coords, name="airfoil"):
    import aerosandbox as asb

    airfoil = asb.Airfoil(name=name, coordinates=np.asarray(coords, dtype=float))
    kulfan = airfoil.to_kulfan_airfoil(n_weights_per_side=config.N_CST)
    values = {}
    for i, w in enumerate(np.asarray(kulfan.upper_weights).ravel(), start=1):
        values[f"upper_w{i}"] = float(w)
    for i, w in enumerate(np.asarray(kulfan.lower_weights).ravel(), start=1):
        values[f"lower_w{i}"] = float(w)
    values["le_weight"] = float(kulfan.leading_edge_weight)
    values["te_thickness"] = float(kulfan.TE_thickness)
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("CST fit produced invalid numbers")
    return values, airfoil

def airfoil_from_cst(row, name="airfoil"):
    import aerosandbox as asb

    kulfan = asb.KulfanAirfoil(
        name=name,
        upper_weights=np.array([row[c] for c in UPPER_COLUMNS], dtype=float),
        lower_weights=np.array([row[c] for c in LOWER_COLUMNS], dtype=float),
        leading_edge_weight=float(row["le_weight"]),
        TE_thickness=float(row["te_thickness"]),
    )
    return kulfan.to_airfoil()

def make_features(df):
    table = df.copy()
    table["log10_Re"] = np.log10(table["Re"].astype(float))
    return table[INPUT_COLUMNS].to_numpy(dtype=np.float32)

class AirfoilMLP(nn.Module):
    def __init__(self, n_inputs=len(INPUT_COLUMNS), hidden_sizes=config.HIDDEN_SIZES,
                 n_outputs=len(OUTPUT_COLUMNS)):
        super().__init__()
        layers = []
        size_in = n_inputs
        for size_out in hidden_sizes:
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())
            size_in = size_out
        layers.append(nn.Linear(size_in, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Scaler:
    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def fit(self, array):
        self.mean = array.mean(axis=0).astype(np.float32)
        std = array.std(axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0
        self.std = std
        return self

    def transform(self, array):
        return ((array - self.mean) / self.std).astype(np.float32)

    def inverse(self, array):
        return (array * self.std + self.mean).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])

def load_trained_model():
    if not config.MODEL_PATH.exists() or not config.SCALERS_PATH.exists():
        raise SystemExit("No trained model found. Run `python 03_train.py` first.")
    checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    model = AirfoilMLP(hidden_sizes=tuple(checkpoint["hidden_sizes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    info = json.loads(config.SCALERS_PATH.read_text())
    return model, Scaler.from_dict(info["x_scaler"]), Scaler.from_dict(info["y_scaler"]), info

def predict(model, x_scaler, y_scaler, features):
    with torch.no_grad():
        x = torch.from_numpy(x_scaler.transform(features))
        y_scaled = model(x).numpy()
    y = y_scaler.inverse(y_scaled)
    cl = y[:, 0]
    cd = 10.0 ** y[:, 1]
    return cl, cd

if __name__ == "__main__":
    model = AirfoilMLP()
    print(model)
    print()
    print(f"{'Layer':<28}{'Weights':>10}{'Biases':>10}{'Total':>10}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w, b = module.weight.numel(), module.bias.numel()
            label = f"{module.in_features} -> {module.out_features}"
            print(f"{label:<28}{w:>10,}{b:>10,}{w + b:>10,}")
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
