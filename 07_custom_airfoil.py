import argparse
import importlib
import warnings

import numpy as np
import pandas as pd

import config
from common import cst_from_coordinates, load_trained_model, make_features, predict, read_dat_file

predict_tool = importlib.import_module("05_predict")

def reshape(airfoil, thickness=None, thickness_scale=None, camber=None, camber_scale=None, n=100):
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    t = np.asarray(airfoil.local_thickness(x_over_c=x), dtype=float)
    c = np.asarray(airfoil.local_camber(x_over_c=x), dtype=float)

    if thickness is not None:
        t = t * thickness / t.max()
    elif thickness_scale is not None:
        t = t * thickness_scale
    if camber is not None:
        biggest = np.abs(c).max()
        if biggest < 1e-6 and camber != 0:
            raise SystemExit("The starting airfoil is symmetric (no camber), so it has no camber line to stretch. "
                             "Start from a cambered airfoil, e.g. --airfoil naca2412.")
        c = c * (camber / biggest if biggest > 0 else 0)
    elif camber_scale is not None:
        c = c * camber_scale

    upper = np.column_stack([x, c + t / 2])
    lower = np.column_stack([x, c - t / 2])
    return np.vstack([upper[::-1], lower[1:]])

def main():
    parser = argparse.ArgumentParser(description="Change an airfoil's thickness/camber and predict CL and CD.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--airfoil", help="starting airfoil name, e.g. naca2412 or e387")
    source.add_argument("--dat", help="starting airfoil coordinate file")
    parser.add_argument("--thickness", type=float, help="new maximum thickness as a fraction of chord, e.g. 0.15")
    parser.add_argument("--thickness-scale", type=float, help="multiply the thickness by this, e.g. 1.2")
    parser.add_argument("--camber", type=float, help="new maximum camber as a fraction of chord, e.g. 0.03")
    parser.add_argument("--camber-scale", type=float, help="multiply the camber by this, e.g. 0.5")
    parser.add_argument("--alpha", type=float, nargs="+", required=True, help="angle(s) of attack in degrees")
    parser.add_argument("--re", type=float, required=True, help="Reynolds number, e.g. 500000 or 5e5")
    parser.add_argument("--save", help="save the new shape to this .dat file")
    parser.add_argument("--plot", action="store_true", help="save a picture of the old and new shape")
    args = parser.parse_args()

    if args.thickness is not None and args.thickness_scale is not None:
        raise SystemExit("Use either --thickness or --thickness-scale, not both.")
    if args.camber is not None and args.camber_scale is not None:
        raise SystemExit("Use either --camber or --camber-scale, not both.")
    if args.thickness is not None and not 0.01 <= args.thickness <= 0.40:
        raise SystemExit("--thickness should be between 0.01 and 0.40 (1% to 40% of the chord).")

    import aerosandbox as asb
    name, coords = predict_tool.shape_from_arguments(args)
    base = asb.Airfoil(name="base", coordinates=coords)
    new_coords = reshape(base, args.thickness, args.thickness_scale, args.camber, args.camber_scale)
    new = asb.Airfoil(name="custom", coordinates=new_coords)

    print(f"\nStarting airfoil: {name}")
    print(f"  thickness {base.max_thickness() * 100:5.2f}%   camber {base.max_camber() * 100:5.2f}%")
    print(f"New airfoil:")
    print(f"  thickness {new.max_thickness() * 100:5.2f}%   camber {new.max_camber() * 100:5.2f}%")

    model, x_scaler, y_scaler, info = load_trained_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cst_old, _ = cst_from_coordinates(coords, name="base")
        cst_new, _ = cst_from_coordinates(new_coords, name="custom")

    print(f"\nPredictions at Re = {args.re:,.0f}  (old shape -> new shape)\n")
    print(f"{'alpha':>7} {'CL old':>8} {'CL new':>8} {'CD old':>9} {'CD new':>9} {'L/D old':>8} {'L/D new':>8}")
    results = {}
    for label, cst in (("old", cst_old), ("new", cst_new)):
        table = pd.DataFrame([cst] * len(args.alpha))
        table["alpha"], table["Re"] = args.alpha, args.re
        results[label] = predict(model, x_scaler, y_scaler, make_features(table))
    for i, a in enumerate(args.alpha):
        clo, cdo = results["old"][0][i], results["old"][1][i]
        cln, cdn = results["new"][0][i], results["new"][1][i]
        print(f"{a:7.2f} {clo:8.4f} {cln:8.4f} {cdo:9.5f} {cdn:9.5f} {clo / cdo:8.1f} {cln / cdn:8.1f}")

    notes = predict_tool.range_warnings(cst_new, args.alpha, args.re, info)
    if notes:
        print("\nWARNING - trust the new-shape numbers less:")
        for note in notes:
            print(f"  - {note}")

    if args.save:
        lines = ["custom airfoil"] + [f"{x:.6f} {y:.6f}" for x, y in new_coords]
        with open(args.save, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nSaved the new shape to {args.save}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        config.ensure_dirs()
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(base.x(), base.y(), "k--", lw=1, label=f"{name} (start)")
        ax.plot(new.x(), new.y(), "r-", lw=1.5, label="new shape")
        ax.set_aspect("equal"); ax.legend(); ax.set_xlabel("x / chord")
        fig.tight_layout()
        path = config.RESULTS / "custom_airfoil.png"
        fig.savefig(path, dpi=130)
        print(f"Saved {path}")
    print("\n(Predictions from the neural network. Check important answers with XFOIL.)")

if __name__ == "__main__":
    main()
