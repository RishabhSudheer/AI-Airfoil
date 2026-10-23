import argparse
import warnings

import pandas as pd

from common import CST_COLUMNS, cst_from_coordinates, load_trained_model, make_features, predict, read_dat_file

def shape_from_arguments(args):
    if args.dat:
        title, coords = read_dat_file(args.dat)
        return title, coords
    import aerosandbox as asb
    airfoil = asb.Airfoil(args.airfoil)
    if airfoil.coordinates is None:
        raise SystemExit(f"AeroSandbox doesn't know an airfoil called '{args.airfoil}'. "
                         "Use --dat with a coordinate file instead.")
    return args.airfoil, airfoil.coordinates

def range_warnings(cst, alphas, reynolds, info):
    ranges = info.get("training_ranges", {})
    notes = []
    a_lo, a_hi = ranges.get("alpha", [None, None])
    if a_lo is not None and (min(alphas) < a_lo or max(alphas) > a_hi):
        notes.append(f"alpha outside the trained range {a_lo:g} to {a_hi:g} degrees")
    r_lo, r_hi = ranges.get("Re", [None, None])
    if r_lo is not None and not (r_lo <= reynolds <= r_hi):
        notes.append(f"Re outside the trained range {r_lo:.0e} to {r_hi:.0e}")
    if "cst_min" in ranges:
        outside = [c for c, lo, hi in zip(CST_COLUMNS, ranges["cst_min"], ranges["cst_max"])
                   if not (lo <= cst[c] <= hi)]
        if outside:
            notes.append(f"shape numbers outside the training airfoils' range: {', '.join(outside)}")
    return notes

def main():
    parser = argparse.ArgumentParser(description="Predict CL and CD for an airfoil with the trained network.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--airfoil", help="airfoil name, e.g. naca4412 or e387")
    source.add_argument("--dat", help="path to an airfoil .dat coordinate file")
    parser.add_argument("--alpha", type=float, nargs="+", required=True, help="angle(s) of attack in degrees")
    parser.add_argument("--re", type=float, required=True, help="Reynolds number, e.g. 500000 or 5e5")
    args = parser.parse_args()

    model, x_scaler, y_scaler, info = load_trained_model()
    name, coords = shape_from_arguments(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cst, _ = cst_from_coordinates(coords, name="query")

    table = pd.DataFrame([cst] * len(args.alpha))
    table["alpha"], table["Re"] = args.alpha, args.re
    cl, cd = predict(model, x_scaler, y_scaler, make_features(table))

    print(f"\n{name}  at  Re = {args.re:,.0f}\n")
    print(f"{'alpha':>7} {'CL':>8} {'CD':>9} {'L/D':>8}")
    for a, l, d in zip(args.alpha, cl, cd):
        print(f"{a:7.2f} {l:8.4f} {d:9.5f} {l / d:8.1f}")

    notes = range_warnings(cst, args.alpha, args.re, info)
    if notes:
        print("\nWARNING - trust these numbers less:")
        for note in notes:
            print(f"  - {note}")
    print("\n(Predictions from the neural network. Check important answers with XFOIL.)")

if __name__ == "__main__":
    main()
