import argparse
import urllib.request
import warnings
import zipfile

import pandas as pd

import config
from common import airfoil_from_cst, cst_from_coordinates, read_dat_file

def download_uiuc_zip():
    if config.UIUC_ZIP_PATH.exists():
        print(f"Zip already downloaded: {config.UIUC_ZIP_PATH}")
        return
    print(f"Downloading {config.UIUC_ZIP_URL} ...")
    request = urllib.request.Request(config.UIUC_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            config.UIUC_ZIP_PATH.write_bytes(response.read())
    except Exception as error:
        raise SystemExit(
            f"Download failed ({error}).\n"
            f"Download the zip in your browser from\n  {config.UIUC_ZIP_URL}\n"
            f"and save it as\n  {config.UIUC_ZIP_PATH}\nthen run this script again."
        )
    print(f"Saved {config.UIUC_ZIP_PATH.stat().st_size / 1e6:.1f} MB")

def unzip_uiuc():
    if config.UIUC_DIR.exists() and any(config.UIUC_DIR.rglob("*.dat")):
        return
    print("Unzipping ...")
    with zipfile.ZipFile(config.UIUC_ZIP_PATH) as archive:
        archive.extractall(config.UIUC_DIR)

    inner = [p for p in config.UIUC_DIR.iterdir() if p.is_dir()]
    if len(inner) == 1 and not any(config.UIUC_DIR.glob("*.dat")):
        for item in inner[0].iterdir():
            item.rename(config.UIUC_DIR / item.name)
        inner[0].rmdir()

def convert_all():
    dat_files = sorted(config.UIUC_DIR.rglob("*.dat"), key=lambda p: p.name.lower())
    if not dat_files:
        raise SystemExit(f"No .dat files found in {config.UIUC_DIR}")
    print(f"Found {len(dat_files)} airfoil files. Converting to CST (this takes a few minutes) ...")

    good_rows, failed_rows = [], []
    for index, path in enumerate(dat_files, start=1):
        airfoil_id = f"af{index:04d}"
        try:
            title, coords = read_dat_file(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cst, _ = cst_from_coordinates(coords, name=airfoil_id)
                rebuilt = airfoil_from_cst(cst, name=airfoil_id)
                max_thickness = float(rebuilt.max_thickness())
                max_camber = float(rebuilt.max_camber())
            if not (0.005 < max_thickness < 0.6):
                raise ValueError(f"unrealistic thickness after CST fit ({max_thickness:.3f})")
            good_rows.append({
                "airfoil_id": airfoil_id,
                "name": path.stem.lower(),
                "source_file": path.name,
                "description": title,
                "n_points": len(coords),
                "max_thickness": round(max_thickness, 5),
                "max_camber": round(max_camber, 5),
                **cst,
            })
        except Exception as error:
            failed_rows.append({"airfoil_id": airfoil_id, "source_file": path.name, "reason": str(error)})

        if index % 100 == 0:
            print(f"  {index}/{len(dat_files)} files processed")

    airfoils = pd.DataFrame(good_rows)
    airfoils.to_csv(config.AIRFOILS_CSV, index=False)
    pd.DataFrame(failed_rows, columns=["airfoil_id", "source_file", "reason"]).to_csv(config.FAILED_CSV, index=False)

    print(f"\nConverted: {len(good_rows)} airfoils -> {config.AIRFOILS_CSV}")
    print(f"Failed:    {len(failed_rows)} files -> {config.FAILED_CSV}")
    return airfoils

def plot_shapes(names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    airfoils = pd.read_csv(config.AIRFOILS_CSV)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.4 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        match = airfoils[airfoils["name"] == name.lower()]
        if match.empty:
            ax.set_title(f"{name}: not found in airfoils.csv")
            ax.axis("off")
            continue
        row = match.iloc[0]
        _, original = read_dat_file(next(config.UIUC_DIR.rglob(row["source_file"])))
        rebuilt = airfoil_from_cst(row, name=row["airfoil_id"]).coordinates

        x0, x1 = original[:, 0].min(), original[:, 0].max()
        original = (original - [x0, 0]) / (x1 - x0)
        ax.plot(original[:, 0], original[:, 1], "k.", markersize=3, label="original points")
        ax.plot(rebuilt[:, 0], rebuilt[:, 1], "r-", linewidth=1.2, label="CST rebuild")
        ax.set_title(f"{row['name']} ({row['airfoil_id']}) - max thickness {row['max_thickness']:.3f}")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.SHAPE_CHECK_PNG, dpi=150)
    print(f"Saved {config.SHAPE_CHECK_PNG}")

def main():
    parser = argparse.ArgumentParser(description="Download UIUC airfoils and convert them to CST numbers.")
    parser.add_argument("--plot", nargs="+", metavar="NAME", help="airfoil names to draw, e.g. naca2412 e387")
    args = parser.parse_args()

    config.ensure_dirs()
    if not config.AIRFOILS_CSV.exists() or not args.plot:
        download_uiuc_zip()
        unzip_uiuc()
        convert_all()
    if args.plot:
        plot_shapes(args.plot)

if __name__ == "__main__":
    main()
