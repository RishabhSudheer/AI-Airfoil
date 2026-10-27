import argparse
import multiprocessing as mp
import time
import warnings
from datetime import date

import pandas as pd

import config
from common import airfoil_from_cst

RESULT_COLUMNS = ["airfoil_id", "Re", "alpha", "CL", "CD", "CM", "mach", "n_crit", "run_date"]
PROGRESS_COLUMNS = ["airfoil_id", "n_attempted", "n_converged", "seconds", "status", "run_date"]

def run_xfoil(airfoil, reynolds, alphas):
    import aerosandbox as asb

    xfoil = asb.XFoil(
        airfoil=airfoil,
        Re=reynolds,
        mach=config.MACH,
        n_crit=config.N_CRIT,
        max_iter=config.XFOIL_MAX_ITER,
        xfoil_command=config.XFOIL_COMMAND,
        timeout=config.XFOIL_TIMEOUT,
    )

    return xfoil.alpha(alphas, start_at=0)

def process_airfoil(row):
    warnings.simplefilter("ignore")
    start = time.time()
    today = date.today().isoformat()
    airfoil_id = row["airfoil_id"]
    results, errors = [], []
    attempted = len(config.REYNOLDS_NUMBERS) * len(config.ALPHAS)

    try:
        airfoil = airfoil_from_cst(row, name=airfoil_id)
    except Exception as error:
        progress = [airfoil_id, attempted, 0, round(time.time() - start, 1), f"error: {error}", today]
        return results, progress

    for reynolds in config.REYNOLDS_NUMBERS:
        try:
            out = run_xfoil(airfoil, reynolds, config.ALPHAS)
        except Exception as error:
            errors.append(f"Re={reynolds:.0e}: {str(error).splitlines()[0][:80]}")
            continue
        for alpha, cl, cd, cm in zip(out["alpha"], out["CL"], out["CD"], out["CM"]):
            results.append([airfoil_id, reynolds, round(float(alpha), 3), float(cl), float(cd),
                            float(cm), config.MACH, config.N_CRIT, today])

    status = "done" if not errors else "done with errors: " + "; ".join(errors)
    progress = [airfoil_id, attempted, len(results), round(time.time() - start, 1), status, today]
    return results, progress

def append_rows(path, columns, rows):
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)

def xfoil_self_test():
    import aerosandbox as asb

    print(f"Testing XFOIL with command: {config.XFOIL_COMMAND!r}")
    airfoil = asb.Airfoil("naca2412")
    try:
        out = run_xfoil(airfoil, 1e6, [0.0, 2.0, 4.0, 6.0, 8.0])
    except FileNotFoundError:
        raise SystemExit("XFOIL not found. Set XFOIL_COMMAND in config.py to the full path of your XFOIL program.")
    except Exception as error:
        raise SystemExit(
            f"XFOIL did not run: {error}\n"
            "Check XFOIL_COMMAND in config.py (use the full path to the program), "
            "and on Linux install xvfb."
        )
    if len(out["alpha"]) == 0:
        raise SystemExit("XFOIL ran but no angles converged. Try running XFOIL by hand (Timeline Task 1.3).")
    print(f"\n{'alpha':>6} {'CL':>8} {'CD':>9} {'CM':>8}")
    for a, cl, cd, cm in zip(out["alpha"], out["CL"], out["CD"], out["CM"]):
        print(f"{a:6.1f} {cl:8.4f} {cd:9.5f} {cm:8.4f}")
    print("\nXFOIL works from Python.")

def main():
    parser = argparse.ArgumentParser(description="Generate CL/CD training data with XFOIL.")
    parser.add_argument("--test", action="store_true", help="only check that XFOIL works")
    parser.add_argument("--limit", type=int, default=None, help="process at most N airfoils this run")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                        help="how many XFOIL copies to run at once (default: CPU cores - 1)")
    args = parser.parse_args()

    config.ensure_dirs()
    if args.test:
        xfoil_self_test()
        return

    if not config.AIRFOILS_CSV.exists():
        raise SystemExit("airfoils.csv not found. Run `python 01_prepare_airfoils.py` first.")
    airfoils = pd.read_csv(config.AIRFOILS_CSV)

    done_ids = set()
    if config.PROGRESS_CSV.exists():
        done_ids = set(pd.read_csv(config.PROGRESS_CSV)["airfoil_id"])
    todo = airfoils[~airfoils["airfoil_id"].isin(done_ids)]
    if args.limit is not None:
        todo = todo.head(args.limit)

    print(f"Airfoils total: {len(airfoils)} | already done: {len(done_ids)} | this run: {len(todo)}")
    print(f"Each airfoil: {len(config.REYNOLDS_NUMBERS)} Reynolds numbers x {len(config.ALPHAS)} angles "
          f"= {len(config.REYNOLDS_NUMBERS) * len(config.ALPHAS)} attempts | workers: {args.workers}")

    if len(todo) > 0:
        rows = todo.to_dict("records")
        start = time.time()
        attempted_total = converged_total = 0
        with mp.Pool(processes=args.workers) as pool:
            for count, (results, progress) in enumerate(pool.imap_unordered(process_airfoil, rows), start=1):
                append_rows(config.XFOIL_CSV, RESULT_COLUMNS, results)
                append_rows(config.PROGRESS_CSV, PROGRESS_COLUMNS, [progress])
                attempted_total += progress[1]
                converged_total += progress[2]
                elapsed = time.time() - start
                per_airfoil = elapsed / count
                eta_min = per_airfoil * (len(rows) - count) / 60
                print(f"[{count}/{len(rows)}] {progress[0]}: {progress[2]}/{progress[1]} converged "
                      f"in {progress[3]:.0f}s | avg {per_airfoil:.1f}s per airfoil | ETA {eta_min:.0f} min")

        print(f"\nThis run: {len(rows)} airfoils in {(time.time() - start) / 60:.1f} min, "
              f"{(time.time() - start) / len(rows):.1f}s per airfoil (wall clock, all workers), "
              f"{100 * converged_total / max(attempted_total, 1):.1f}% of points converged.")

    progress = pd.read_csv(config.PROGRESS_CSV) if config.PROGRESS_CSV.exists() else pd.DataFrame()
    n_rows = sum(1 for _ in open(config.XFOIL_CSV)) - 1 if config.XFOIL_CSV.exists() else 0
    if not progress.empty:
        rate = 100 * progress["n_converged"].sum() / max(progress["n_attempted"].sum(), 1)
        print(f"Overall: {len(progress)}/{len(airfoils)} airfoils done, {n_rows} result rows, "
              f"{rate:.1f}% converged.")
        if len(progress) >= len(airfoils):
            print("All airfoils done")

if __name__ == "__main__":
    main()
