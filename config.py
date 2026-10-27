from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
POLARS = RESULTS / "polars"

UIUC_ZIP_URL = "https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip"
UIUC_ZIP_PATH = DATA_RAW / "coord_seligFmt.zip"
UIUC_DIR = DATA_RAW / "coord_seligFmt"
MIN_POINTS = 20
N_CST = 8

XFOIL_COMMAND = "xfoil"

REYNOLDS_NUMBERS = [1e5, 2e5, 5e5, 1e6, 2e6]
ALPHAS = [float(a) for a in range(-4, 15)]
MACH = 0.0
N_CRIT = 9.0
XFOIL_TIMEOUT = 15

XFOIL_MAX_ITER = 100

CL_MAX_ABS = 3.0
CST_MAX_ABS = 10.0
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

RANDOM_SEED = 42

HIDDEN_SIZES = (128, 128, 128)
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 300
PATIENCE = 20

STALL_ALPHA = 8.0
N_POLAR_AIRFOILS = 4

AIRFOILS_CSV = DATA_PROCESSED / "airfoils.csv"
FAILED_CSV = DATA_PROCESSED / "failed_airfoils.csv"
XFOIL_CSV = DATA_PROCESSED / "xfoil_results.csv"
PROGRESS_CSV = DATA_PROCESSED / "xfoil_progress.csv"
SPLIT_CSV = DATA_PROCESSED / "split.csv"

MODEL_PATH = MODELS / "surrogate.pt"
SCALERS_PATH = MODELS / "scalers.json"

SHAPE_CHECK_PNG = RESULTS / "shape_check.png"
DATA_OVERVIEW_PNG = RESULTS / "data_overview.png"
HISTORY_CSV = RESULTS / "training_history.csv"
LOSS_CURVE_PNG = RESULTS / "loss_curve.png"
TRAINING_RUNS_CSV = RESULTS / "training_runs.csv"
METRICS_JSON = RESULTS / "test_metrics.json"
PREDICTIONS_CSV = RESULTS / "test_predictions.csv"
PARITY_CL_PNG = RESULTS / "parity_CL.png"
PARITY_CD_PNG = RESULTS / "parity_CD.png"

def ensure_dirs():
    for folder in (DATA_RAW, DATA_PROCESSED, MODELS, RESULTS, POLARS):
        folder.mkdir(parents=True, exist_ok=True)
