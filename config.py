from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

OBS_LEN = 8          # 2 sec at 4 Hz
PRED_LEN = 12        # 3 sec at 4 Hz
INPUT_DIM = 4        # x, y, dx, dy
NUM_MODES = 3        # 3 future trajectories
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LR = 1e-3
EPOCHS = 20
SEED = 42