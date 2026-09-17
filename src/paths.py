"""项目内运行数据的统一路径定义。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

DATASETS_DIR = INPUT_DIR / "datasets"
MODELS_DIR = INPUT_DIR / "models"
EXAMPLES_DIR = INPUT_DIR / "examples"

CACHE_DIR = OUTPUT_DIR / "cache"
RUNS_DIR = OUTPUT_DIR / "runs"
SMOKE_OUTPUT_DIR = OUTPUT_DIR / "smoke"
