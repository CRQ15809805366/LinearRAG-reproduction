"""项目内运行数据的统一路径定义。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

DATASETS_DIR = INPUT_DIR / "datasets"
DERIVED_CORPORA_DIR = INPUT_DIR / "derived_corpora"
MODELS_DIR = INPUT_DIR / "models"
EXAMPLES_DIR = INPUT_DIR / "examples"

CACHE_DIR = DATA_DIR / "cache"
LINEARRAG_CACHE_DIR = CACHE_DIR / "linearrag"
DATASET_CACHE_DIR = LINEARRAG_CACHE_DIR / "datasets"
DERIVED_CACHE_DIR = LINEARRAG_CACHE_DIR / "derived_corpora"
EMBEDDING_CACHE_DIR = LINEARRAG_CACHE_DIR / "embedding_models"
HIPPORAG_CACHE_DIR = CACHE_DIR / "hipporag"
EXPERIMENT_RESULTS_DIR = OUTPUT_DIR
SMOKE_OUTPUT_DIR = OUTPUT_DIR / "smoke"
