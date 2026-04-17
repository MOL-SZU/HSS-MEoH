from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_DIR.parent
CORE_DIR = WORKSPACE_DIR / 'core'
SCRIPTS_DIR = WORKSPACE_DIR / 'scripts'
PLOTS_DIR = WORKSPACE_DIR / 'plots'
RESULTS_DIR = WORKSPACE_DIR / 'results'
LOGS_DIR = RESULTS_DIR / 'logs'
IMAGES_DIR = RESULTS_DIR / 'image'
RESULT_CODE_DIR = RESULTS_DIR / 'result_code'
TEST_RESULT_DIR = RESULTS_DIR / 'test_result'
TRAIN_RESULT_DIR = RESULTS_DIR / 'train_result'
DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_DATA_DIR = DATA_DIR / 'train_data'
TEST_DATA_DIR = DATA_DIR / 'test_data'
HSS_BENCHMARK_DIR = PROJECT_ROOT / 'HSS_benchmark'


def ensure_result_dirs() -> None:
    for path in (RESULTS_DIR, LOGS_DIR, IMAGES_DIR, RESULT_CODE_DIR, TEST_RESULT_DIR, TRAIN_RESULT_DIR):
        path.mkdir(parents=True, exist_ok=True)
