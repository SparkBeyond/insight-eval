from pathlib import Path

N_RF_ESTIMATORS = 100
RANDOM_STATE = 42
N_SAMPLES = 5000
MAX_FEATURES = 20
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 10
COVERAGE_CORRELATION_THRESHOLD = 0.15
ELIGIBILITY_THRESHOLD_FOR_GT_COVERAGE = 0.0
USE_SPEARMAN_FOR_COVERAGE = True
EVALUATE_GROUND_TRUTH_COVERAGE = True
EVALUATE_TARGET_LEAK = True

COVERAGE_METRICS_TO_INCLUDE = {
    'correlation_coverage': True,
    'incremental_performance_coverage': True,
    'predictive_coverage': True,
    'single_column_predictive_coverage': True
}

COVERAGE_METRIC_WEIGHTS: dict[str, float | int] = {
        'correlation_coverage': 0.0,
        'incremental_performance_coverage': 0.3,
        'predictive_coverage': 0.0,
        'single_column_predictive_coverage': 0.7
}

def create_local_curriculum(lc_path: Path) -> None:
    if not lc_path.exists():
        lc_path.mkdir(parents=True, exist_ok=True)

class JsonFileNames:
    @classmethod
    def problem_json_file_name(cls) -> str:
        return 'problem.json'

    @classmethod
    def solution_json_file_name(cls) -> str:
        return 'solution.json'
