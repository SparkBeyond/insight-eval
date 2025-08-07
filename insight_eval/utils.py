from __future__ import annotations

from insight_eval.core_classes.performance_evaluation_results import PerformanceEvaluationResults
from insight_eval.core_classes.coverage_evaluation_results import CoverageEvaluationResults
from insight_eval.core_classes.combined_evaluation_score import CombinedEvaluationScore
from insight_eval.logging_config import loggers
from scipy.stats import spearmanr
from typing import Callable, Any
from pandas import Series
import numpy as np



def spearman_safe(x: list[float], y: list[float]) -> tuple[float, float]:
    is_constant: Callable[[Any], bool] = lambda arr: np.nanstd(arr) == 0

    # Convert to numpy arrays for easy handling
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if is_constant(x_array) and is_constant(y_array):
        return 1.0, 0.0
    elif is_constant(x_array) or is_constant(y_array):
        return 0.0, 1.0

    rho, p_value = spearmanr(x_array, y_array, nan_policy='omit')
    return float(rho), float(p_value)


def unique_preserving_order(input_list: list[str] | Series[Any]) -> list[Any]:
    # The unique_preserving_order function takes a list and returns a new list with unique elements
    # while preserving the original order.
    ret = []
    seen = set()
    for elem in input_list:
        if elem not in seen:
            seen.add(elem)
            ret.append(elem)
    return ret


def report_evaluation_results(evaluation_results: PerformanceEvaluationResults | CoverageEvaluationResults | CombinedEvaluationScore, error_message: str) -> None:
    match evaluation_results:
        case PerformanceEvaluationResults():
            loggers.eval_logger.info(evaluation_results.summary())
        case CoverageEvaluationResults():
            loggers.eval_logger.info(evaluation_results.summary(names=False))
        case CombinedEvaluationScore():
            loggers.eval_logger.info(evaluation_results.summary())
        case None:
            loggers.eval_logger.error(error_message)
