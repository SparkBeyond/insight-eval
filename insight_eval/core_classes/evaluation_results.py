from insight_eval.core_classes.performance_evaluation_results import PerformanceEvaluationResults
from insight_eval.core_classes.coverage_evaluation_results import CoverageEvaluationResults
from insight_eval.core_classes.combined_evaluation_score import CombinedEvaluationScore
from typing import Optional
from pathlib import Path
import attrs


@attrs.define(eq=False, hash=False)
class EvaluationResults:
    performance_evaluation_results: PerformanceEvaluationResults
    ground_truth_performance_evaluation_results: PerformanceEvaluationResults
    coverage_evaluation_results: CoverageEvaluationResults
    ground_truth_coverage_evaluation_results: Optional[CoverageEvaluationResults]
    target_leak_evaluation_results: bool
    combined_score: CombinedEvaluationScore

    def to_directory(self, folder: Path) -> None:
        self.performance_evaluation_results.to_directory(folder)
        self.ground_truth_performance_evaluation_results.to_directory(folder, name_prefix='ground_truth_')
        self.coverage_evaluation_results.to_directory(folder)
        if self.ground_truth_coverage_evaluation_results is not None:
            self.ground_truth_coverage_evaluation_results.to_directory(folder, 'ground_truth_')
        self.combined_score.to_directory(folder)
