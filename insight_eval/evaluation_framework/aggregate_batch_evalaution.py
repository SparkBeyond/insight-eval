from insight_eval.core_classes.coverage_evaluation_results import CoverageEvaluationResults, DetailedCoveragesDictionary, CoveragesDictionary
from insight_eval.core_classes.performance_evaluation_results import PerformanceEvaluationResults
from insight_eval.evaluation_framework.batch_evaluate import run_evaluate, EvaluationSummary
from insight_eval.core_classes.combined_evaluation_score import CombinedEvaluationScore
from insight_eval.readers.file_utils import list_directory_items
from insight_eval.readers.json_utils import dict_from_json_file
from insight_eval.logging_config import loggers
from traceback import format_exc
from insight_eval import converter
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import shutil
import os


reports_logger = loggers.reports_logger
flow_logger = loggers.flow_logger


def aggregate_solution_evaluations(
        eval_dirs: list[Path], problems_dir: Path, output_path:
        Path, agent_short_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:

    flow_logger.info("Starting aggregate_solution_evaluations:")

    if len(eval_dirs) == 0:
        return pd.DataFrame(), dict()

    stats_list = []
    for eval_dir in eval_dirs:
        problem_name = eval_dir.name
        try:  # recover on error and continue to next dir
            flow_logger.info(f'*** ---> new dir: {problem_name}, eval_dir={eval_dir.parent} <--- ***')
            if eval_dir.name.startswith('.DS_Store'):
                continue
            coverage_raw = dict_from_json_file(eval_dir / 'coverage.json')
            performance_raw = dict_from_json_file(eval_dir / 'performance.json')
            solution = dict_from_json_file(eval_dir / 'solution_attributes.json')
            ground_truth_performance_raw = dict_from_json_file(eval_dir / 'ground_truth_performance.json')
            combined_raw = dict_from_json_file(eval_dir / 'combined_score.json')

            coverage: CoverageEvaluationResults = converter.structure(coverage_raw, CoverageEvaluationResults)
            ground_truth_performance = converter.structure(ground_truth_performance_raw, PerformanceEvaluationResults)
            performance = converter.structure(performance_raw, PerformanceEvaluationResults)
            combined = converter.structure(combined_raw, CombinedEvaluationScore)

            agent_name = ''
            if solution:
                solved_by = solution['solved_by']
                agent_name = solved_by

                correlation_coverage: float | None = coverage.mean_correlation_coverage
                correlation_coverages: DetailedCoveragesDictionary = coverage.correlation_coverages
                correlation_coverage_weights: CoveragesDictionary = coverage.correlation_coverage_weights
                incremental_performance_coverage: float | None = coverage.min_incremental_performance_coverage
                incremental_performance_coverages: CoveragesDictionary = coverage.incremental_performance_coverages
                predictive_coverage: float | None = coverage.mean_predictive_coverage
                predictive_coverages: CoveragesDictionary = coverage.predictive_coverages
                predictive_coverage_weights: CoveragesDictionary = coverage.predictive_coverage_weights
                single_column_predictive_coverage: float | None = coverage.mean_single_column_predictive_coverage
                single_column_predictive_coverages: DetailedCoveragesDictionary = coverage.single_column_predictive_coverages
                single_column_predictive_coverage_weights: CoveragesDictionary = coverage.single_column_predictive_coverage_weights
                coverage_score: float | None = combined.coverage
                combined_score: float | None = combined.combined_score
                target_leak_indicator: bool = combined.target_leak

                solution_inclusive_performance_value: float = performance.inclusive_performance
                solution_exclusive_performance_value: float = performance.exclusive_performance
                naive_performance_value: float = performance.naive_performance
                ground_truth_inclusive_performance_value: float = ground_truth_performance.inclusive_performance
                ground_truth_exclusive_performance_value: float = ground_truth_performance.exclusive_performance

                problem_row = {
                    f'problem_name_{agent_short_name}' : problem_name,
                    f'agent_{agent_short_name}': agent_name,
                    f'status_{agent_short_name}': "completed",
                    f'combined_score_{agent_short_name}': combined_score,
                    f'inclusive_performance_{agent_short_name}': solution_inclusive_performance_value,
                    f'exclusive_performance_{agent_short_name}': solution_exclusive_performance_value,
                    f'naive_performance_{agent_short_name}': naive_performance_value,
                    f'coverage_score_{agent_short_name}': coverage_score,
                    f'mean_correlation_coverage_{agent_short_name}': correlation_coverage,
                    f'min_incremental_performance_coverage_{agent_short_name}': incremental_performance_coverage,
                    f'mean_predictive_coverage_{agent_short_name}': predictive_coverage,
                    f'mean_single_column_predictive_coverage_{agent_short_name}': single_column_predictive_coverage,
                    f'ground_truth_inclusive_performance_{agent_short_name}': ground_truth_inclusive_performance_value,
                    f'ground_truth_exclusive_performance_{agent_short_name}': ground_truth_exclusive_performance_value,
                    f'correlation_coverages_{agent_short_name}': correlation_coverages,
                    f'correlation_coverage_weights_{agent_short_name}': correlation_coverage_weights,
                    f'incremental_performance_coverages_{agent_short_name}': incremental_performance_coverages,
                    f'predictive_coverages_{agent_short_name}': predictive_coverages,
                    f'predictive_coverage_weights_{agent_short_name}': predictive_coverage_weights,
                    f'single_column_predictive_coverages_{agent_short_name}': single_column_predictive_coverages,
                    f'single_column_predictive_coverage_weights_{agent_short_name}': single_column_predictive_coverage_weights,
                    f'target_leak_indicator_{agent_short_name}' : target_leak_indicator
                }
            else:  # no solution
                problem_row = {
                    f'problem_name_{agent_short_name}': problem_name,
                    f'agent_{agent_short_name}': agent_name,
                    f'status_{agent_short_name}': "failed",
                    f'combined_score_{agent_short_name}': '',
                    f'inclusive_performance_{agent_short_name}': '',
                    f'exclusive_performance_{agent_short_name}': '',
                    f'naive_performance_{agent_short_name}': '',
                    f'coverage_score_{agent_short_name}': '',
                    f'mean_correlation_coverage_{agent_short_name}': '',
                    f'min_incremental_performance_coverage_{agent_short_name}': '',
                    f'mean_predictive_coverage_{agent_short_name}': '',
                    f'mean_single_column_predictive_coverage_{agent_short_name}': '',
                    f'ground_truth_inclusive_performance_{agent_short_name}': '',
                    f'ground_truth_exclusive_performance_{agent_short_name}': '',
                    f'correlation_coverages_{agent_short_name}': '',
                    f'correlation_coverage_weights_{agent_short_name}': '',
                    f'incremental_performance_coverages_{agent_short_name}': '',
                    f'predictive_coverages_{agent_short_name}': '',
                    f'predictive_coverage_weights_{agent_short_name}': '',
                    f'single_column_predictive_coverages_{agent_short_name}': '',
                    f'single_column_predictive_coverage_weights_{agent_short_name}': '',
                    f'target_leak_indicator_{agent_short_name}': ''
                }
            stats_list.append(problem_row)
        except:  # noQA
            error_message = format_exc()
            reports_logger.error(f"Error in problem {problem_name}: \n{error_message}")

    def positives_average(s: pd.Series) -> np.floating[Any] | float:
        num_s = [x for x in s if (type(x) == int or type(x) == float or type(x) == bool) and x >= 0 and (not np.isnan(x))]  # note that x>=0 for nan is False, so this will also remove nan's  # noQA
        if len(num_s) > 0:
            return np.average(num_s)

        return np.nan

    if stats_list:
        temp_df = pd.DataFrame(stats_list)
        average_row = {
            'problem_name_' + agent_short_name: "average",
            f'agent_{agent_short_name}': agent_short_name,
            f'status_{agent_short_name}': "",
            f'combined_score_{agent_short_name}': positives_average(temp_df['combined_score_' + agent_short_name]),
            f'inclusive_performance_{agent_short_name}': positives_average(temp_df['inclusive_performance_' + agent_short_name]),
            f'exclusive_performance_{agent_short_name}': positives_average(temp_df['exclusive_performance_' + agent_short_name]),
            f'naive_performance_{agent_short_name}': positives_average(temp_df['naive_performance_' + agent_short_name]),
            f'coverage_score_{agent_short_name}': positives_average(temp_df['coverage_score_' + agent_short_name]),
            f'mean_correlation_coverage_{agent_short_name}':  positives_average(temp_df['mean_correlation_coverage_' + agent_short_name]),
            f'min_incremental_performance_coverage_{agent_short_name}': positives_average(temp_df['min_incremental_performance_coverage_' + agent_short_name]),
            f'mean_predictive_coverage_{agent_short_name}': positives_average(temp_df['mean_predictive_coverage_' + agent_short_name]),
            f'mean_single_column_predictive_coverage_{agent_short_name}': positives_average(temp_df['mean_single_column_predictive_coverage_' + agent_short_name]),
            f'ground_truth_inclusive_performance_{agent_short_name}': positives_average(temp_df['ground_truth_inclusive_performance_' + agent_short_name]),
            f'ground_truth_exclusive_performance_{agent_short_name}': positives_average(temp_df['ground_truth_exclusive_performance_' + agent_short_name]),
            f'target_leak_indicator_{agent_short_name}' : positives_average(temp_df['target_leak_indicator_' + agent_short_name])
        }

        stats_list.append(average_row)
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(output_path / ('eval_' + agent_short_name + '.csv'), index=False)
        return stats_df, average_row
    else:
        reports_logger.error('*** aggregate_solution_evaluation: Unable to create evaluations for any of the solutions !')
        reports_logger.error('*** Abort ***')
        raise ValueError('aggregate_solution_evaluation: Aborted - Unable to create evaluations for any of the solutions !')


def generate_agent_aggregate_report(agent_name: str, problems_folder: Path, agent_evaluations_folder: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    flow_logger.info(f'*** *** Generating aggregate report for agent: {agent_name} problems_folder: {problems_folder} *** ***')
    eval_dirs = list_directory_items(agent_evaluations_folder, dirs_only=True)

    stats_df, average_row = aggregate_solution_evaluations(eval_dirs, problems_folder, agent_evaluations_folder, agent_name)

    generic_columns = [
        'problem_name', 'combined_score', 'inclusive_performance', 'exclusive_performance', 'naive_performance',
        'coverage_score', 'mean_correlation_coverage', 'min_incremental_performance_coverage', 'mean_predictive_coverage',
        'mean_single_column_predictive_coverage', 'ground_truth_inclusive_performance',
        'ground_truth_exclusive_performance', 'correlation_coverages', 'incremental_performance_coverages',
        'predictive_coverages', 'predictive_coverage_weights', 'single_column_predictive_coverages',
        'single_column_predictive_coverage_weights', 'target_leak_indicator'
    ]

    columns = [c + f'_{agent_name}' for c in generic_columns]

    if len(stats_df.columns) == 0 or len(stats_df) == 0:
        reports_logger.warning(f'Stats_df shape is {stats_df.shape}, attempting to generate report for {agent_name}')
        reports_logger.warning('Skipping ')
        return pd.DataFrame(), dict()

    stats_df = stats_df[columns]
    stats_df.to_csv(agent_evaluations_folder / f'eval_{agent_name}.csv', index=False)

    clean_average_row = {k: v for k, v in average_row.items() if k not in ['problem_name_' + agent_name, 'status_' + agent_name]}
    return stats_df, clean_average_row


def evaluate_agent_on_folder(
        agent_name: str, problems_path: Path, solutions_path: Path, evaluations_path: Path,
        delete_existing_evaluations: bool = False) -> tuple[EvaluationSummary, tuple[pd.DataFrame, dict[str, Any]]]:

    if not Path.exists(solutions_path):
        reports_logger.error(f'Solutions folder {solutions_path} does not exist')
        raise RuntimeError(f'Solutions folder {solutions_path} does not exist')

    os.makedirs(evaluations_path, exist_ok=True)
    if delete_existing_evaluations:
        shutil.rmtree(evaluations_path, ignore_errors=True)  # remove previous evaluations if exist

    evaluation_summary: EvaluationSummary = run_evaluate(solutions_path, problems_path, evaluations_path)
    report_results = generate_agent_aggregate_report(agent_name, problems_path, evaluations_path)
    return evaluation_summary, report_results