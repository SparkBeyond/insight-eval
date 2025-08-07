from insight_eval.readers.read_curriculum_problems_and_solution import read_problem_and_ground_truth_from_folder, read_solution
from insight_eval.evaluation_framework.batch_evaluate import evaluate_and_save_results
from insight_eval.evaluation_framework.batch_evaluate import load_evaluations
from insight_eval.core_classes.evaluation_results import EvaluationResults
from insight_eval.config import COVERAGE_METRICS_TO_INCLUDE
from insight_eval.utils import report_evaluation_results
from insight_eval.logging_config import loggers
from pathlib import Path
import shutil


def test_partial_evaluation() -> None:
    loggers.set_logging_directory(Path('__file__').parent / 'loggers')

    batch_data_path = Path(__file__).parent / 'data'
    problems_path = batch_data_path
    solutions_path = batch_data_path / 'Solution_3'
    output_folder = batch_data_path / 'evaluations_single_problem'
    problem_name = 'Problem_3'

    shutil.rmtree(output_folder, ignore_errors=True)  # remove previous evaluations if exist

    problem, ground_truth = read_problem_and_ground_truth_from_folder(problem_name, problems_path)
    solution = read_solution(solutions_path, problem)

    COVERAGE_METRICS_TO_INCLUDE['predictive_coverage'] = False
    evaluation_results: EvaluationResults = evaluate_and_save_results(problem, solution, ground_truth, solutions_path, output_folder / problem_name)

    loggers.eval_logger.info(f'Evaluation results of solution for problem: {problem_name}')
    loggers.eval_logger.info(f'Problem path: {problems_path / problem_name}')
    loggers.eval_logger.info(f'Solution path: {solutions_path}')
    loggers.eval_logger.info(f'Output path: {output_folder / problem_name}')

    report_evaluation_results(
        evaluation_results=evaluation_results.performance_evaluation_results,
        error_message='Performance evaluation failed for this problem.'
    )

    report_evaluation_results(
        evaluation_results=evaluation_results.coverage_evaluation_results,
        error_message='Coverage evaluation failed for this problem.',
    )

    report_evaluation_results(
        evaluation_results=evaluation_results.combined_score,
        error_message='Combined score failed for this problem.'
    )

    try:
        load_evaluations(output_folder / problem_name)
    except Exception as e:
        assert False, f'Error loading evaluation results: {e}'
