from insight_eval.readers.read_curriculum_problems_and_solution import read_problem_and_ground_truth_from_folder, read_solution
from insight_eval.core_classes.performance_evaluation_results import PerformanceEvaluationResults
from insight_eval.core_classes.coverage_evaluation_results import CoverageEvaluationResults
from insight_eval.evaluation_framework.evaluators import flow_logger, evaluate, eval_logger
from insight_eval.readers.file_utils import list_directory_items, is_directory_empty
from insight_eval.core_classes.evaluation_results import EvaluationResults
from insight_eval.core_classes.combined_evaluation_score import CombinedEvaluationScore
from insight_eval.readers.json_utils import read_schema
from insight_eval.core_classes.solution import Solution
from insight_eval.core_classes.problem import Problem
import insight_eval.config as config
from traceback import format_exc
from typing import Optional
from pathlib import Path
import shutil
import attrs
import os


@attrs.define(eq=False, hash=False)
class EvaluationSummary:
    problem_evaluation_results: dict[str, tuple[Optional[Solution], Optional[EvaluationResults]]] = dict()
    n_problems: int = 0
    n_solutions: int = 0
    n_evaluations: int = 0


def run_evaluate(solution_path: Path, problems_path: Path, output_folder: Path) -> EvaluationSummary:
    flow_logger.info(f'Running evaluation for solutions in {solution_path}. Problems in {problems_path}, outputting to {output_folder}')
    solution, evaluation_results, performance_eval, coverage_eval, target_leak_eval, combined = None, None, None, None, None, None
    problem_evaluate_res: dict[str, tuple[Optional[Solution], Optional[EvaluationResults]]] = {}
    # go over all problems in folder_fore_results dir, each folder is problem name
    os.makedirs(output_folder, exist_ok=True)
    n_problems = 0
    n_solutions = 0
    n_evaluations = 0
    for problem_dir in list_directory_items(problems_path, dirs_only=True):
        n_problems += 1
        problem_name = problem_dir.name
        flow_logger.info(f'--- *** run_evaluate: start processing - problem {problem_name}  *** ---')
        try:  # recover and continue to next problem_dir
            if os.path.exists(problems_path / problem_name / 'problem'):
                problem, ground_truth = read_problem_and_ground_truth_from_folder(problem_name, problems_path)

                if Path.exists(solution_path / problem_name) and not is_directory_empty(solution_path / problem_name):
                    solution = read_solution(solution_path / problem_name, problem)
                else:
                    continue  # no solution for this problem, skip it

                if os.path.exists(output_folder / problem_name / 'performance.json'):
                    flow_logger.info(f'--- *** run_evaluate: already evaluated - problem {problem_name}  *** ---')
                    evaluation_results = load_evaluations(output_folder / problem_name)
                    performance_eval = evaluation_results.performance_evaluation_results
                    coverage_eval = evaluation_results.coverage_evaluation_results
                    combined = evaluation_results.combined_score
                else:
                    evaluation_results = evaluate_and_save_results(problem, solution, ground_truth, solution_path / problem_name, output_folder / problem_name)
                    performance_eval = evaluation_results.performance_evaluation_results
                    coverage_eval = evaluation_results.coverage_evaluation_results
                    combined = evaluation_results.combined_score

                    # the below is for convenience, so the evaluation results are self-contained with the problem and solution
                    if solution and Path.exists(solution_path / problem_name / 'solution_attributes.json'):
                        shutil.copy(problems_path / problem_name / 'problem' / config.JsonFileNames.problem_json_file_name(), output_folder / problem_name / config.JsonFileNames.problem_json_file_name())
                        shutil.copy(solution_path / problem_name / 'solution_attributes.json', output_folder / problem_name / 'solution_attributes.json')
                    else:
                        flow_logger.info(f'No solution generated for problem {problem_name}')
        except Exception as e:
            evaluation_results = None
            error_message = format_exc()
            eval_logger.error(f'Error evaluating problem {problem_name}: {e},\nerror message: {error_message}')

        n_solutions += (1 if solution else 0)
        n_evaluations += (1 if performance_eval and coverage_eval and combined else 0)
        problem_evaluate_res[problem_name] = solution, evaluation_results

    return EvaluationSummary(
        problem_evaluation_results=problem_evaluate_res,
        n_problems=n_problems,
        n_solutions=n_solutions,
        n_evaluations=n_evaluations
    )


def load_evaluations(evaluations_folder_path: Path) -> EvaluationResults:
    performance_eval = PerformanceEvaluationResults.from_directory(evaluations_folder_path)
    gt_performance_eval = PerformanceEvaluationResults.from_directory(evaluations_folder_path, name_prefix='ground_truth_')
    coverage_eval = CoverageEvaluationResults.from_directory(evaluations_folder_path)
    gt_coverage_eval = CoverageEvaluationResults.from_directory(evaluations_folder_path, name_prefix='ground_truth_')
    combined = read_schema(Path(os.path.join(evaluations_folder_path, 'combined_score.json')), CombinedEvaluationScore)

    if any(item is None for item in [performance_eval, gt_performance_eval, coverage_eval, gt_coverage_eval, combined]):
        raise RuntimeError('Error loading evaluation results')

    # types here ignored since from_directory returns None as well, but we raise an exception in such cases
    target_leak_eval = combined.target_leak
    evaluation_results = EvaluationResults(
        performance_evaluation_results=performance_eval,                            # type: ignore
        ground_truth_performance_evaluation_results=gt_performance_eval,            # type: ignore
        coverage_evaluation_results=coverage_eval,                                  # type: ignore
        ground_truth_coverage_evaluation_results=gt_coverage_eval,
        combined_score=combined,
        target_leak_evaluation_results=target_leak_eval
    )

    return evaluation_results


def evaluate_and_save_results(
        problem: Problem,
        solution: Solution,
        ground_truth: Solution,
        solution_path: Path,
        output_folder: Path) -> EvaluationResults:
    """
    Evaluate a solution against a problem and its ground truth, saving the results to the specified output folder.
    """
    flow_logger.info(f'evaluate_and_save_results: \nproblem_name: {problem.name} \nsolution_path: {solution_path} \nground_truth_name: {ground_truth.name()} \n  output_folder: {output_folder}\n\n')
    flow_logger.info('Calling evaluate')
    evaluation_results = evaluate(problem, solution, ground_truth, solution_path / problem.name)
    flow_logger.info('Evaluation completed')
    #
    output_folder.mkdir(parents=True, exist_ok=True)

    evaluation_results.to_directory(output_folder)

    flow_logger.info(f'Done - See results under {output_folder}')

    return evaluation_results
