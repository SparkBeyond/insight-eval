from insight_eval.readers.read_curriculum_problems_and_solution import read_problem_and_ground_truth_from_folder, read_solution
from insight_eval.evaluation_framework.evaluate_multiple_agents import evaluate_multiple_agents
from insight_eval.evaluation_framework.evaluators import evaluate_coverage_of_solution
from insight_eval.core_classes.evaluation_results import EvaluationResults
from insight_eval.evaluation_framework.batch_evaluate import load_evaluations
from insight_eval.evaluation_framework.evaluators import evaluate
from insight_eval.logging_config import loggers
import insight_eval.config as config
from pathlib import Path
from typing import Any
import pandas as pd
import dataclasses
import pytest
import attrs
import os


#
# end-to-end tests will be provided for the following functions:
# evaluate()
#    implicitly this will test the results from
#        evaluate_performance_of_solution
#        evaluate_coverage_of_solution
#        evaluate_target_leak

eval_logger = loggers.eval_logger
flow_logger = loggers.flow_logger


def _test_evaluate_single_problem(problem_path: Path, solution_path: Path) -> EvaluationResults:
    eval_logger.info('*** starting end-to-end test = test_evaluate_single_problem: ***')

    problem, ground_truth = read_problem_and_ground_truth_from_folder(problem_path.name, problem_path.parent)
    solution = read_solution(solution_path, problem)

    evaluation_results: EvaluationResults = evaluate(problem, solution, ground_truth, solution_path)

    # validate results - generic asserts
    assert evaluation_results.combined_score.inclusive_performance == evaluation_results.performance_evaluation_results.inclusive_performance
    assert evaluation_results.combined_score.exclusive_performance == evaluation_results.performance_evaluation_results.exclusive_performance

    return evaluation_results


def compare_evaluation_results(ref: EvaluationResults, obj: EvaluationResults, compare_float_threshold: float = 0.002) -> bool:
    assert isinstance(ref, EvaluationResults), "Reference must be an instance of EvaluationResults"
    assert isinstance(obj, EvaluationResults), "Object must be an instance of EvaluationResults"

    def extract_fields(ref: Any, obj: Any, simple_obj_ok: bool = False) -> tuple[dict, dict]:
        if dataclasses.is_dataclass(ref) and dataclasses.is_dataclass(obj):
            ret_ref = dataclasses.asdict(ref)
            ret_obj = dataclasses.asdict(obj)
        elif attrs.has(ref) and attrs.has(obj):
            ret_ref = attrs.asdict(ref)
            ret_obj = attrs.asdict(obj)
        elif isinstance(ref, object) and hasattr(ref, "__dict__") and isinstance(obj, object) and hasattr(obj, "__dict__"):
            ret_ref = ref.__dict__
            ret_obj = obj.__dict__
        elif simple_obj_ok:
            ret_ref = ref
            ret_obj = obj
        else:
            raise ValueError("Both reference and object must be dataclasses or objects with __dict__ attributes, or simple objects if simple_obj_ok is True.")

        return ret_ref, ret_obj
    # end extract_fields

    ref_dict, obj_dict = extract_fields(ref, obj, simple_obj_ok=False)

    equal_results = True
    for key in ref_dict:
        ref_val, obj_val = extract_fields(ref_dict[key], obj_dict.get(key, None), simple_obj_ok=True)
        if isinstance(ref_val, dict) and isinstance(obj_val, dict):
            for subkey in ref_val:
                ref_sub_val = ref_val[subkey]
                obj_sub_val = obj_val.get(subkey, None)
                if ref_sub_val != obj_sub_val:
                    if isinstance(ref_sub_val, float) and isinstance(obj_sub_val, float):
                        if abs(ref_sub_val - obj_sub_val) > compare_float_threshold:
                            equal_results = False
                            eval_logger.error(f"Difference in '{key}':reference={ref_sub_val}, type={type(ref_sub_val)}, object={obj_sub_val}, type={type(obj_sub_val)}: diff={abs(ref_sub_val - obj_sub_val)}")
                    elif type(obj_sub_val) not in [bool, int, float, str] or type(ref_sub_val) not in [bool, int, float, str]:
                        # if one of the values is not a flat type, skip the comparison
                        eval_logger.debug(f"Skipping comparison with non-flat (bool, int, float, str) element in ref/obj object: {type(ref_sub_val)}/{type(obj_sub_val)}")
                        continue
                    else:
                        equal_results = False
                        eval_logger.error(f"Difference in '{key}.{subkey}':reference={ref_sub_val}, type={type(ref_sub_val)}, object={obj_sub_val}, type={type(obj_sub_val)}")
        elif ref_val != obj_val:
            if isinstance(ref_val, float) and isinstance(obj_val, float):
                if abs(ref_val - obj_val) > compare_float_threshold:
                    equal_results = False
                    eval_logger.error(f"Difference in '{key}':reference={ref_val}, type={type(ref_val)}, object={obj_val}, type={type(obj_val)}: diff={abs(ref_val - obj_val)}")
            else:
                equal_results = False
                eval_logger.error(f"Difference in '{key}':reference={ref_val}, type={type(ref_val)}, object={obj_val}, type={type(obj_val)}")
        elif ref_val == obj_val:
            continue
        else:
            equal_results = False
            eval_logger.error(f'Unexpected structure: Evaluation results do not match reference results: ref_val={ref_val}, type={type(ref_val)}, obj_val={obj_val}, type={type(obj_val)}')

    return equal_results


def flat_compare_dataframes(ref_df: pd.DataFrame, obj_df: pd.DataFrame, compare_float_threshold: float = 0.002) -> bool:
    """
    Compare two dataframes for equality, allowing for small floating-point differences.
    Elements in the dataframes may be non-flat (e.g., dicts or attrs classes), in which case they are skipped.
    """
    if ref_df.shape != obj_df.shape:
        eval_logger.error(f"DataFrames have different shapes: {ref_df.shape} vs {obj_df.shape}")
        return False

    flag = True
    for index, row in ref_df.iterrows():
        if index not in obj_df.index:
            eval_logger.error(f"Index {index} not found in object DataFrame")
            return False

        for col in ref_df.columns:
            ref_value = row[col]
            obj_value = obj_df.at[index, col]
            if type(obj_value) not in [bool, int, float, str] or type(ref_value) not in [bool, int, float, str]:
                # if one of the values is not a flat type, skip the comparison
                eval_logger.debug(f"Skipping comparison with non-flat (bool, int, float, str) element in ref/obj dataframe: {type(ref_value)}/{type(obj_value)} in column '{col}' at index {index}")
                continue

            if isinstance(ref_value, (bool, int, float)) and isinstance(obj_value, (bool, int, float)):
                if abs(ref_value - obj_value) > compare_float_threshold:
                    eval_logger.error(f"Difference at index {index}, column '{col}': reference={ref_value}, object={obj_value}")
                    flag = False
            elif isinstance(ref_value, (str, bool)) and isinstance(obj_value, (str, bool)) and str(ref_value) == str(obj_value):
                continue  # ok <True, 'True'> pairs
            elif ref_value != obj_value:
                eval_logger.error(f"Difference at index {index}, column '{col}': reference={ref_value}, object={obj_value}")
                flag = False
            else:
                continue

    return flag


def test_e2e_solution1():
    eval_logger.info('*** starting test_e2e_solution1: ***')

    test_data_path = Path(__file__).parent / 'data'
    problem_path = test_data_path / 'Problem_1'
    solution_path = test_data_path / 'Solution_1'
    reference_evaluations_path = test_data_path / 'Reference_Evaluation_1'

    # ensure all coverage metrics are included
    config.COVERAGE_METRICS_TO_INCLUDE['correlation_coverage'] = True
    config.COVERAGE_METRICS_TO_INCLUDE['incremental_performance_coverage'] = True
    config.COVERAGE_METRICS_TO_INCLUDE['predictive_coverage'] = True
    config.COVERAGE_METRICS_TO_INCLUDE['single_column_predictive_coverage'] = True

    evaluation_results = _test_evaluate_single_problem(problem_path, solution_path)
    assert evaluation_results.combined_score.target_leak == False, 'test_e2e_solution1: Target leak should be False for this problem'
    reference_evaluations_results = load_evaluations(reference_evaluations_path)
    assert reference_evaluations_results.combined_score.target_leak == False, 'test_e2e_solution1: Target leak should be False for the reference evaluation'

    flag = compare_evaluation_results(reference_evaluations_results, evaluation_results)
    assert flag, 'test_e2e_solution1: Evaluation results do not match reference results'

    eval_logger.info('*** test_e2e_solution1 passed successfully! ***')

def test_e2e_solution2():
    eval_logger.info('*** starting test_e2e_solution2: ***')

    test_data_path = Path(__file__).parent / 'data'
    problem_path = test_data_path / 'Problem_2'

    problem, ground_truth = read_problem_and_ground_truth_from_folder(problem_path.name, problem_path.parent)

    for k in config.COVERAGE_METRICS_TO_INCLUDE:
        config.COVERAGE_METRICS_TO_INCLUDE[k] = True

    gt_coverage_eval = evaluate_coverage_of_solution(
        ground_truth, ground_truth.enriched_train_data, ground_truth.enriched_test_data,
        ground_truth.enriched_column_names, ground_truth.name()
    )

    msg = '\n*** the below is NOT an error message ***\n'
    msg += f'Ground truth coverage summary:\n {gt_coverage_eval.summary()}\n'
    msg += f'Incremental_performance_coverages: {gt_coverage_eval.incremental_performance_coverages}\n'
    msg += f'Predictive_coverages: {gt_coverage_eval.predictive_coverages}\n'
    msg += f'Single_column_predictive_coverages: {gt_coverage_eval.single_column_predictive_coverages}\n'

    eval_logger.info(msg)

    if gt_coverage_eval.mean_correlation_coverage > 0.99 and \
            gt_coverage_eval.min_incremental_performance_coverage > 0.99 and \
            gt_coverage_eval.mean_predictive_coverage > 0.99 and \
            gt_coverage_eval.mean_single_column_predictive_coverage > 0.99:
        assert False, 'Ground truth coverage passes the strict thresholds, which is unexpected'
    else:
        eval_logger.warning('GT coverage is not ~1.0 by strict thresholds (this is by design)')

    if gt_coverage_eval.mean_correlation_coverage > 0.99 and \
            gt_coverage_eval.min_incremental_performance_coverage > 0.95 and \
            gt_coverage_eval.mean_predictive_coverage > 0.95 and \
            gt_coverage_eval.mean_single_column_predictive_coverage > 0.99:
        eval_logger.info('GT coverage is ~1.0 by soft thresholds')
    else:
        assert False, 'Ground truth coverage does not pass the soft thresholds, which is unexpected'

    eval_logger.info('*** test_e2e_solution2 passed successfully! ***')

def test_e2e_multiple_agents():
    eval_logger.info('*** starting test_e2e_multiple_agents: ***')

    test_data_path = Path(__file__).parent / 'data' / 'multiple_agents'
    problems_path = test_data_path / 'problems'
    solutions_path = test_data_path / 'agents_solutions'
    evaluations_path = test_data_path / 'agents_evaluations'
    reference_evaluations_path = test_data_path / 'reference_agents_evaluations'
    combined_reports_path = evaluations_path / 'combined_reports'
    reference_combined_reports_path = reference_evaluations_path / 'combined_reports'
    os.makedirs(combined_reports_path, exist_ok=True)

    agents_for_combined_report = ['agent1', 'agent2']

    try:
        agent_averages_df, detailed_eval_df = evaluate_multiple_agents(
            agents_for_combined_report,
            problems_path,
            solutions_path,
            evaluations_path,
            delete_existing_evaluations=True,
            problem_sets_hierarchy=False,
            combined_reports_output_path=combined_reports_path
        )

        if len(detailed_eval_df) == 0:
            assert False, 'test_e2e_multiple_agents: No evaluation results were generated'

        reference_averages_df = pd.read_csv(reference_combined_reports_path / 'agent_averages.csv')
        reference_detailed_eval_df = pd.read_csv(reference_combined_reports_path / 'detailed_eval.csv')
        flow_logger.debug('compare agent_averages_df')
        flag = flat_compare_dataframes(reference_averages_df, agent_averages_df)
        assert flag, 'test_e2e_multiple_agents: reference_averages_df does not match agent_averages_df'

        flow_logger.debug('compare detailed_eval')
        flag = flat_compare_dataframes(reference_detailed_eval_df, detailed_eval_df)
        assert flag, 'test_e2e_multiple_agents: reference_detailed_eval_df does not match detailed_eval_df'
    except Exception as e:
        assert False, f'test_e2e_multiple_agents failed with exception: {e}'

    pass
    eval_logger.info('*** test_e2e_multiple_agents passed successfully! ***')


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    loggers.set_logging_directory(Path(__file__).parent / 'loggers', delete_existing_results=True)
    loggers.eval_logger.info('*** starting end-to-end tests ***')


if __name__ == '__main__':
    loggers.set_logging_directory(Path(__file__).parent / 'loggers', delete_existing_results=True)
    eval_logger.info('*** starting end-to-end tests ***')

    test_e2e_solution1()

    test_e2e_solution2()

    test_e2e_multiple_agents()

    eval_logger.info('*** end-to-end tests ended successfully ***')
