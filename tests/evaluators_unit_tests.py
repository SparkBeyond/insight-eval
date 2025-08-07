from insight_eval.evaluation_framework.evaluators import measure_rf_auc, measure_performance, correlation_coverage, incremental_performance_coverage, predictive_coverage
from insight_eval.evaluation_framework.evaluators import prepare_data, single_column_predictive_coverage, evaluate_performance_of_solution
from insight_eval.readers.read_curriculum_problems_and_solution import read_problem_and_ground_truth_from_folder, read_solution
from tests.test_objects import TestObjects, TestObjectsForEvalPerformance, gen_df_for_tests
from insight_eval.core_classes.function import Function
from insight_eval.logging_config import loggers
import insight_eval.config as config
from pathlib import Path
import pandas as pd
import numpy as np
import pytest


def test_measure_performance() -> None:
    test_data_path = Path(__file__).parent / 'data' / 'measure_performance'
    train_df = pd.read_csv(test_data_path / 'solution' / 'enriched_train.csv')
    test_df = pd.read_csv(test_data_path / 'solution' / 'enriched_test.csv')

    gt_train_df = pd.read_csv(test_data_path / 'ground_truth' / 'enriched_train.csv')
    gt_test_df = pd.read_csv(test_data_path / 'ground_truth' / 'enriched_test.csv')

    col1 = 'carrier_route_popularity'
    col2 = 'weather_impact_score_during_planned_shipment_windows'

    auc, _ = measure_performance(
                                train_df=pd.concat([train_df[[col1]], gt_train_df[col2]], axis=1),
                                test_df=pd.concat([test_df[[col1]], gt_test_df[col2]], axis=1),
                                target=col2, fast_mode=False)

    loggers.eval_logger.info(f'test_measure_performance: auc={auc}')

    assert abs(auc - 0.5) < 0.05

    loggers.eval_logger.info('*** test_measure_performance passed successfully! ***')


def test_measure_rf_auc() -> None:
    loggers.eval_logger.info('*** starting test_measure_rf_auc: ***')

    test_objects = TestObjects()
    problem = test_objects.problem
    solution = test_objects.solution
    gt = test_objects.ground_truth

    target_name = problem.target_column
    # first test gt columns against the target
    auc_gt, _ = measure_rf_auc(
        train_df=gt.enriched_train_data[gt.enriched_column_names + [target_name]],
        target=target_name,
        test_df=gt.enriched_test_data[gt.enriched_column_names + [target_name]],
    )

    loggers.eval_logger.info(f'test_measure_rf_auc: auc_gt={auc_gt}')
    assert auc_gt > 0.9

    # now test each gt column against the target
    for col in gt.enriched_column_names:
        auc_col, _ = measure_rf_auc(
            train_df=gt.enriched_train_data[[col, target_name]],
            target=target_name,
            test_df=gt.enriched_test_data[[col, target_name]],
        )
        loggers.eval_logger.info(f'test_measure_rf_auc: auc_gt({col})={auc_col}')
        assert 0.65 < auc_col < 0.75

    # now test solution against the target, note that solution has only one column
    auc_solution, _ = measure_rf_auc(
        train_df=solution.enriched_train_data[solution.enriched_column_names + [target_name]],
        target=target_name,
        test_df=solution.enriched_test_data[solution.enriched_column_names + [target_name]]
    )
    loggers.eval_logger.info(f'test_measure_rf_auc: auc_solution = {auc_solution}')
    assert auc_solution > 0.75

    # now test the solution against each gt column
    for col in gt.enriched_column_names:
        auc_col, _ = measure_rf_auc(
            train_df=pd.concat([solution.enriched_train_data[solution.enriched_column_names], gt.enriched_train_data[[col, target_name]]], axis=1),
            target=target_name,
            test_df=pd.concat([solution.enriched_test_data[solution.enriched_column_names], gt.enriched_test_data[[col, target_name]]], axis=1)
        )
        loggers.eval_logger.info(f'test_measure_rf_auc: auc_solution({col})={auc_col}')
        assert auc_col > 0.75

    loggers.eval_logger.info('*** test_measure_rf_auc passed successfully! ***')


def test_correlation_coverage() -> None:
    loggers.eval_logger.info('*** starting test_correlation_coverage: ***')

    test_objects = TestObjects()
    solution = test_objects.solution
    gt = test_objects.ground_truth

    # first test gt against itself
    mean_gt_correlation_coverage, gt_correlation_coverages = correlation_coverage(
        inclusive_features_df=gt.enriched_train_data[gt.enriched_column_names],
        exclusive_gt_df=gt.enriched_train_data[gt.enriched_column_names],
        column_weights=None
    )

    loggers.eval_logger.info(f'test_correlation_coverage: mean_gt_coverage = {mean_gt_correlation_coverage}')
    loggers.eval_logger.info(f'test_correlation_coverage: gt_coverages = {gt_correlation_coverages}')
    assert mean_gt_correlation_coverage == 1.0
    assert gt_correlation_coverages == {'n_orders': {'n_orders': 1.0}, 'common_category': {'common_category': 1.0}}

    # now test solution against gt
    mean_solution_correlation_coverage_train, solution_correlation_coverages_train = correlation_coverage(
        inclusive_features_df=solution.enriched_train_data[solution.enriched_column_names],
        exclusive_gt_df=gt.enriched_train_data[gt.enriched_column_names],
        column_weights=None
    )

    loggers.eval_logger.info(f'test_correlation_coverage: mean_solution_coverage (train) = {mean_solution_correlation_coverage_train}')
    loggers.eval_logger.info(f'test_correlation_coverage: solution_coverages (train) = {solution_correlation_coverages_train}')
    assert 0.75 < mean_solution_correlation_coverage_train < 0.8
    assert solution_correlation_coverages_train['n_orders'] == {'total_amount_ordered': 1.0}
    assert solution_correlation_coverages_train['common_category']['total_amount_ordered'] < 0.6

    mean_solution_correlation_coverage_test, solution_correlation_coverages_test = correlation_coverage(
        inclusive_features_df=solution.enriched_test_data[solution.enriched_column_names],
        exclusive_gt_df=gt.enriched_test_data[gt.enriched_column_names],
        column_weights=None
    )

    loggers.eval_logger.info(f'test_correlation_coverage: mean_solution_coverage (test) = {mean_solution_correlation_coverage_test}')
    loggers.eval_logger.info(f'test_correlation_coverage: solution_coverages (test) = {solution_correlation_coverages_test}')
    assert abs(mean_solution_correlation_coverage_test - 0.6) < 0.05
    assert abs(solution_correlation_coverages_test['n_orders']['total_amount_ordered'] - 0.3) < 0.05
    assert abs(solution_correlation_coverages_test['common_category']['total_amount_ordered'] - 0.9) < 0.05

    loggers.eval_logger.info('*** test_correlation_coverage passed successfully! ***')


def test_incremental_performance_coverage() -> None:
    loggers.eval_logger.info('*** starting test_incremental_performance_coverage: ***')

    test_objects = TestObjects()
    problem = test_objects.problem
    solution = test_objects.solution
    gt = test_objects.ground_truth

    target_name = problem.target_column
    # first test gt against itself
    min_marginal_coverage, gt_coverages = incremental_performance_coverage(
        exclusive_solution_train_w_target_df=gt.enriched_train_data[gt.enriched_column_names + [target_name]],
        exclusive_solution_test_w_target_df=gt.enriched_test_data[gt.enriched_column_names + [target_name]],
        target_column=target_name,
        exclusive_ground_truth_train_wo_target_df=gt.enriched_train_data[gt.enriched_column_names],
        exclusive_ground_truth_test_wo_target_df=gt.enriched_test_data[gt.enriched_column_names]
    )

    loggers.eval_logger.info(f'test_incremental_performance_coverage: min_incremental_performance_coverage = {min_marginal_coverage}')
    loggers.eval_logger.info(f'test_incremental_performance_coverage: gt_coverages = {gt_coverages}')
    assert min_marginal_coverage == 1.0
    assert gt_coverages == {'n_orders': 1.0, 'common_category': 1.0}

    # now test solution against gt
    min_marginal_solution_coverage, marginal_solution_coverages = incremental_performance_coverage(
        exclusive_solution_train_w_target_df=solution.enriched_train_data[solution.enriched_column_names + [target_name]],
        exclusive_solution_test_w_target_df=solution.enriched_test_data[solution.enriched_column_names + [target_name]],
        target_column=target_name,
        exclusive_ground_truth_train_wo_target_df=gt.enriched_train_data[gt.enriched_column_names],
        exclusive_ground_truth_test_wo_target_df=gt.enriched_test_data[gt.enriched_column_names]
    )

    loggers.eval_logger.info(f'test_incremental_performance_coverage: min_marginal_solution_coverage = {min_marginal_solution_coverage}')
    loggers.eval_logger.info(f'test_incremental_performance_coverage: marginal_solution_coverages = {marginal_solution_coverages}')
    assert 0.7 < min_marginal_solution_coverage < 0.8
    assert marginal_solution_coverages['n_orders'] == 1.0
    assert 0.7 < marginal_solution_coverages['common_category'] < 0.8

    loggers.eval_logger.info('*** test_incremental_performance_coverage passed successfully! ***')


def test_predictive_coverage() -> None:
    loggers.eval_logger.info('*** starting test_predictive_coverage: ***')

    test_objects = TestObjects()
    problem = test_objects.problem
    solution = test_objects.solution
    gt = test_objects.ground_truth

    target_name = problem.target_column
    # first test gt against itself
    mean_predictive_gt_coverage, gt_coverages, _ = predictive_coverage(
        exclusive_solution_train_w_target_df=gt.enriched_train_data[gt.enriched_column_names + [target_name]],
        exclusive_solution_test_w_target_df=gt.enriched_test_data[gt.enriched_column_names + [target_name]],
        target_column=target_name,
        exclusive_ground_truth_train_wo_target_df=gt.enriched_train_data[gt.enriched_column_names],
        exclusive_ground_truth_test_wo_target_df=gt.enriched_test_data[gt.enriched_column_names]
    )

    loggers.eval_logger.info(f'test_predictive_coverage: mean_predictive_gt_coverage = {mean_predictive_gt_coverage}')
    loggers.eval_logger.info(f'test_predictive_coverage: gt_coverages = {gt_coverages}')
    assert mean_predictive_gt_coverage > 0.85
    assert gt_coverages['n_orders'] > 0.85
    assert gt_coverages['common_category'] > 0.85

    # now test solution against gt
    mean_predictive_solution_coverage, predictive_solution_coverages, _ = predictive_coverage(
        exclusive_solution_train_w_target_df=solution.enriched_train_data[solution.enriched_column_names + [target_name]],
        exclusive_solution_test_w_target_df=solution.enriched_test_data[solution.enriched_column_names + [target_name]],
        target_column=target_name,
        exclusive_ground_truth_train_wo_target_df=gt.enriched_train_data[gt.enriched_column_names],
        exclusive_ground_truth_test_wo_target_df=gt.enriched_test_data[gt.enriched_column_names]
    )

    loggers.eval_logger.info(f'test_predictive_coverage: mean_predictive_solution_coverage = {mean_predictive_solution_coverage}')
    loggers.eval_logger.info(f'test_predictive_coverage: predictive_solution_coverages = {predictive_solution_coverages}')
    assert 0.4 < mean_predictive_solution_coverage < 0.5
    assert 0.3 < predictive_solution_coverages['n_orders'] < 0.4
    assert 0.5 < predictive_solution_coverages['common_category'] < 0.6

    loggers.eval_logger.info('*** test_predictive_coverage passed successfully! ***')


def test_single_column_predictive_coverage() -> None:
    loggers.eval_logger.info('*** starting test_single_column_predictive_coverage: ***')

    test_objects = TestObjects()
    problem = test_objects.problem
    solution = test_objects.solution
    gt = test_objects.ground_truth

    target_name = problem.target_column
    # first test gt against itself
    mean_gt_coverage, gt_coverages, _ = single_column_predictive_coverage(
        exclusive_solution_train_w_target_df=gt.enriched_train_data[gt.enriched_column_names + [target_name]],
        exclusive_solution_test_w_target_df=gt.enriched_test_data[gt.enriched_column_names + [target_name]],
        target_column=target_name,
        exclusive_ground_truth_train_wo_target_df=gt.enriched_train_data[gt.enriched_column_names],
        exclusive_ground_truth_test_wo_target_df=gt.enriched_test_data[gt.enriched_column_names]
    )

    loggers.eval_logger.info(f'test_single_column_predictive_coverage: mean_predictive_coverage = {mean_gt_coverage}')
    loggers.eval_logger.info(f'test_single_column_predictive_coverage: gt_coverages = {gt_coverages}')
    assert mean_gt_coverage > 0.9
    assert gt_coverages['n_orders']['n_orders'] > 0.9
    assert gt_coverages['common_category']['common_category'] > 0.9

    # now test solution against gt
    mean_predictive_solution_coverage, predictive_solution_coverages, _ = single_column_predictive_coverage(
        exclusive_solution_train_w_target_df=solution.enriched_train_data[solution.enriched_column_names + [target_name]],
        exclusive_solution_test_w_target_df=solution.enriched_test_data[solution.enriched_column_names + [target_name]],
        target_column=target_name,
        exclusive_ground_truth_train_wo_target_df=gt.enriched_train_data[gt.enriched_column_names],
        exclusive_ground_truth_test_wo_target_df=gt.enriched_test_data[gt.enriched_column_names]
    )

    loggers.eval_logger.info(f'test_single_column_predictive_coverage: mean_predictive_solution_coverage = {mean_predictive_solution_coverage}')
    loggers.eval_logger.info(f'test_single_column_predictive_coverage: predictive_solution_coverages = {predictive_solution_coverages}')
    assert 0.3 < mean_predictive_solution_coverage < 0.4
    assert 0.1 < predictive_solution_coverages['n_orders']['total_amount_ordered'] < 0.15
    assert 0.5 < predictive_solution_coverages['common_category']['total_amount_ordered'] < 0.6

    loggers.eval_logger.info('*** test_single_column_predictive_coverage passed successfully! ***')



@pytest.mark.parametrize("train_size,test_size,max_samples_for_subsampling", [
    (200, 100, 75),
    (200, 100, 150),
    (200, 100, 300),
])

def test_prepare_data(train_size: int, test_size: int, max_samples_for_subsampling: int) -> None:
    loggers.eval_logger.info('*** starting test_prepare_data: ***')

    # Generate train and test data.
    # Columns generated would be: id, num, bool, float, cat1, cat2, target
    train_df = gen_df_for_tests(train_size)
    test_df = gen_df_for_tests(test_size)
    target = 'target'

    x_train, target_train, x_test, target_test = prepare_data(train_df, target, test_df, max_samples_for_subsampling)

    # all columns except target
    input_columns = [col for col in train_df.columns if col != target]

    # all columns except high-categorical
    valid_columns = [
        col for col in input_columns if pd.api.types.is_numeric_dtype(train_df[col]) or
                                        pd.api.types.is_bool_dtype(train_df[col]) or
                                        train_df[col].nunique() <= config.MAX_UNIQUE_VALUES_FOR_CATEGORICAL
    ]

    # all *valid* categorical columns
    categorical_columns = [
        col for col in input_columns
        if (not pd.api.types.is_numeric_dtype(train_df[col]))
        and (not pd.api.types.is_bool_dtype(train_df[col]))
        and (train_df[col].nunique() <= config.MAX_UNIQUE_VALUES_FOR_CATEGORICAL)
    ]

    # all high-categorical columns
    high_categorical_columns = [
        col for col in input_columns
        if (not pd.api.types.is_numeric_dtype(train_df[col]))
        and (not pd.api.types.is_bool_dtype(train_df[col]))
        and (train_df[col].nunique() > config.MAX_UNIQUE_VALUES_FOR_CATEGORICAL)
    ]

    # Ensure the outputs have the correct numbers of rows
    assert len(x_train) == min(len(train_df), max_samples_for_subsampling)
    assert len(x_test) == min(len(test_df), max_samples_for_subsampling)
    assert len(target_train) == min(len(train_df), max_samples_for_subsampling)
    assert len(target_test) == min(len(test_df), max_samples_for_subsampling)

    assert list(x_train.columns) == list(x_test.columns)

    assert target not in x_train.columns
    assert target not in x_test.columns

    # Ensure each valid non-categorical column is in x_train and x_test
    for col in valid_columns:
        if col not in categorical_columns:
            assert col in x_train.columns, f"Non-categorical column '{col}' missing in x_train"
            assert col in x_test.columns, f"Non-categorical column '{col}' missing in x_test"

    # verify that high-categorical columns are excluded from the output
    assert len(set(high_categorical_columns).intersection(x_train.columns)) == 0
    assert len(set(high_categorical_columns).intersection(x_test.columns)) == 0

    # verify that for each valid categorical column with k unique values:
    for categorical_col in categorical_columns:
        # 1. there same number of unique values appears in the train and test data
        unique_vals = train_df[categorical_col].unique()
        num_unique = train_df[categorical_col].nunique()
        assert num_unique == test_df[categorical_col].nunique()
        # 2. there are n corresponding one-hot columns in the output.
        count = sum(col.startswith(categorical_col+'_') for col in x_train.columns)
        assert count == num_unique

        category_cols = [col for col in x_train.columns if col.startswith(categorical_col+'_')]

        # Generate expected one-hot data

        expected_one_hot_list_train = [(train_df[categorical_col] == val) for val in unique_vals]
        expected_one_hot_train = pd.DataFrame(
            data={f"{categorical_col}_{val}": col for val, col in zip(unique_vals, expected_one_hot_list_train)})
        expected_one_hot_train = expected_one_hot_train.loc[x_train.index]

        expected_one_hot_list_test = [(train_df[categorical_col] == val) for val in unique_vals]
        expected_one_hot_test = pd.DataFrame(
            data={f"{categorical_col}_{val}": col for val, col in zip(unique_vals, expected_one_hot_list_test)})
        expected_one_hot_test = expected_one_hot_test.loc[x_test.index]

        # Ensure 1-hot output is as expected (up to column permutation)
        assert x_train[category_cols].sort_index(axis=1).equals(expected_one_hot_train.sort_index(axis=1)), \
            f"One-hot mismatch in train for {categorical_col}"

        assert x_test[category_cols].sort_index(axis=1).equals(expected_one_hot_test.sort_index(axis=1)), \
            f"One-hot mismatch in test for {categorical_col}"

    loggers.eval_logger.info('*** test_prepare_data passed successfully! ***')


@pytest.mark.parametrize("train_size,test_size,noise_strength", [
    (10000, 5000, 0.3),
    (10000, 3000, 0.2)
])
def test_evaluate_performance_of_solution(train_size: int, test_size: int, noise_strength: float) -> None:

    loggers.eval_logger.info('*** starting test_evaluate_performance_of_solution: ***')

    # noise_strength is relative to signal_strength=1
    # Note: For some reason, the obtained r^2 and expected r^2 start to diverge as noise_strength increases towards 1.
    #       Keep noise_strength <= 0.3

    assert noise_strength <= 0.3

    obj = TestObjectsForEvalPerformance(train_size=train_size, test_size=test_size, noise_strength=noise_strength)

    perf_eval = evaluate_performance_of_solution(solution=obj.solution)

    solution_r2 = perf_eval.exclusive_performance
    ground_truth_r2 = obj.ground_truth_r2

    thresh = 0.1
    assert np.abs(solution_r2-ground_truth_r2) <= thresh, f'fGround-truth r^2 ({solution_r2}) too different from expected r^2 with perfect knowledge ({ground_truth_r2})'
    # print(f'Exclusive performance: {solution_r2} Perfect r2: {ground_truth_r2}')

    loggers.eval_logger.info('*** test_evaluate_performance_of_solution passed successfully! ***')


def test_function_constructor():
    loggers.eval_logger.info('*** starting test_function_constructor: ***')

    # note: the code of the functions generated within the solution is referring to aux_data, not secondary_data
    code_str = "\n\ndef total_purchase_amount(row, aux_data: dict[str, pd.DataFrame]):\n    # Access the purchase history dataframe from aux_data\n    purchase_history_df = aux_data['purchase_history_table.csv']\n    \n    # Filter the purchase history for the given customer_id\n    customer_purchases = purchase_history_df[purchase_history_df['customer_id'] == row['customer_id']]\n    \n    # Return the sum of the 'purchase_amount' column\n    return customer_purchases['purchase_amount'].sum()\n"
    name_str = 'total_purchase_amount'
    f = Function.from_code_str(name_str, code_str, generic_boilerplate=Function.generic_boilerplate)
    assert isinstance(f, Function), 'Function constructor failed to create a Function object'

    loggers.eval_logger.info('*** test_function_constructor passed successfully! ***')


def test_target_leak_evaluation():
    loggers.eval_logger.info('*** starting test_target_leak_evaluation: ***')

    test_data_path = Path(__file__).parent / 'data'
    problem_path = test_data_path / 'Problem_1'
    solution_path = test_data_path / 'Solution_1'

    from insight_eval.evaluation_framework.evaluate_target_leak import evaluate_target_leak

    problem, ground_truth = read_problem_and_ground_truth_from_folder(problem_path.name, problem_path.parent)
    solution = read_solution(solution_path, problem)

    flag = evaluate_target_leak(solution)

    assert flag == False, f'test_target_leak_evaluation: target leak evaluation failed, flag={flag}'

    loggers.eval_logger.info('*** test_target_leak_evaluation passed successfully! ***')


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    loggers.set_logging_directory(Path(__file__).parent / 'loggers', delete_existing_results=True)
    loggers.eval_logger.info('*** starting unit tests ***')


if __name__ == '__main__':
    loggers.set_logging_directory(Path(__file__).parent / 'loggers', delete_existing_results=True)
    loggers.eval_logger.info('*** starting unit tests ***')

    test_target_leak_evaluation()
    test_measure_performance()

    test_measure_rf_auc()
    test_correlation_coverage()
    test_incremental_performance_coverage()
    test_predictive_coverage()
    test_single_column_predictive_coverage()
    test_prepare_data(train_size=200, test_size=100, max_samples_for_subsampling=75)
    test_prepare_data(train_size=200, test_size=100, max_samples_for_subsampling=150)
    test_prepare_data(train_size=200, test_size=100, max_samples_for_subsampling=300)

    test_evaluate_performance_of_solution(train_size=10000, test_size=5000, noise_strength=0.3)
    test_evaluate_performance_of_solution(train_size=10000, test_size=3000, noise_strength=0.2)

    test_function_constructor()
    loggers.eval_logger.info('*** unit tests ended successfully ***')
