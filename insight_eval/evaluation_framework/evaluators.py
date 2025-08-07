from insight_eval.core_classes.performance_evaluation_results import PerformanceEvaluationResults
from insight_eval.core_classes.coverage_evaluation_results import CoverageEvaluationResults, \
    DetailedCoveragesDictionary, CoveragesDictionary
from insight_eval.evaluation_framework.evaluate_target_leak import evaluate_target_leak
from insight_eval.core_classes.problem import get_problem_type, Problem, ProblemType
from insight_eval.core_classes.evaluation_results import EvaluationResults
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  #type: ignore
from insight_eval.core_classes.combined_evaluation_score import CombinedEvaluationScore
from typing import Optional, Hashable, Callable, Any
from insight_eval.core_classes.solution import Solution
from insight_eval.utils import unique_preserving_order
from sklearn.metrics import roc_auc_score, r2_score  #type: ignore
from insight_eval.logging_config import loggers
from insight_eval.utils import spearman_safe
import insight_eval.config as config
from traceback import format_exc
import copy as copy_package
from pathlib import Path
import pandas as pd
import numpy as np


pd.set_option('future.no_silent_downcasting', True)

eval_logger = loggers.eval_logger
flow_logger = loggers.flow_logger

# -----------------------
# Evaluating the performance of a discoverer on a problem
# Note: we use random forest and one hot low category fields ourselves
# The alternative is to use e.g. lightgbm, but I had some issues with the library
# -----------------------

# comment about fast_mode:
# evaluating performance and coverage metrics on very large data may be very time-consuming
# therefore, with fast_mode=True, we sample the data to a fixed number of samples (set in config.py)


def prepare_data(train_df_full: pd.DataFrame, target: str, test_df_full: pd.DataFrame,
                 max_samples: Optional[int] = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """ Prepare data for training a model
    1. Sample if more than max_samples rows in dataframe
    2. Select only numeric, boolean and categorical columns
    3. Convert categorical columns to 1-hot encoding
    """
    if max_samples is not None and len(train_df_full) > max_samples:
        train_df = train_df_full.sample(max_samples, random_state=config.RANDOM_STATE).sort_index()
    else:
        train_df = train_df_full  # no sampling

    if max_samples is not None and len(test_df_full) > max_samples:
        test_df = test_df_full.sample(max_samples, random_state=config.RANDOM_STATE).sort_index()
    else:
        test_df = test_df_full   # no sampling

    numeric_train = train_df.select_dtypes(include=[np.number, bool])
    numeric_columns = numeric_train.columns.to_list()

    def is_cat(col: Hashable) -> bool:
        try:  # recover and return False
            return 0 < train_df[col].nunique() < config.MAX_UNIQUE_VALUES_FOR_CATEGORICAL
        except:  # noQA
            return False

    categorical = [col for col in train_df.columns if (not (col in numeric_columns)) and col != target and is_cat(col)]
    if len(categorical) > 0:
        train_df, test_df = compress_high_cat_columns(
            train_df, test_df, categorical, max_nvalues=config.MAX_UNIQUE_VALUES_FOR_CATEGORICAL
        )

    columns = [c for c in train_df.columns if (c in numeric_columns or c in categorical) and c != target]
    original_cols_no_target = set(train_df.columns) - {target}
    if set(columns) != original_cols_no_target:
        eval_logger.warning(f'Some columns are ignored {original_cols_no_target - set(columns)}')
    y_train = train_df[target]
    y_test = test_df[target]
    if len(categorical) > 0:
        # apply 1 hot encoding to categorical columns
        try:  # raise error and let calling function handle
            df_combined = safe_concat(train_df, test_df, axis=0)[columns]
            df_encoded = pd.get_dummies(df_combined, columns=categorical)
        except Exception as e:
            error_message = format_exc()
            eval_logger.error(f'Error in prepare_data for categorical: {e} \n{error_message=}')
            raise e

        train_encoded = df_encoded.iloc[:len(train_df)]
        test_encoded = df_encoded.iloc[len(train_df):]
    else:
        train_encoded = train_df[columns]
        test_encoded = test_df[columns]

    eval_logger.debug(f'Columns for classifier: {train_encoded.columns.to_list()}')
    assert train_encoded.columns.to_list() == test_encoded.columns.to_list()

    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        cdf = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
        numeric_cols = cdf.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cdf[col] = cdf[col].replace(np.inf, cdf[col][~np.isinf(cdf[col])].max() + 1)
            cdf[col] = cdf[col].replace(-np.inf, cdf[col][~np.isinf(cdf[col])].min() - 1)
        cdf = cdf.fillna(0)
        return cdf

    x_train = clean_df(train_encoded)
    x_test = clean_df(test_encoded)
    assert len(x_train) == len(y_train) and len(x_test) == len(y_test)
    cols = train_encoded.columns.to_list()
    assert len(cols) == len(set(cols))

    x_train, y_train, x_test, y_test = x_train.fillna(0), y_train.fillna(0), x_test.fillna(0), y_test.fillna(0)
    return x_train, y_train, x_test, y_test


def measure_rf_auc(train_df: pd.DataFrame, target: str, test_df: pd.DataFrame,
                   fast_mode: bool = True) -> tuple[float, RandomForestClassifier | None]:

    x_train, y_train, x_test, y_test = prepare_data(
        train_df, target, test_df, max_samples=config.N_SAMPLES if fast_mode else None
    )

    # Create the Random Forest Classifier and evaluate the performance
    rf: RandomForestClassifier = RandomForestClassifier(n_estimators=config.N_RF_ESTIMATORS, random_state=config.RANDOM_STATE)

    if len(x_train.columns) == 0 or len(y_train) == 0:
        eval_logger.error('measure_rf_auc: no columns in train data')
        return np.nan, None

    rf.fit(x_train, y_train)

    test_probs = rf.predict_proba(x_test)[:, 1]
    if len(unique_preserving_order(y_test)) > 1:
        try:  # roc_auc calculations may fail, if they do, we add an indicator of the failure and continue
            train_auc = roc_auc_score(y_train, rf.predict_proba(x_train)[:, 1])
            test_auc = roc_auc_score(y_test, test_probs)
        except:  # noQA
            eval_logger.warning('Failed computing roc_auc_score, return None')
            return np.nan, None

        eval_logger.debug(f'Train AUC: {train_auc}, test auc: {test_auc}')
    else:
        eval_logger.warning('Single target value in target columns, using accuracy instead of auc')
        test_auc = 1 - np.mean(np.abs(y_test - test_probs))
    return test_auc, rf


def measure_rf_r2_modified(train_df: pd.DataFrame, target: str, test_df: pd.DataFrame,
                           fast_mode: bool = True) -> tuple[float, RandomForestRegressor]:
    eval_logger.debug('Measure_rf_r2_modified: training model for regression')
    x_train, y_train, x_test, y_test = prepare_data(train_df, target, test_df, max_samples=config.N_SAMPLES if fast_mode else None)

    # Creating and Training the Random Forest Model
    regressor: RandomForestRegressor = RandomForestRegressor(n_estimators=config.N_RF_ESTIMATORS, random_state=config.RANDOM_STATE)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = (r2 + 1) / 2  # make it auc like
    if len(set(y_pred)) > 1:
        correlation_coefficient = np.corrcoef(y_test.to_list(), y_pred)[0, 1]  # just for to comparison
    else:
        correlation_coefficient = np.nan

    eval_logger.debug(f'R2 and adjusted_r2 r2 are {r2}, {adjusted_r2} {correlation_coefficient=}')
    return adjusted_r2, regressor


def measure_performance(train_df: pd.DataFrame, target: str, test_df: pd.DataFrame | None = None,
                        problem_type: ProblemType | None = None, fast_mode: bool = True) -> tuple[float, RandomForestClassifier | None]:
    # Note that this may be called when the target is one of the ground-truth feature columns, which may be boolean,
    # numeric, or even small-category strings all scales are like auc: 1 is perfect, 0.5 is like random,
    # below that is awful.

    # Since we call measure_performance also for coverage evaluation, we need to resolve problem_type when not explicitly provided
    if test_df is None:
        test_df = train_df

    if problem_type is None:
        problem_type = get_problem_type(train_df, target, test_df)

    assert train_df.columns.tolist() == test_df.columns.tolist(), 'measure_performance: train and test dataframes must have the same columns'
    n_uniques_train_columns = train_df.drop(columns=[target], inplace=False).nunique()
    n_uniques_test_columns = test_df.drop(columns=[target], inplace=False).nunique()
    if (len(train_df.columns) == 0 or len(test_df.columns) <= 1 or
            all([n <= 1 for n in n_uniques_train_columns]) or all([n <= 1 for n in n_uniques_test_columns])):
        eval_logger.warning(f'Measure_performance: all columns of train_df are constant. return 0.5, None')
        return 0.5, None

    try:  # recover and return np.nan, None
        if problem_type == ProblemType.CLASSIFICATION:
            ret = measure_rf_auc(train_df, target, test_df, fast_mode)
        elif problem_type == ProblemType.MULTICLASS:
            # We do not support multiclass problems or categorical columns in the ground truth
            # when supported, we should call:
            # ret = measure_rf_auc_multiclass(train_df, target, test_df, fast_mode)
            raise NotImplementedError('measure_rf_auc_multiclass is not implemented yet')
        elif problem_type == ProblemType.REGRESSION:
            ret = measure_rf_r2_modified(train_df, target, test_df, fast_mode)
        else:
            raise ValueError(f'Problem type not supported, {problem_type}')
        return ret
    except:  # noQA
        error_message = format_exc()
        eval_logger.warning(f'Error in measure_performance: {error_message}, return np.nan, None')
        return np.nan, None


def _evaluate_performance_of_solution(solution: Solution, fast_mode: bool = True) -> float:
    if len(solution.enriched_train_data.columns) <= 1 or len(solution.enriched_test_data.columns) <= 1:
        eval_logger.warning('Not enough columns in enriched data to evaluate performance')
        return np.nan

    p, rf = measure_performance(solution.enriched_train_data, solution.problem.target_column,
                                solution.enriched_test_data, solution.problem.problem_type,
                                fast_mode=fast_mode)
    return p


def _evaluate_naive(solution: Solution, fast_mode: bool = True) -> float:
    p, rf = measure_performance(solution.problem.train, solution.problem.target_column,
                                solution.problem.test, solution.problem.problem_type,
                                fast_mode=fast_mode)
    return p


def remove_original_columns_from_enriched_data(solution: Solution) -> Solution:
    exclusive_enriched_train_data = solution.enriched_train_data[solution.enriched_column_names + [solution.problem.target_column]]
    exclusive_enriched_test_data = solution.enriched_test_data[solution.enriched_column_names + [solution.problem.target_column]]
    exclusive_solution = copy_package.copy(solution)
    exclusive_solution.enriched_train_data = exclusive_enriched_train_data
    exclusive_solution.enriched_test_data = exclusive_enriched_test_data

    return exclusive_solution


def evaluate_performance_of_solution(solution: Solution, fast_mode: bool = True) -> PerformanceEvaluationResults:
    try:  # recover and return np.nan
        inclusive_performance = _evaluate_performance_of_solution(solution, fast_mode)
        exclusive_solution = remove_original_columns_from_enriched_data(solution)
        if len(exclusive_solution.enriched_column_names) > 0:
            exclusive_performance = _evaluate_performance_of_solution(exclusive_solution, fast_mode)
        else:
            exclusive_performance = 0.5
        naive_performance = _evaluate_naive(solution, fast_mode)
    except Exception:  # noQA
        error_message = format_exc()
        eval_logger.error(f'Error in evaluate_performance_of_solution: {error_message}')
        inclusive_performance = np.nan
        exclusive_performance = np.nan
        naive_performance = np.nan
    return PerformanceEvaluationResults(solution.name(), inclusive_performance, exclusive_performance, naive_performance)


def evaluate_coverage_of_solution(solution: Solution, reference_train_enriched: pd.DataFrame,
                                  reference_test_enriched: pd.DataFrame,
                                  reference_columns: list[str] | None = None,
                                  reference_name: str = 'GroundTruth') -> CoverageEvaluationResults:

    if reference_columns is None:
        raise (RuntimeError('Evaluate_coverage_of_solution - do not know how to find ref columns'))
    else:
        relevant_reference_columns = reference_columns
    target = solution.problem.target_column
    exclusive_solution = remove_original_columns_from_enriched_data(solution)

    coverage_results = evaluate_coverage(
        solution.enriched_train_data, exclusive_solution.enriched_train_data, exclusive_solution.enriched_test_data,
        reference_train_enriched, reference_test_enriched,
        target, relevant_reference_columns,
        solution.name(),
        problem_type=solution.problem.problem_type
    )

    if coverage_results is not None:
        return coverage_results
    else:
        eval_logger.warning(f'Coverage evaluation failed for this problem/solution: {solution.name()}')
        eval_logger.warning('Proceed to evaluate next solution')
        return CoverageEvaluationResults.invalid_coverage_evaluation_results(solution.name(), reference_name)


def evaluate_coverage(inclusive_solution_train_enriched: pd.DataFrame,
                      exclusive_solution_train_enriched: pd.DataFrame, exclusive_solution_test_enriched: pd.DataFrame,
                      ground_truth_train_df: pd.DataFrame, ground_truth_test_df: pd.DataFrame, target_column: str,
                      ground_truth_column_names: list[str],
                      solution_name: str,
                      problem_type: ProblemType) -> CoverageEvaluationResults | None:
    assert len(inclusive_solution_train_enriched) == len(ground_truth_train_df)
    try:  # recover and return None
        inclusive_solution_enriched_numeric_train_df, _, _, _ = \
            prepare_data(inclusive_solution_train_enriched, target_column, inclusive_solution_train_enriched, max_samples=None)  # calling prepare_data only to apply the column processing (without the split of target column); setting fast_mode=True would result in size misalignment

        exclusive_gt_train_df = ground_truth_train_df[unique_preserving_order(ground_truth_column_names + [target_column])]  # remove duplicate columns if existed, while preserving the order of the columns
        exclusive_gt_test_df = ground_truth_test_df[unique_preserving_order(ground_truth_column_names + [target_column])]  # remove duplicate columns if existed, while preserving the order of the columns
        exclusive_ground_truth_train_df_numeric_df, target_train, exclusive_ground_truth_test_numeric_df, target_test = \
            prepare_data(exclusive_gt_train_df, target_column, exclusive_gt_test_df, max_samples=None)

        coverage_metrics_flags = config.COVERAGE_METRICS_TO_INCLUDE

        if coverage_metrics_flags.get('correlation_coverage', False):
            eval_logger.debug('   -> calling eligible_reference_features')
            eligible_ref_features = eligible_reference_features(
                exclusive_ground_truth_train_df_numeric_df,
                pd.DataFrame(target_train), problem_type
            )
            try:
                eval_logger.debug('   -> calling correlation_coverage')
                mean_correlation_coverage, correlation_coverages = correlation_coverage(
                    inclusive_solution_enriched_numeric_train_df,
                    exclusive_ground_truth_train_df_numeric_df[eligible_ref_features.keys()],
                    column_weights=eligible_ref_features)
            except Exception as e:
                error_message = format_exc()
                eval_logger.error(f'Correlation_coverage failed: {e}, \n {error_message}, setting to np.nan, {{}}')
                mean_correlation_coverage = np.nan
                correlation_coverages = DetailedCoveragesDictionary()

        else:
            mean_correlation_coverage, correlation_coverages = np.nan, DetailedCoveragesDictionary()
            eligible_ref_features = CoveragesDictionary()

        if coverage_metrics_flags.get('incremental_performance_coverage', False):
            # note that ground_truth_train_df_numeric_df and ground_truth_test_numeric_df contain only the gt enriched columns
            try:  # capture errors if any and continue
                eval_logger.debug('   -> calling incremental_performance_coverage')
                min_incremental_performance_coverage, incremental_performance_coverages = incremental_performance_coverage(
                    exclusive_solution_train_enriched,
                    exclusive_solution_test_enriched,
                    target_column,
                    exclusive_ground_truth_train_df_numeric_df,
                    exclusive_ground_truth_test_numeric_df
                )
            except Exception as e:
                error_message = format_exc()
                eval_logger.error(f'Incremental_performance_coverage failed: {e}, \n {error_message}, setting to np.nan, {{}}')
                min_incremental_performance_coverage = np.nan
                incremental_performance_coverages = CoveragesDictionary()
        else:
            min_incremental_performance_coverage, incremental_performance_coverages = np.nan, CoveragesDictionary()

        if coverage_metrics_flags.get('predictive_coverage', False):
            try:  # capture errors if any and continue
                eval_logger.debug('   -> calling predictive_coverage')
                mean_predictive_coverage, predictive_coverages, predictive_coverage_weights = predictive_coverage(
                    exclusive_solution_train_enriched, exclusive_solution_test_enriched,
                    target_column,
                    exclusive_ground_truth_train_df_numeric_df,
                    exclusive_ground_truth_test_numeric_df
                )
            except Exception as e:
                error_message = format_exc()
                eval_logger.error(f'predictive_coverage failed: {e}, \nerror_message={error_message} \nsetting to np.nan, {{}}')
                mean_predictive_coverage = np.nan
                predictive_coverages = CoveragesDictionary()
                predictive_coverage_weights = CoveragesDictionary()

        else:
            mean_predictive_coverage, predictive_coverages, predictive_coverage_weights = np.nan, CoveragesDictionary(), CoveragesDictionary()

        if coverage_metrics_flags.get('single_column_predictive_coverage', False):
            try:  # capture errors if any and continue
                eval_logger.debug('   -> calling single_column_predictive_coverage')
                mean_single_column_predictive_coverage, single_column_predictive_coverages, single_column_predictive_coverage_weights = single_column_predictive_coverage(
                    exclusive_solution_train_enriched, exclusive_solution_test_enriched,
                    target_column,
                    exclusive_ground_truth_train_df_numeric_df,
                    exclusive_ground_truth_test_numeric_df
                )
            except Exception as e:
                error_message = format_exc()
                eval_logger.error(f'predictive_coverage failed: {e}, \nerror_message={error_message} \nsetting to np.nan, {{}}')
                mean_single_column_predictive_coverage = np.nan
                single_column_predictive_coverages = DetailedCoveragesDictionary()
                single_column_predictive_coverage_weights = CoveragesDictionary()

        else:
            mean_single_column_predictive_coverage, single_column_predictive_coverages, single_column_predictive_coverage_weights = np.nan, DetailedCoveragesDictionary(), CoveragesDictionary()

    except:  # noQA
        error_message = format_exc()
        eval_logger.error(f'Evaluate_coverage() failed: {error_message=}, return None')
        return None  # it is ok to return None here, calling function will handle

    return CoverageEvaluationResults(
        solution_name=solution_name,
        reference_name='ground truth',
        mean_correlation_coverage=mean_correlation_coverage,
        correlation_coverages=correlation_coverages,
        correlation_coverage_weights=eligible_ref_features,
        min_incremental_performance_coverage=min_incremental_performance_coverage,
        incremental_performance_coverages=incremental_performance_coverages,
        mean_predictive_coverage=mean_predictive_coverage,
        predictive_coverages=predictive_coverages,
        predictive_coverage_weights=predictive_coverage_weights,
        mean_single_column_predictive_coverage=mean_single_column_predictive_coverage,
        single_column_predictive_coverages=single_column_predictive_coverages,
        single_column_predictive_coverage_weights=single_column_predictive_coverage_weights
    )


def compress_high_cat_columns(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              columns: list[str], max_nvalues: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in columns:
        # Ensure the column is categorical
        if pd.api.types.is_numeric_dtype(train_df[col].dtype):
            raise Exception(f"{col} must be a categorical column.")
        value_counts = train_df[col].value_counts()
        if len(value_counts) > max_nvalues:
            top_values = value_counts.nlargest(max_nvalues - 1).index
            train_df[col] = train_df[col].where(train_df[col].isin(top_values), '_misc')
            if test_df is not None:
                test_df[col] = test_df[col].where(test_df[col].isin(top_values), '_misc')
    return train_df, test_df


# -----------------------
# Coverage evaluation
# -----------------------


def safe_concat(df1: pd.DataFrame, df2: pd.DataFrame, axis: int) -> pd.DataFrame:
    assert isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame), 'Both inputs must be pandas DataFrames'
    match axis:
        case 0:  # rows
            assert len(df1.columns) == len(df2.columns), f'Column mismatch: {len(df1.columns)} vs {len(df2.columns)}'
            assert df1.columns.equals(df2.columns), f'Column names mismatch: {df1.columns} vs {df2.columns}'
            return pd.concat([df1, df2], axis="index")
        case 1:  # columns
            assert len(df1) == len(df2)
            assert df1.index.equals(df2.index), f'Index mismatch: {df1.index} vs {df2.index}'
            return pd.concat([df1, df2], axis="columns")

    raise ValueError(f'Invalid axis {axis}, must be 0 or 1')


def correlation_coverage(inclusive_features_df: pd.DataFrame, exclusive_gt_df: pd.DataFrame, fast_mode: bool = True,
                         column_weights: CoveragesDictionary | None = None) -> tuple[float, DetailedCoveragesDictionary]:
    # for each column in golden set, find the feature that covers it best, and take average
    coverages = DetailedCoveragesDictionary()
    assert len(exclusive_gt_df) == len(inclusive_features_df)
    gt_df = exclusive_gt_df.replace(True, 1).replace(False, 0).fillna(0)
    features_df = inclusive_features_df.replace(True, 1).replace(False, 0).fillna(0)
    if fast_mode and len(exclusive_gt_df) > 15000:
        n_samples = config.N_SAMPLES
        gt_df = gt_df.sample(n_samples, random_state=config.RANDOM_STATE)
        features_df = features_df.sample(n_samples, random_state=config.RANDOM_STATE)
    correlation_matrix = pd.DataFrame(index=gt_df.columns, columns=features_df.columns).fillna(0.0)
    mean_coverage = 0.0
    coverage_strengths = []
    coverage_weights = []
    for gt_col in gt_df.columns:  # golden-set df
        for features_col in features_df.columns:  # features df
            try:  # recover and populate with np.nan (note that next line is fillna(0))
                if config.USE_SPEARMAN_FOR_COVERAGE:
                    rho, p_value = spearman_safe(list(gt_df[gt_col]), list(features_df[features_col]))

                    correlation_matrix.at[gt_col, features_col] = rho
                else:
                    correlation_matrix.at[gt_col, features_col] = np.corrcoef(list(gt_df[gt_col]), (list(features_df[features_col])))[0, 1]

            except:  # noQA
                eval_logger.warning(f'Something is wrong, {gt_col} vs {features_col} please check!')
                correlation_matrix.at[gt_col, features_col] = np.nan

        correlation_matrix.fillna(0.0)
        correlation_matrix.replace([np.inf, -np.inf], 0.0, inplace=True)
        correlations = np.nan_to_num(np.abs(np.array(list(correlation_matrix.loc[gt_col]))))
        if len(correlations) > 0:
            best_coverer_ind = np.argmax(correlations)
            coverage_strength = float(correlations[best_coverer_ind])
            coverage_strengths.append(coverage_strength)
            covering_feature = str(inclusive_features_df.columns[int(best_coverer_ind)])
            coverages[gt_col] = {covering_feature: coverage_strength}
        else:
            coverage_strength = 0.0
            coverage_strengths.append(coverage_strength)
            covering_feature = ''
            coverages[gt_col] = {covering_feature: coverage_strength}

        if column_weights is not None:
            coverage_weights.append(column_weights[gt_col])
        else:
            coverage_weights.append(1.0)
    if len(exclusive_gt_df.columns) > 0:
        mean_coverage = float(np.average(coverage_strengths, weights=coverage_weights))
    return mean_coverage, coverages


def incremental_performance_coverage(
        exclusive_solution_train_w_target_df: pd.DataFrame,
        exclusive_solution_test_w_target_df: pd.DataFrame,
        target_column: str,
        exclusive_ground_truth_train_wo_target_df: pd.DataFrame,
        exclusive_ground_truth_test_wo_target_df: pd.DataFrame,
        fast_mode: bool = True) -> tuple[float, CoveragesDictionary]:
    # Both the solution_df's and ground_truth_df's are "exclusive" - they contain only the enriched columns.
    # Note that we use the exclusive_ground_truth_train_wo_target_df and exclusive_ground_truth_test_wo_target_df
    # don't contain the target exclusive indicates that we don't include the original columns of the problem in the
    # solution, it does not indicate anything wrt including the target
    assert target_column in exclusive_solution_train_w_target_df.columns and target_column in exclusive_solution_test_w_target_df.columns
    assert target_column not in exclusive_ground_truth_train_wo_target_df.columns and target_column not in exclusive_ground_truth_test_wo_target_df.columns

    auc_solution = measure_rf_auc(train_df=exclusive_solution_train_w_target_df, target=target_column,
                                  test_df=exclusive_solution_test_w_target_df, fast_mode=fast_mode)[0]
    coverages = CoveragesDictionary()

    for c in exclusive_ground_truth_train_wo_target_df.columns:
        train_augmented = safe_concat(exclusive_solution_train_w_target_df, exclusive_ground_truth_train_wo_target_df[[c]].copy().rename(columns={c: f"{c}_aug"}), axis=1)
        test_augmented = safe_concat(exclusive_solution_test_w_target_df, exclusive_ground_truth_test_wo_target_df[[c]].copy().rename(columns={c: f"{c}_aug"}), axis=1)

        auc_augmented_curr = measure_rf_auc(train_df=train_augmented, target=target_column,
                                            test_df=test_augmented, fast_mode=fast_mode)[0]

        rho: Callable[[float], float] = lambda x: 2.0*max(x-0.5, 0.0)
        coverage_curr = 1.0 - max(rho(auc_augmented_curr) - rho(auc_solution), 0.0)  # here we measure how much of the column c we covered by the rest
        coverages.data[c] = coverage_curr

    coverage_min = float(np.min(list(coverages.data.values())))  # average with uniform correlation_coverage_weights (the importance is already captured by the coverage of each column)

    return coverage_min, coverages


def predictive_coverage(exclusive_solution_train_w_target_df: pd.DataFrame,
                        exclusive_solution_test_w_target_df: pd.DataFrame,
                        target_column: str,
                        exclusive_ground_truth_train_wo_target_df: pd.DataFrame,
                        exclusive_ground_truth_test_wo_target_df: pd.DataFrame,
                        fast_mode: bool = True) -> tuple[float, CoveragesDictionary, CoveragesDictionary]:
    # both the solution_df's and ground_truth_df's are "exclusive" - they contain only the enriched columns
    # note that exclusive_solution_train_w_target_df and exclusive_solution_test_w_target_df DO contain the target

    assert target_column not in exclusive_ground_truth_train_wo_target_df.columns and target_column not in exclusive_ground_truth_test_wo_target_df.columns

    coverages = CoveragesDictionary()
    weights = CoveragesDictionary()

    sum_weights = 0.0

    for c in exclusive_ground_truth_train_wo_target_df.columns:
        c_aug = f"{c}_aug"
        train_augmented = safe_concat(exclusive_solution_train_w_target_df, exclusive_ground_truth_train_wo_target_df[[c]].copy().rename(columns={c: c_aug}), axis=1)
        test_augmented = safe_concat(exclusive_solution_test_w_target_df, exclusive_ground_truth_test_wo_target_df[[c]].copy().rename(columns={c: c_aug}), axis=1)

        auc_augmented_curr = measure_performance(train_df=train_augmented,
                                                 test_df=test_augmented,
                                                 target=c_aug, fast_mode=fast_mode)[0]

        coverage_curr = 2.0 * max(auc_augmented_curr - 0.5, 0.0)
        coverages[c] = coverage_curr

        # note - ground_truth df's don't have the target column, so we take it from the solution df's
        gt_train_single_col = safe_concat(exclusive_solution_train_w_target_df[[target_column]], exclusive_ground_truth_train_wo_target_df[[c]], axis=1)
        gt_test_single_col = safe_concat(exclusive_solution_test_w_target_df[[target_column]], exclusive_ground_truth_test_wo_target_df[[c]], axis=1)

        weight_curr = measure_performance(train_df=gt_train_single_col, test_df=gt_test_single_col,
                                          target=target_column, fast_mode=fast_mode)[0]
        weight_curr = 2.0 * max(weight_curr - 0.5, 0.0)

        weights[c] = weight_curr
        sum_weights += weight_curr

    weight_regularization_threshold = 1e-5

    # Compensate weights in case their sum is below the threshold
    compensation = max(weight_regularization_threshold - sum_weights, 0.0) / len(weights)
    weights = CoveragesDictionary({c: weights[c] + compensation for c in weights.keys()})
    sum_weights = max(sum_weights, weight_regularization_threshold)

    sum_coverages = np.sum([weights[c] * coverages[c] for c in coverages.data.keys()])
    coverage_mean = sum_coverages / sum_weights

    return coverage_mean, coverages, weights


def single_column_predictive_coverage(
        exclusive_solution_train_w_target_df: pd.DataFrame,
        exclusive_solution_test_w_target_df: pd.DataFrame,
        target_column: str,
        exclusive_ground_truth_train_wo_target_df: pd.DataFrame,
        exclusive_ground_truth_test_wo_target_df: pd.DataFrame,
        fast_mode: bool = True) -> tuple[float, DetailedCoveragesDictionary, CoveragesDictionary]:
    # both the solution_df's and ground_truth_df's are "exclusive" - they contain only the enriched columns
    # note that we use the exclusive_solution_train_w_target_df and exclusive_solution_test_w_target_df don't contain the target

    assert target_column in exclusive_solution_train_w_target_df.columns and target_column in exclusive_solution_test_w_target_df.columns
    assert target_column not in exclusive_ground_truth_train_wo_target_df.columns and target_column not in exclusive_ground_truth_test_wo_target_df.columns

    coverages = DetailedCoveragesDictionary()
    weights = CoveragesDictionary()

    sum_weights = 0.0

    for c in exclusive_ground_truth_train_wo_target_df.columns:
        c_aug = f"{c}_aug"

        coverage_per_coverer = {}
        for coverer in exclusive_solution_train_w_target_df.columns.difference([target_column]):
            train_single_covering_col = safe_concat(exclusive_solution_train_w_target_df[[coverer]], exclusive_ground_truth_train_wo_target_df[[c]].copy().rename(columns={c: c_aug}), axis=1)
            test_single_covering_col = safe_concat(exclusive_solution_test_w_target_df[[coverer]], exclusive_ground_truth_test_wo_target_df[[c]].copy().rename(columns={c: c_aug}), axis=1)

            auc_tmp, _ = measure_performance(train_df=train_single_covering_col,
                                             test_df=test_single_covering_col,
                                             target=c_aug, fast_mode=fast_mode)
            coverage_per_coverer[coverer] = 2.0 * max(auc_tmp - 0.5, 0.0)

        best_coverer = max(coverage_per_coverer, key=lambda x: coverage_per_coverer[x])
        best_coverage = coverage_per_coverer[best_coverer]

        coverages[c] = {best_coverer: best_coverage}

        # note - ground_truth df's don't have the target column, so we take it from the solution df's
        gt_train_single_col = safe_concat(exclusive_solution_train_w_target_df[[target_column]], exclusive_ground_truth_train_wo_target_df[[c]], axis=1)
        gt_test_single_col = safe_concat(exclusive_solution_test_w_target_df[[target_column]], exclusive_ground_truth_test_wo_target_df[[c]], axis=1)

        weight_curr, _ = measure_performance(
            train_df=gt_train_single_col, test_df=gt_test_single_col,
            target=target_column, fast_mode=fast_mode)
        weight_curr = 2.0 * max(weight_curr - 0.5, 0.0)
        weights[c] = weight_curr
        sum_weights += weight_curr

    weight_regularization_threshold = 1e-5

    # Compensate weights in case their sum is below the threshold
    compensation = max(weight_regularization_threshold - sum_weights, 0.0) / len(weights)
    weights = CoveragesDictionary({c: weights[c] + compensation for c in weights.keys()})
    sum_weights = max(sum_weights, weight_regularization_threshold)

    # Convenient functions for getting the value of a dictionary that contains a single element
    get_single_value: Callable[[Any], Any] = lambda singleton_dict: next(iter(singleton_dict.values()))

    sum_coverages = np.sum([weights[c] * get_single_value(coverages[c]) for c in coverages.keys()])
    coverage_mean = sum_coverages / sum_weights

    return coverage_mean, coverages, weights


def eligible_reference_features(gt_df: pd.DataFrame, target_df: pd.DataFrame, problem_type: ProblemType) -> CoveragesDictionary:
    # target_df is a single column df with the target column
    if problem_type != ProblemType.MULTICLASS:
        coverages = correlation_coverage(target_df, gt_df, column_weights=None)[1]  # for each feature, how much it covers the target
    else:
        # we currently do not support MULTICLASS problems as well as categroical columns in the ground truth
        # once supported, we will add here -
        # coverages = correlation_with_multiclass_target(target_df, gt_df)
        raise NotImplementedError('Eligible_reference_features is not implemented for multiclass problems')

    # Convenient functions for getting the value of a dictionary that contains a single element
    get_single_value: Callable[[Any], float] = lambda singleton_dict: next(iter(singleton_dict.values()))

    eligible_features_with_corrs: dict[str, float] = {c: get_single_value(coverages[c]) for c in coverages.keys() if get_single_value(coverages[c]) > config.ELIGIBILITY_THRESHOLD_FOR_GT_COVERAGE}
    return CoveragesDictionary({
        key: value
        for key, value in sorted(
            eligible_features_with_corrs.items(), key=lambda kv: kv[1], reverse=True
        )
    })


def evaluate(problem: Problem, solution: Solution, ground_truth: Solution, solution_path: Path) -> EvaluationResults:
    # Note - "solution path" is provided only for logging purposes, to track which solution is being evaluated
    if problem is None or ground_truth is None:
        eval_logger.error('In evaluate() problem or ground_truth is None')
        raise ValueError('evaluate(): problem or ground_truth is None')

    try:  # report error and issue Exception (calling function should handle)
        report_str = f'*** Starting evaluation for problem: {problem.name} ***'
        report_underline = '-' * len(report_str)
        flow_logger.info(f'{report_str}')
        flow_logger.info(f'{report_underline}')
        flow_logger.info(f'{solution_path=}')

        flow_logger.info('Evaluating solution performance')
        performance_eval = evaluate_performance_of_solution(solution)

        flow_logger.info('Evaluating ground-truth performance')
        ground_truth_performance_eval = evaluate_performance_of_solution(ground_truth)
        eval_logger.debug(f'Ground truth performance:\n {ground_truth_performance_eval.summary()}')
        eval_logger.debug(f'Solution performance:\n {performance_eval.summary()}')

        if config.EVALUATE_GROUND_TRUTH_COVERAGE:
            flow_logger.info('Evaluating ground-truth coverage')
            gt_coverage_eval = evaluate_coverage_of_solution(
                ground_truth,
                ground_truth.enriched_train_data,
                ground_truth.enriched_test_data,
                ground_truth.enriched_column_names,
                reference_name=ground_truth.name()
            )
        else:
            gt_coverage_eval = None

        # Ensure the train and test data are not empty
        if len(solution.enriched_column_names) == 0:
            eval_logger.warning('Solution has no enriched columns, coverage metrics are 0.0, target leak evaluation is False')
            coverage_eval: CoverageEvaluationResults = CoverageEvaluationResults(
                solution.name(), ground_truth.name(),
                mean_correlation_coverage=np.nan,
                min_incremental_performance_coverage=0.0,
                mean_predictive_coverage=0.0,
                mean_single_column_predictive_coverage=0.0
            )
            target_leak_eval = False
        else:
            flow_logger.info('Evaluating coverage and target leak')
            coverage_eval = evaluate_coverage_of_solution(solution, ground_truth.enriched_train_data,
                                                          ground_truth.enriched_test_data,
                                                          ground_truth.enriched_column_names,
                                                          reference_name=ground_truth.name())
            eval_logger.debug(coverage_eval.summary(detailed=True))
            target_leak_eval = evaluate_target_leak(solution)

        metrics_dict = {
            'correlation_coverage': 'mean_correlation_coverage',
            'incremental_performance_coverage': 'min_incremental_performance_coverage',
            'predictive_coverage': 'mean_predictive_coverage',
            'single_column_predictive_coverage': 'mean_single_column_predictive_coverage'
        }

        if coverage_eval:
            flags = config.COVERAGE_METRICS_TO_INCLUDE
            weights = config.COVERAGE_METRIC_WEIGHTS
            sum_weights = 0.0
            sum_score = 0.0
            for config_key, coverage_key in metrics_dict.items():
                coverage_value = getattr(coverage_eval, coverage_key, np.nan)
                if flags[config_key] and not np.isnan(coverage_value):
                    sum_weights += weights[config_key]
                    sum_score += weights[config_key] * coverage_value

            coverage_score = sum_score / sum_weights if sum_weights > 0 else 0.0

        combined_score = CombinedEvaluationScore(
            inclusive_performance=performance_eval.inclusive_performance,
            exclusive_performance=performance_eval.exclusive_performance,
            naive_performance=performance_eval.naive_performance,
            coverage=coverage_score,
            target_leak=target_leak_eval,
            ground_truth_inclusive_performance=ground_truth_performance_eval.inclusive_performance,
            ground_truth_exclusive_performance=ground_truth_performance_eval.exclusive_performance
        )
        eval_logger.debug(f'Combined score: {combined_score.summary()}')
        report_str = f'**** Finished successfully processing problem: {problem.name} ***'
        report_underline = '-' * len(report_str)
        flow_logger.info(f'{report_str}')
        flow_logger.info(f'{report_underline}')

        if target_leak_eval:
            eval_logger.warning(f'Target leak detected for problem {problem.name}')

        evaluation_results = EvaluationResults(
            performance_evaluation_results=performance_eval,
            ground_truth_performance_evaluation_results=ground_truth_performance_eval,
            coverage_evaluation_results=coverage_eval,
            ground_truth_coverage_evaluation_results=gt_coverage_eval if config.EVALUATE_GROUND_TRUTH_COVERAGE else None,
            target_leak_evaluation_results=target_leak_eval,
            combined_score=combined_score
        )
        return evaluation_results
    except Exception as e:
        error_message = format_exc()
        eval_logger.error(f'Error evaluating problem {problem.name}: {e}')
        eval_logger.error(f'Solution path: {solution_path}\nerror message: {error_message}')

        report_str = f'**** Exception while processing problem: {problem.name} ***'
        report_underline = '-' * len(report_str)
        eval_logger.error(f'{report_str}')
        eval_logger.error(f'{report_underline}')
        raise e
