from insight_eval.reports.repo_stats import construct_problem_file_name
from insight_eval.logging_config import loggers
from typing import cast, Any
import numpy.typing as npt
from pandas import Series
from pathlib import Path
import pandas as pd
import numpy as np
from natsort import natsort_keygen


flow_logger = loggers.flow_logger


def average_of_x_by_y(df: pd.DataFrame, x: str, y: list[str]) -> dict[str, pd.Series]:
    grouped_df = df.groupby(y)[x]
    dist = grouped_df.value_counts(dropna=False).sort_index().unstack().fillna(np.nan)
    y_counts = dist.sum(axis=1)
    assert all(np.isreal(dist.columns)), 'average_of_x_by_y: values of y-columns must be real numbers'
    averages = (dist*dist.columns).sum(axis=1) / y_counts
    return {'averages': averages, 'counts': y_counts}


def dist_of_x(df: pd.DataFrame, x: str) -> pd.Series:
    res = pd.Series(dict(sorted(df[x].value_counts().to_dict().items())))
    return res


def estimate_problem_difficulty(evaluation_dict: dict[str, float]) -> dict[str, float | str]:
    performance_thresholds = np.array([0.6, 0.7, 0.8, 0.9])
    coverage_thresholds = np.array([0.1, 0.2, 0.4, 0.6, 0.75])
    target_leak_thresholds = np.array([1.0])

    def eval_score(eval_field_name: str, eval_thresholds: npt.NDArray[np.float64]) -> float:
        # note - the higher the score, the easier the problem was to solve by the agents
        value = evaluation_dict.get(eval_field_name, np.nan)
        value = 0 if str(value).lower()=='false' else 1 if str(value).lower()=='true' else value  # convert False/True strings to 0/1
        return sum(value >= eval_thresholds) / len(eval_thresholds) if not np.isnan(value) else np.nan
    # end of eval_score

    thresholds_dict = {
        'exclusive_performance': performance_thresholds,
        'mean_correlation_coverage': coverage_thresholds,
        'min_incremental_performance_coverage': coverage_thresholds,
        'mean_predictive_coverage': coverage_thresholds,
        'mean_single_column_predictive_coverage': coverage_thresholds,
        'ground_truth_exclusive_performance': performance_thresholds,
        'target_leak_indicator': target_leak_thresholds,
    }

    weights_dict = {
        'exclusive_performance': 0.4,
        'mean_correlation_coverage': 0.0,
        'min_incremental_performance_coverage': 0.18,
        'mean_predictive_coverage': 0.0,
        'mean_single_column_predictive_coverage': 0.42,
        'ground_truth_exclusive_performance': 0,
        'target_leak_indicator': 0
    }

    scores = {}
    for field_name, thresholds in thresholds_dict.items():
        score_min = eval_score(f'min_{field_name}', thresholds)
        score_max = eval_score(f'max_{field_name}', thresholds)
        scores[field_name] = (score_min + score_max) / 2  # in [0, 1] range

    clean_weighted_score_values = [(v, weights_dict[k]) for k, v in scores.items() if not np.isnan(v)]
    sum_clean_weights = sum(w for v, w in clean_weighted_score_values)
    if len(clean_weighted_score_values) > 0 and sum_clean_weights > 0:
        difficulty_scale = ['4 very hard', '3 hard', '2 medium', '1 easy', '0 very easy']
        aggregate_score = sum(v*w for v, w in clean_weighted_score_values) / sum_clean_weights
        scale = len(difficulty_scale)
        int_score = min(scale-1, int(scale * aggregate_score))  # scale to [0, scale-1] range
        difficulty_level = difficulty_scale[int_score]
        difficulty_score = 1 - aggregate_score  # difficulty_score in [0, 1], 1 is hardest
        flow_logger.debug(f'Problem difficulty level: {difficulty_level}, {difficulty_score=}, {scale=}')
        return {'problem_difficulty_level': difficulty_level, 'problem_difficulty_score': difficulty_score}

    return {'problem_difficulty_level': 'n/a', 'problem_difficulty_score': np.nan}


def combined_repo_eval_stats(df_repo_stats: pd.DataFrame, df_eval_stats: pd.DataFrame, agent_names_for_difficulty: list[str], output_path: Path) -> pd.DataFrame:
    df_repo_stats_w_difficulty = df_repo_stats.copy()
    df_repo_stats_w_difficulty.insert(6, 'problem_difficulty_level', 'n/a')
    df_repo_stats_w_difficulty.insert(7, 'problem_difficulty_score', np.nan)

    df_evaluation_stats = df_eval_stats.copy()
    df_evaluation_stats = df_evaluation_stats.reset_index().drop(columns=['index']).set_index(['agent', 'problem_name'])

    new_rows = []

    fields_to_process = ['exclusive_performance', 'ground_truth_exclusive_performance',
                         'mean_correlation_coverage', 'mean_single_column_predictive_coverage',
                         'min_incremental_performance_coverage',
                         'target_leak_indicator']

    field_types = df_evaluation_stats[fields_to_process].dtypes
    assert all(t in [float, int, bool] for t in field_types), \
        f'Fields {fields_to_process} must be of type float, int or bool, found: {field_types[fields_to_process]}'
    assert list(field_types)[-1] == 'bool'
    for indx, row in df_repo_stats_w_difficulty.iterrows():
        problem_identifier = construct_problem_file_name(row.problem_id, row.problem_name, row.variation, row.primary_target_type, row.secondary_target_type)
        flow_logger.debug(f'Processing problem: {problem_identifier}')

        eval_rows: dict[str, Series | dict[Any, Any]] = {}
        new_eval_rows: dict[str, Series | dict[Any, Any]] = {}
        for agent_name in agent_names_for_difficulty:
            eval_rows[agent_name] = cast(pd.Series, df_evaluation_stats.loc[(agent_name, problem_identifier)]) if (agent_name, problem_identifier) in df_evaluation_stats.index else {}

        for agent_name in agent_names_for_difficulty:
            new_eval_rows[agent_name] = {
                f"{agent_name}_{key}": value for key, value in eval_rows[agent_name].items() if key != "index"
            }

        flow_logger.debug(f'problem identifier: {problem_identifier}')
        flow_logger.debug(f'eval_rows: {eval_rows}')
        flow_logger.debug(f'new_eval_rows: {new_eval_rows}')

        aggregates_row_dict = {}

        for field_name in fields_to_process:
            field_values = [new_eval_row.get(agent_name + '_' + field_name, np.nan) if field_name != 'target_leak_indicator' else 'False'
                            for agent_name, new_eval_row in new_eval_rows.items()]
            field_values = [v for v in field_values if not pd.isna(v) and v != '']
            if len(field_values) > 0:
                aggregates_row_dict[f'min_{field_name}'] = min(field_values)
                aggregates_row_dict[f'max_{field_name}'] = max(field_values)
                # note: in the assert below aggregates_row_dict[f'min_{field_name}'] may be the satring 'True' or 'False'
                v= aggregates_row_dict[f'min_{field_name}']
                assert v in ['True', 'False'] or not np.isnan(v), f'Problem {problem_identifier} has NaN in min_{field_name} / max_{field_name}'
            else:
                aggregates_row_dict[f'min_{field_name}'] = np.nan
                aggregates_row_dict[f'max_{field_name}'] = np.nan

        difficulty_dict = estimate_problem_difficulty(aggregates_row_dict)
        df_repo_stats_w_difficulty.at[indx, 'problem_difficulty_level'] = difficulty_dict['problem_difficulty_level']
        df_repo_stats_w_difficulty.at[indx, 'problem_difficulty_score'] = difficulty_dict['problem_difficulty_score']

        new_row_dict = {'problem_identifier': problem_identifier} | row.to_dict() | difficulty_dict | aggregates_row_dict
        for agent_name in agent_names_for_difficulty:
            new_row_dict.update(new_eval_rows[agent_name])

        new_rows.append(new_row_dict)

    new_df = pd.DataFrame(new_rows)
    output_path.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_path / 'repo_eval_stats_with_difficulty.csv', index=False)

    stats_df = pd.DataFrame(columns=['dimension', 'count', 'average problem difficulty'])
    with open(output_path / 'repo_eval_stats.txt', 'w') as f:
        stat_rows = []
        for field_names in [['n_tables'], ['problem_domain'], ['n_primary_columns'], ['total_columns'], ['n_gt_insights'], ['problem_difficulty_level'], ['primary_target_type', 'secondary_target_type']]:
            flow_logger.info(f'problem_difficulty_score by {field_names}:')
            f.write(f'problem_difficulty_score by {field_names}:\n')
            hdr_dict = {'dimension': field_names[0], 'count': 'count', 'average problem difficulty': 'average problem difficulty'}
            stats_df = pd.concat([stats_df, pd.DataFrame(hdr_dict, index=[0])])
            dict_xy = average_of_x_by_y(new_df, x='problem_difficulty_score', y=field_names)

            for k, v in dict_xy.items():
                flow_logger.info(f'{k}: {v}')
                f.write(f'{k}: {v}\n')
            f.write('\n')

            crnt_stats_df = pd.DataFrame(dict_xy)[['counts', 'averages']].reset_index().rename(columns={field_names[0]: 'dimension', 'counts': 'count', 'averages': 'average problem difficulty'})
            stats_df = pd.concat([stats_df, crnt_stats_df])
            pass

    stats_df.to_csv(output_path / 'repo_eval_stats.csv', index=False)
    return new_df


def agent_performance_by_difficulty_level(detailed_eval_stats_df: pd.DataFrame, repo_stats_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    detailed_eval_modified = detailed_eval_stats_df.copy()
    repo_stats_df_modified = repo_stats_df.copy()

    detailed_eval_modified.rename(columns={'problem_name': 'problem_identifier'}, inplace=True)

    new_columns = ['problem_id', 'problem_name', 'variation',
                   'primary_target_type', 'secondary_target_type', 'problem_difficulty_level', 'problem_difficulty_score']

    # Merge and add new columns from repo_stats_df_modified
    merged_df = pd.merge(
        detailed_eval_modified,
        repo_stats_df_modified[new_columns + ['problem_identifier']],
        on='problem_identifier',
        how='left',
        suffixes=('', '_repo')
    ).fillna({'problem_difficulty_level': 'n/a', 'problem_difficulty_score': np.nan})

    assert all('_repo' not in col for col in merged_df.columns), "agent_performance_by_difficulty: merge should not have introduced suffixes to column names"

    columns_reordered = merged_df.columns.tolist()[:2] + new_columns + merged_df.columns.tolist()[2:-len(new_columns)]
    detailed_eval_w_difficulty = merged_df[columns_reordered]

    # Use natsort_keygen to generate a key function for natural sorting
    key_func = natsort_keygen()
    sorted_detailed_eval_w_difficulty = detailed_eval_w_difficulty.sort_values(
        by=['agent', 'problem_identifier'], key=lambda col: col.map(key_func)
    )
    # detailed_eval_w_difficulty = detailed_eval_w_difficulty.sort_values(['agent', 'problem_identifier'])

    sorted_detailed_eval_w_difficulty.to_csv(output_path / 'detailed_eval_with_difficulty.csv', index=False)

    # Choose columns to average (adjust as needed)
    coverage_cols = [
        'combined_score',
        'inclusive_performance',
        'exclusive_performance',
        'coverage_score',
        'mean_correlation_coverage',
        'mean_single_column_predictive_coverage',
        'min_incremental_performance_coverage',
        'mean_predictive_coverage'
    ]

    # Group by agent and problem_difficulty_level, then calculate mean
    averages_df = (
        sorted_detailed_eval_w_difficulty
        .groupby(['agent', 'problem_difficulty_level'])[coverage_cols]
        .mean()
        .reset_index()
    )

    # Calculate group sizes
    group_sizes = (
        sorted_detailed_eval_w_difficulty
        .groupby(['agent', 'problem_difficulty_level'])
        .size()
        .reset_index(name='group_count')
    )

    # Merge group counts into averages DataFrame
    averages_df = pd.merge(averages_df, group_sizes, on=['agent', 'problem_difficulty_level'], how='left')
    averages_df = averages_df[['agent', 'problem_difficulty_level', 'group_count'] + coverage_cols]

    # Save or print the result
    averages_df.to_csv(output_path / 'agent_averages_with_difficulty.csv', index=False)
    print(averages_df)
    print(f'number of evaluations: {len(sorted_detailed_eval_w_difficulty)}')

    return averages_df
