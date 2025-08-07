from insight_eval.evaluation_framework.aggregate_batch_evalaution import evaluate_agent_on_folder, generate_agent_aggregate_report
from insight_eval.evaluation_framework.batch_evaluate import run_evaluate
from insight_eval.readers.file_utils import list_directory_items
from insight_eval.logging_config import loggers
from natsort import natsort_keygen
from pathlib import Path
import pandas as pd
import shutil


def evaluate_multiple_agents(
        agent_names: list[str],
        problems_folder_path: Path,
        solutions_folder_path: Path,
        evaluations_folder_path: Path,
        delete_existing_evaluations: bool,
        problem_sets_hierarchy: bool,
        combined_reports_output_path: Path
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    #
    if delete_existing_evaluations:
        shutil.rmtree(evaluations_folder_path, ignore_errors=True)  # remove existing evaluations if exist

    combined_reports_output_path.mkdir(parents=True, exist_ok=True)

    averages = []
    detailed_eval_df = pd.DataFrame()
    for agent_name in agent_names:
        solutions_folder = solutions_folder_path / agent_name
        if not Path.exists(solutions_folder):
            loggers.eval_logger.warning(f'*** Solutions folder for agent {agent_name} does not exist, skipping combined report generation for this agent')
            continue

        evaluations_folder = evaluations_folder_path / agent_name

        # In each solution: folder we now have several problem_sets, in this case the flag
        if not problem_sets_hierarchy:
            evaluation_summary, report_results = evaluate_agent_on_folder(
                agent_name, problems_folder_path, solutions_folder, evaluations_folder, delete_existing_evaluations
            )

            stats_df, average_row = generate_agent_aggregate_report(
                agent_name, problems_folder_path, evaluations_folder
            )

            # remove average row from stats_df
            stats_df = stats_df[stats_df[f'problem_name_{agent_name}'] != 'average']

            average_row = {key.removesuffix(f'_{agent_name}'): value for key, value in average_row.items()}
            average_row = {
                'n_problems': evaluation_summary.n_problems,
                'n_solutions': evaluation_summary.n_solutions,
                'agent': agent_name,
                'n_evaluations': evaluation_summary.n_evaluations,
                'n_valid_evaluations': len(stats_df)
            } | average_row

            averages.append(average_row)
            stats_df_clean_columns = stats_df.rename(
                {col_name: col_name.removesuffix(f'_{agent_name}') for col_name in stats_df.columns if
                 col_name.endswith(f'_{agent_name}')}, axis=1)
            stats_df_clean_columns.insert(0, 'agent', agent_name)  # add agent name to the stats_df
            detailed_eval_df = pd.concat([detailed_eval_df, stats_df_clean_columns], ignore_index=True)
        else:  # problem_sets_hierarchy is True
            problem_sets_dirs = list_directory_items(solutions_folder)
            for solutions_dir in problem_sets_dirs:
                problems_dir = problems_folder_path / solutions_dir.name
                evaluations_dir = evaluations_folder / solutions_dir.name
                evaluation_summary = run_evaluate(solutions_dir, problems_dir, evaluations_dir)
                stats_df, average_row = generate_agent_aggregate_report(agent_name, problems_dir, evaluations_dir)
                stats_df_clean_columns = stats_df.rename(
                    {col_name: col_name.removesuffix(f'_{agent_name}')
                     for col_name in stats_df.columns
                     if col_name.endswith(f'_{agent_name}')}, axis=1
                )

                stats_df_clean_columns.insert(0, 'problem_set', problems_dir.name)  # add problem_set to the stats_df
                stats_df_clean_columns.insert(1, 'agent', agent_name)  # add agent name to the stats_df
                detailed_eval_df = pd.concat([detailed_eval_df, stats_df_clean_columns], ignore_index=True)

                average_row = {key.removesuffix(f'_{agent_name}'): value for key, value in average_row.items()}
                average_row = {
                    'problem_set': problems_dir.name,
                    'n_problems': evaluation_summary.n_problems,
                    'n_solutions': evaluation_summary.n_solutions,
                    'agent': agent_name,
                    'n_evaluations': evaluation_summary.n_evaluations,
                    'n_valid_evaluations': len(stats_df)
                } | average_row

                averages.append(average_row)

    if len(averages) == 0:
        loggers.flow_logger.error('Unable to read/process any of the agents solutions. Aborting.')
        return pd.DataFrame(), pd.DataFrame()

    agent_averages_df = pd.DataFrame(averages)
    if 'problem_set' in agent_averages_df.columns:
        problem_set_col = agent_averages_df.pop('problem_set')
        agent_averages_df.insert(0, 'problem_set', problem_set_col)
        agent_averages_df.sort_values(by=['problem_set', 'agent'], inplace=True)
    else:
        agent_averages_df.sort_values(by=['agent'], inplace=True)

    save_to_path_averages = combined_reports_output_path / 'agent_averages.csv'
    agent_averages_df.to_csv(save_to_path_averages, index=False)
    loggers.eval_logger.info(f'Saved aggregated agent averages to {save_to_path_averages}, shape= {agent_averages_df.shape}')

    # sort detailed_eval_df
    key_func = natsort_keygen()
    if 'problem_set' in detailed_eval_df.columns:
        detailed_eval_df = detailed_eval_df.sort_values(
            by=['problem_set', 'agent', 'problem_name'], key=lambda col: col.map(key_func)
        )
    else:
        detailed_eval_df = detailed_eval_df.sort_values(
            by=['agent', 'problem_name'], key=lambda col: col.map(key_func)
        )

    save_to_path_details = combined_reports_output_path / 'detailed_eval.csv'
    detailed_eval_df.to_csv(save_to_path_details, index=False)
    loggers.eval_logger.info(f'Saved detailed aggregated results to {save_to_path_details}, shape= {detailed_eval_df.shape}')

    return agent_averages_df, detailed_eval_df
