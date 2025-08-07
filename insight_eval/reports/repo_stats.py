from insight_eval.readers.read_curriculum_problems_and_solution import read_problem_and_ground_truth_from_folder
from insight_eval.readers.file_utils import list_directory_items
from insight_eval.logging_config import loggers
from traceback import format_exc
from pathlib import Path
import pandas as pd
import re


flow_logger = loggers.flow_logger


def parse_problem_file_name(problem_file_name: str) -> dict[str, str]:
    pattern = r'^(?P<problem_id>[^-]+)-(?P<problem_name>.+)-variation_(?P<variation>\d+)-type_(?P<primary_target_type>\d)(?P<secondary_target_type>[a-zA-Z])$'
    match = re.match(pattern, problem_file_name)
    if not match:
        raise ValueError(f'Invalid problem file name format: {problem_file_name}')

    d = match.groupdict()

    # primary_target_type: 1 for Logistic, 2 for Linear
    # secondary_target_type: a for balanced, b for biased
    match d['primary_target_type']:
        case '1': d['primary_target_type'] = 'Logistic'
        case '2': d['primary_target_type'] = 'Linear'
        case _:  d['primary_target_type'] = 'Invalid'

    match d['secondary_target_type']:
        case 'a': d['secondary_target_type'] = 'Balanced'
        case 'b': d['secondary_target_type'] = 'Weakly Biased'
        case 'c': d['secondary_target_type'] = 'Strongly Biased'
        case _: d['secondary_target_type'] = 'Invalid'

    return d


def construct_problem_file_name(problem_id: str, problem_name: str, variation: str, primary_target_type: str, secondary_target_type: str) -> str:
    match primary_target_type:
        case 'Logistic':
            primary_target_char = '1'
        case 'Linear':
            primary_target_char = '2'
        case _:
            raise ValueError(f"Unknown primary_target_type: {primary_target_type}")

    match secondary_target_type:
        case 'Balanced':
            secondary_target_char = 'a'
        case 'Weakly Biased':
            secondary_target_char = 'b'
        case 'Strongly Biased':
            secondary_target_char = 'c'
        case _:
            raise ValueError(f"Unknown secondary_target_type: {secondary_target_type}")

    out = f'{problem_id}-{problem_name}-variation_{variation}-type_{primary_target_char}{secondary_target_char}'
    return out


def create_repo_report(problems_path: Path) -> pd.DataFrame:
    flow_logger.info(f'problems_path={problems_path}')

    problem_dirs = list_directory_items(problems_path, dirs_only=True)

    columns = [
        'problem_identifier', 'problem_id', 'problem_name', 'variation', 'primary_target_type', 'secondary_target_type', 'problem_domain',
        'target', 'n_tables', 'n_primary_columns', 'total_columns', 'n_train_rows', 'n_test_rows', 'n_gt_insights'
    ]

    problem_lines = []

    for problem_dir in problem_dirs:
        try:
            problem, ground_truth = read_problem_and_ground_truth_from_folder(problem_dir.name, problem_dir.parent)
            flow_logger.debug(f'Processing problem directory: {problem_dir}')
            problem_name_dict = parse_problem_file_name(problem_dir.name)
            problem_data_dict = {
                'problem_domain': problem.problem_domain,
                'target': problem.target_column,
                'n_tables': len(problem.secondary_data) + 1,
                'n_primary_columns': len(problem.train.columns),
                'total_columns': sum(
                    [len(secondary_table.columns) for secondary_table in problem.secondary_data.values()],
                    start=len(problem.train.columns)
                ),
                'n_train_rows': len(problem.train),
                'n_test_rows': len(problem.test),
                'n_gt_insights': len(ground_truth.enriched_column_names)
            }
            problem_lines.append({'problem_identifier': problem_dir.name} | problem_name_dict | problem_data_dict)
        except:
            flow_logger.error(f'*** Error reading problem directory {problem_dir}:\n {format_exc()}\n ** skipping problem **')
            continue

    problems_df = pd.DataFrame(problem_lines)
    assert problems_df.columns.tolist() == columns, f'Columns mismatch: {problems_df.columns.tolist()} != {columns}'

    return problems_df
