from insight_eval.core_classes.solution import Solution, load_solution_from_directory
from insight_eval.readers.json_utils import class_object_from_json_file
from insight_eval.readers.file_utils import read_tables_from_data_dir
from insight_eval.utils import unique_preserving_order
from insight_eval.core_classes.problem import Problem
from insight_eval.logging_config import loggers
import insight_eval.config as config
from pathlib import Path
import traceback
import attrs
import os


readers_logger = loggers.readers_logger
flow_logger = loggers.flow_logger


@attrs.define(eq=False, hash=False)
class ProblemJson:
    target_column: str
    description: str
    name: str
    comments: str
    problem_domain: str = attrs.field(default="")


@attrs.define(eq=False, hash=False)
class SolutionJson:
    enriched_column_names: list[str]
    features_descriptions: list[str]


@attrs.define(eq=False, hash=False)
class CurriculumProblemDirectoryData:
    ds_problem: Problem
    ds_solution: Solution


def read_curriculum_problem_data_directory(directory_path: Path) -> CurriculumProblemDirectoryData:
    problem_json: ProblemJson = class_object_from_json_file(
        directory_path / "problem" / config.JsonFileNames.problem_json_file_name(), ProblemJson
    )

    problem_data = read_tables_from_data_dir(directory_path / "problem" / "data")
    train = problem_data['train.csv']
    test = problem_data['test.csv']
    secondary_data = {}
    for name, df in problem_data.items():
        if name not in ["train.csv", "test.csv"]:
            secondary_data[name] = df

    ds_problem = Problem(
        target_column=problem_json.target_column, description=problem_json.description, train=train, test=test,
        split_method='random', split_col='', secondary_data=secondary_data, name=problem_json.name,
        comments=problem_json.comments, problem_domain=problem_json.problem_domain
    )

    enriched_data = read_tables_from_data_dir(directory_path / "ground_truth" / "data")
    solution_json: SolutionJson = class_object_from_json_file(
        directory_path / "ground_truth" / config.JsonFileNames.solution_json_file_name(), SolutionJson
    )

    enriched_train_df = enriched_data['enriched_train.csv']
    enriched_test_df = enriched_data['enriched_test.csv']
    columns = unique_preserving_order(ds_problem.train.columns.to_list() + solution_json.enriched_column_names)
    columns = [c for c in columns if c in enriched_train_df.columns]

    ds_solution = Solution(
        problem=ds_problem, enriched_train_data=enriched_train_df[columns], enriched_test_data=enriched_test_df[columns],
        solved_by='ground_truth', enriched_column_names=solution_json.enriched_column_names,
        is_ground_truth=True, feature_descriptions=solution_json.features_descriptions
    )

    problem_data_obj = CurriculumProblemDirectoryData(ds_problem, ds_solution)

    return problem_data_obj


def read_problem_and_ground_truth_from_folder(problem_name: str, folder: Path) -> tuple[Problem, Solution]:
    problem_dir = folder / problem_name
    try:  # capture error and return None, None
        problem_data_obj = read_curriculum_problem_data_directory(problem_dir)
        return problem_data_obj.ds_problem, problem_data_obj.ds_solution
    except Exception as e:
        readers_logger.error(f"Failed to load problem {problem_dir}'." + str(e))
        raise RuntimeError(f"Failed to load problem {problem_dir}'." + str(e))


def read_solution(solution_folder: Path, problem: Problem) -> Solution:
    os.makedirs(solution_folder, exist_ok=True)  # solution folder may not exist
    if Path.exists(solution_folder / 'solution_attributes.json'):
        try:  # capture error and throw exception (calling function should handle)
            solution = load_solution_from_directory(solution_folder, problem, is_ground_truth=False)
            flow_logger.info(f'Successfully read Solution from folder {solution_folder}')
            return solution
        except Exception as e:
            readers_logger.error(
                f'Unable to read Solution from folder {solution_folder}, '
                f'error_message={traceback.format_exc()}'
            )
            raise e

    raise Exception(f'Solution_attributes.json is missing in folder {solution_folder}')
