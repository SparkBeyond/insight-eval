from insight_eval.readers.json_utils import dict_from_json_file
from insight_eval.core_classes.function import Function
from insight_eval.core_classes.problem import Problem
from insight_eval.logging_config import loggers
import insight_eval.config as config
from pathlib import Path
import pandas as pd
import textwrap
import attrs


flow_logger = loggers.flow_logger
readers_logger = loggers.readers_logger


def compress_high_cat_columns(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              columns: list[str], max_nvalues: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in columns:
        assert not pd.api.types.is_numeric_dtype(train_df[col].dtype)
        value_counts = train_df[col].value_counts()
        if len(value_counts) > max_nvalues:
            top_values = value_counts.nlargest(max_nvalues - 1).index
            train_df[col] = train_df[col].where(train_df[col].isin(top_values), '_misc')
            if test_df is not None:
                test_df[col] = test_df[col].where(test_df[col].isin(top_values), '_misc')
    return train_df, test_df


@attrs.define(eq=False, hash=False)
class Solution:
    problem: Problem
    enriched_train_data: pd.DataFrame
    enriched_test_data: pd.DataFrame
    enriched_column_names: list[str]
    solved_by: str
    is_ground_truth: bool = False
    new_feature_functions: list[Function] | None = None  # when we produce from a list of functions
    sorted_feature_functions: dict[float, dict[str, str]] | None = None  # when we produce from a list of functions
    feature_descriptions: list[str] | None = None  # additional information about the features if available

    def __attrs_post_init__(self) -> None:
        self.new_feature_functions = self.new_feature_functions if self.new_feature_functions is not None else []
        self.sorted_feature_functions = self.sorted_feature_functions if self.sorted_feature_functions is not None else {}
        self.feature_descriptions = self.feature_descriptions if self.feature_descriptions is not None else []
        self.ensure_target_column_in_enriched_data()
        self.extract_consistent_enriched_data(config.MAX_FEATURES)

        if len(self.new_feature_functions) == 0:
            flow_logger.debug(f'Solution.__init__(): name={self.name()} new_feature_functions is empty, populate from sorted_feature_functions (dict)')
            # sorted_feature_functions is a dict[float, dict[str, str]], with {'score': float, {'name': str, 'code': str}
            for score, func_dict in sorted(self.sorted_feature_functions.items(), key=lambda x: x[0], reverse=True):
                if 'name' in func_dict and 'code' in func_dict:
                    name_str = func_dict['name']
                    code_str = func_dict['code']
                    new_function = Function.from_code_str(name_str, code_str, Function.generic_boilerplate)
                    if new_function is not None:
                        self.new_feature_functions.append(new_function)
                        flow_logger.debug(f'Solution.__init__(): problem_name={self.problem.name} added new function {new_function.name} from sorted_feature_functions')
                    else:
                        flow_logger.warning(f'Solution.__init__(): problem_name={self.problem.name} failed to create Function from sorted_feature_functions item {func_dict}, skipping')
                        continue
                else:
                    flow_logger.warning(f'Solution.__init__(): problem_name={self.problem.name} sorted_feature_functions item {func_dict} does not have "name" or "code" keys, skipping')
                    continue
        else:
            # do nothing, new_feature_functions was populated in __init__ parameter
            flow_logger.debug(f'Solution.__init__(): name={self.name()} new_feature_functions is populated via init parameter')


    def name(self) -> str:
        problem_name: str = self.problem.name if self.problem and type(self.problem) == Problem else 'solution name n/a: problem not provided'  # noQA
        return problem_name + ' solved by ' + self.solved_by

    def ensure_target_column_in_enriched_data(self) -> None:
        target_column = self.problem.target_column
        if target_column not in self.enriched_train_data.columns:
            readers_logger.warning(f'Target column {target_column} is not in enriched train data, adding it from problem')
            self.enriched_train_data[target_column] = self.problem.train[target_column]
        else:
            target_series_train = self.enriched_train_data[target_column]
            if not pd.api.types.is_numeric_dtype(target_series_train.dtype) and not pd.api.types.is_bool_dtype(target_series_train.dtype):
                readers_logger.warning(f'Target column {target_column} in enriched train data is not numeric or boolean, take from problem.train')
                self.enriched_train_data[target_column] = self.problem.train[target_column]

        if target_column not in self.enriched_test_data.columns:
            readers_logger.warning(f'Target column {target_column} is not in enriched test data, adding it from problem')
            self.enriched_test_data[target_column] = self.problem.test[target_column]
        else:
            target_series_test = self.enriched_test_data[target_column]
            if not pd.api.types.is_numeric_dtype(target_series_test.dtype) and not pd.api.types.is_bool_dtype(target_series_test.dtype):
                readers_logger.warning(f'Target column {target_column} in enriched test data is not numeric or boolean, take from problem.test')
                self.enriched_test_data[target_column] = self.problem.test[target_column]


    def extract_consistent_enriched_data(self, max_features: int) -> None:
        orig_enriched_column_names = self.enriched_column_names.copy()

        if len(self.enriched_column_names) == 0:
            raise ValueError(f'Solution.extract_consistent_enriched_data(), problem={self.problem.name}: enriched_column_names is empty, skipping solution')
        elif len(self.enriched_column_names) > max_features:
            # we have more features than allowed, truncate - use sorted_feature_functions is exist, and if not - just truncate
            if self.sorted_feature_functions is not None and len(self.sorted_feature_functions) > 0:
                # sort by the first key in the sorted_feature_functions
                sorted_features_indices: list[tuple[float, dict[str, str]]] = sorted(self.sorted_feature_functions.items(), key=lambda x: x[0], reverse=True)
                sorted_names = [func_dict['name'] for score, func_dict in sorted_features_indices]

            else:
                sorted_names = self.enriched_column_names

            dropped_names = sorted_names[max_features:]
            self.enriched_column_names = sorted_names[:max_features]
            self.enriched_train_data.drop(columns=dropped_names, inplace=True, errors='ignore')
            self.enriched_test_data.drop(columns=dropped_names, inplace=True, errors='ignore')
            readers_logger.warning(f'More than {max_features} enriched columns names, truncating ({dropped_names}) and continue processing')

        problem_train_columns = self.problem.train.columns.to_list()
        problem_test_columns = self.problem.test.columns.to_list()
        if problem_train_columns != problem_test_columns:
            readers_logger.error(f'Solution-{self.name()}: problem train and test columns are not the same, train={problem_train_columns}, test={problem_test_columns}')
            readers_logger.error(f'{self.enriched_column_names} after adjustment attempts={self.enriched_column_names}, original={orig_enriched_column_names}')
            raise ValueError(f'Problem train and test columns are not the same, train={problem_train_columns}, test={problem_test_columns}')

        correction_required = False
        correct_enriched_columns = problem_train_columns + self.enriched_column_names
        if set(self.enriched_train_data.columns) != set(correct_enriched_columns):
            readers_logger.warning(f'Enriched train data columns are not consistent with problem train columns and enriched column names, train={self.enriched_train_data.columns}, correct={correct_enriched_columns}')
            readers_logger.warning('Attempting automatic fix by extracting consistent columns')
            correction_required = True

        if set(self.enriched_test_data.columns) != set(correct_enriched_columns):
            readers_logger.warning(f'Enriched test data columns are not consistent with problem test columns and enriched column names, test={self.enriched_test_data.columns}, correct={correct_enriched_columns}')
            readers_logger.warning('Attempting automatic fix by extracting consistent columns')
            correction_required = True

        if correction_required:
            if all(col in self.enriched_train_data.columns for col in correct_enriched_columns) and \
               all(col in self.enriched_test_data.columns for col in correct_enriched_columns):
                readers_logger.warning('Automatic fix is possible')
                self.enriched_train_data = self.enriched_train_data[correct_enriched_columns]
                self.enriched_test_data = self.enriched_test_data[correct_enriched_columns]
            else:
                readers_logger.error(f'Solution.extract_consistent_enriched_data(), problem={self.problem.name}: automatic fix is not possible, please check the enriched data')
                raise ValueError('Enriched train and test data columns are not consistent with problem train and test columns and enriched column names')

        # check and fix size
        if len(self.enriched_train_data) != len(self.problem.train):
            readers_logger.warning(f'Warning: solution train data size {len(self.enriched_train_data)} does not match problem train data size {len(self.problem.train)}')
            readers_logger.warning(f'Truncating enriched train data')
            self.enriched_train_data = self.enriched_train_data.iloc[:len(self.problem.train)]

        if len(self.enriched_test_data) != len(self.problem.test):
            readers_logger.warning(f'Warning: solution test data size {len(self.enriched_test_data)} does not match problem test data size {len(self.problem.test)}')
            readers_logger.warning(f'Truncating enriched test data')
            self.enriched_test_data = self.enriched_test_data.iloc[:len(self.problem.test)]

        return

    # =================
    # helper method
    # =================

    def summary(self, brief: bool = False) -> str:
        lines = [self.name(), self.solved_by, {'GroundTruth' if self.is_ground_truth else 'Solution'},
                 f'{self.enriched_column_names=}']

        if not brief:
            if self.feature_descriptions:
                lines.append(f'Feature_descriptions={self.feature_descriptions}')
            if self.sorted_feature_functions:
                lines.append(f"{self.sorted_feature_functions=}")

        return textwrap.dedent(f"""
                {'\n'.join(map(str, lines))}
    
                Enriched_train=
                {self.enriched_train_data.head().to_string()}
    
                Enriched_test=
                {self.enriched_test_data.head().to_string()}
            """
        )


def load_solution_from_directory(folder: Path, problem: Problem, is_ground_truth: bool) -> Solution:
    solution_attributes = dict_from_json_file(folder / 'solution_attributes.json')
    enriched_train = pd.read_csv(folder / 'enriched_train.csv')
    enriched_test = pd.read_csv(folder / 'enriched_test.csv')

    solution = Solution(
        problem=problem,
        enriched_train_data=enriched_train,
        enriched_test_data=enriched_test,
        enriched_column_names=solution_attributes['enriched_column_names'],
        solved_by=solution_attributes['solved_by'],
        is_ground_truth=is_ground_truth,
        new_feature_functions=solution_attributes['new_feature_functions'],
        sorted_feature_functions=solution_attributes['sorted_feature_functions'],
    )
    return solution
