from insight_eval.logging_config import loggers
from traceback import print_exc, format_exc
from typing import Callable, Any, Optional
import pandas as pd


eval_logger = loggers.eval_logger
running_count = 0


class Function:
    generic_boilerplate = 'from typing import Dict, List, Any\nimport pandas as pd\nimport numpy as np\nimport re'

    def __init__(self, fn_obj: Callable[[pd.Series, dict[str, pd.DataFrame]], Any], code_str: str) -> None:
        self.fn = fn_obj
        self.standalone = code_str
        self.boilerplate = 'from typing import Dict, List, Any\nimport pandas as pd\nimport numpy as np\nimport re'
        self.boilerplate_vectorized = self.boilerplate
        self.name = fn_obj.__name__
        self.body = self.get_fn_body()
        self.used_keys = None  # self.get_used_keys()
        self.acc_type = 'naive'  # other types of code we may generate but currently don't support: vectorization, cython_numba

        self.is_df_train_depended = 'df_train' in self.body.split('):')[0]
        self.is_secondary_dependent = 'aux_data' in self.body.split('):')[0] or 'secondary_data' in self.body.split('):')[0]
        self.secondary_data: dict[str, pd.DataFrame] = {}

        self.target = ''
        self.df_train: pd.DataFrame = pd.DataFrame()
        self.sample_df: pd.DataFrame = pd.DataFrame()

        # self.problem = None
    # end of __init__

    def static_target_leak_check(self) -> bool:
        delims = ['"', "'"]
        patterns = [f'row[{delim}{self.target}{delim}]' for delim in delims]
        patterns += [f'row.get({delim}{self.target}{delim})' for delim in delims]
        if any([pattern in self.body for pattern in patterns]):
            return True
        return False

    def get_fn_body(self) -> str:
        delim_start = f'def {self.name}('
        return delim_start + self.standalone.split(delim_start)[1].split('\ndef ')[0]

    @staticmethod
    def from_code_str(name_str: str, code_str: str, generic_boilerplate: str) -> Optional["Function"]:
        try:
            code_with_boilerplate = generic_boilerplate + '\n\n' + code_str
            exec(code_with_boilerplate)
            f_obj: Callable[[pd.Series, dict[str, pd.DataFrame]], Any] = locals()[name_str]
            return Function(f_obj, code_with_boilerplate)
        except Exception as e:
            eval_logger.error(f'Error executing code string for function {name_str}: {e}')
            print_exc()
            return None

    @staticmethod
    def get_standalone_code(code_str: str, name_str: str) -> str:
        """
        Extracts the standalone code from the provided code string.
        This is a placeholder for the actual implementation that would
        extract the relevant part of the code.
        """
        # Assuming the code_str is a valid Python function definition
        delim_start = f'def {name_str}('
        return code_str.split(delim_start)[1].split('\n\n')[0]

    def enrich_naive(self, df: pd.DataFrame, col_suffix: str = '', target_leak_detection: bool = False) -> pd.DataFrame:
        eval_logger.debug('enriching naive')
        global running_count

        try:
            exec(self.boilerplate)
            exec(self.standalone)
        except Exception as e:
            eval_logger.debug(f"112 - Error:{e}\n{format_exc()}")

        def naive_single_row_fn(row: pd.Series) -> Any:
            match (self.is_df_train_depended, self.is_secondary_dependent):
                case True, True:
                    raise ValueError("Function is not implemented for both df_train and secondary dependencies")
                case True, False:
                    raise ValueError("Function is not implemented for df_train dependency only")
                case False, True:
                    return self.fn(row, self.secondary_data)
                case False, False:
                    return self.fn(row, {})
                case _:
                    return None

        def robust_fn(row: pd.Series) -> Any:
            global running_count

            row = row.copy()
            if running_count >= 100:
                return None
            try:
                val = naive_single_row_fn(row)
                old_val = None

                if target_leak_detection:
                    row_in_train = ((self.df_train.isna() & row.isna()) | (self.df_train == row)).all(axis=1).any()
                    # check if row is in self.df_train, so that we can None-ify the target column there as well
                    if row_in_train:
                        old_val = self.df_train.at[(row.name, self.target)]
                        self.df_train.at[(row.name, self.target)] = None
                    row[self.target] = None
                    val_ = naive_single_row_fn(row)

                    if row_in_train:
                        self.df_train.at[row.name, self.target] = old_val

                    if str(val_)[:10] != str(val)[:10]:
                        raise ValueError('direct target leak detected')

                running_count = 0
                return val

            except Exception as ex:
                self.error_msg = '\n'.join(format_exc().split('val = ')[-1].split('\n')[1:])
                if running_count <= 5:
                    eval_logger.debug(f"Function. naive_enrich() - robust_fn(row) - Error:{ex}")

                if 'target leak detected' in str(ex):
                    raise ValueError(str(ex))

                running_count += 1
                return None  # or some default value

        # The signature gets a row dict and returns a value. We want to add a column to the df with the name of the
        # function, and the value of the function applied to the row

        df[self.name + col_suffix] = df.apply(robust_fn, axis=1)
        running_count = 0
        return df
