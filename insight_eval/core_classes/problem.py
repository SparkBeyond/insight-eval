from enum import Enum, auto
import pandas as pd
import numpy as np
import attrs


class ProblemType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    MULTICLASS = auto()


@attrs.define(eq=False, hash=False)
class Problem:
    target_column: str
    description: str
    train: pd.DataFrame
    test: pd.DataFrame
    split_method: str
    split_col: str
    secondary_data: dict[str, pd.DataFrame] = dict()
    name: str = attrs.field(default="")
    comments: str = attrs.field(default="")
    problem_domain: str = attrs.field(default="")
    problem_type: ProblemType = attrs.field(init=False)


    def __attrs_post_init__(self) -> None:
        self.problem_type: ProblemType = get_problem_type(self.train, self.target_column, self.test)

def get_problem_type(df: pd.DataFrame, target: str, df2: pd.DataFrame | None = None) -> ProblemType:
    # Note that the problems in this benchmark version are all classification problems.
    # We still need to handle predictive‐coverage cases where the ground‐truth column is numeric.
    # We currently don’t support (and don’t generate) problems with categorical string columns in the ground‐truth.
    assert pd.api.types.is_numeric_dtype(df[target]) or pd.api.types.is_bool_dtype(df[target])

    if df2 is not None:
        concat_df = pd.concat([df, df2], ignore_index=True, axis=0, sort=False)
    else:
        concat_df = df.copy()

    # Coerce booleans to integers; otherwise drop NaNs
    if pd.api.types.is_bool_dtype(concat_df[target]):
        t_values = concat_df[target].fillna(0).astype(int)
    else:
        t_values = concat_df[target].fillna(0)

    unique_values = np.unique(t_values)

    # below only Classification and Regression are supported
    # in the future, we will support categorical target columns as well.
    # for example, if all values are integers (can be int+0.0) and range/len > 0.75 → multiclass classification
    if set(unique_values).issubset({0, 1}):
        # 1. All values are 0 or 1 → binary classification
        return ProblemType.CLASSIFICATION
    else:
        # 2. Otherwise ProblemType → regression
        return ProblemType.REGRESSION


def is_date(col_series: pd.Series) -> bool:
    datetime_strs = ['datetime', 'date', 'time']
    return any([x in str(col_series.dtype) for x in datetime_strs])


def identify_temporal_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if is_date(df[col]):
            return col
    return None
