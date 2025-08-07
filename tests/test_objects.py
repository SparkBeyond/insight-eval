import pandas as pd
from insight_eval.core_classes.problem import Problem
from insight_eval.core_classes.solution import Solution
from datetime import datetime as dt
from enum import Enum
import numpy as np
from typing import Tuple, Callable, Any


class CategoryEnum(Enum):
    Electronics = 1
    Clothing = 2
    Food = 3
    Books = 4
    Furniture = 5


class TestObjects:
    problem: Problem
    ground_truth: Solution
    solution: Solution

    def __init__(self) -> None:
        train_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6, 7],
            'date': [
                dt.strptime("2023-08-12", "%Y-%m-%d"),
                dt.strptime("2023-08-01", "%Y-%m-%d"),
                dt.strptime("2023-08-23", "%Y-%m-%d"),
                dt.strptime("2023-08-17", "%Y-%m-%d"),
                dt.strptime("2023-08-25", "%Y-%m-%d"),
                dt.strptime("2023-08-27", "%Y-%m-%d"),
                dt.strptime("2023-08-27", "%Y-%m-%d")
            ],
            're_order': [0, 1, 0, 1, 0, 1, 0]
        })

        test_df = pd.DataFrame({
            'id': [11, 12, 13, 14, 15, 16],
            'date': [
                dt.strptime("2023-09-30", "%Y-%m-%d"),
                dt.strptime("2023-09-30", "%Y-%m-%d"),
                dt.strptime("2023-09-30", "%Y-%m-%d"),
                dt.strptime("2023-09-30", "%Y-%m-%d"),
                dt.strptime("2023-09-30", "%Y-%m-%d"),
                dt.strptime("2023-09-30", "%Y-%m-%d")
            ],
            're_order': [1, 0, 0, 1, 1, 0]
        })

        self.problem = Problem(
            problem_domain="Retail",
            name="Orders",
            description="Find drivers for follow up orders (re-order) by customers.",
            target_column="re_order",
            train=train_df,
            test=test_df,
            comments="train data is during 08-2023, target refers to 09-2023.",
            split_method='random_split',
            split_col='n/a',
        )

        gt_feature_descriptions = [
            "number of orders in previous month",
            "most common product category last quarter"]
        gt_feature_column_names = [
            'n_orders',
            'common_category'
            ]

        gt_enriched_train_df = train_df
        gt_enriched_test_df = test_df

        gt_enriched_train_df['n_orders'] = [1, 12, 0, 17, 1, 2, 5]
        gt_enriched_train_df['common_category'] = [
            CategoryEnum.Books.value, CategoryEnum.Furniture.value,
            CategoryEnum.Food.value, CategoryEnum.Books.value, CategoryEnum.Electronics.value,
            CategoryEnum.Furniture.value, CategoryEnum.Books.value
        ]

        gt_enriched_test_df['n_orders'] = [11, 3, 0, 15, 1, 7]
        gt_enriched_test_df['common_category'] = [
            CategoryEnum.Clothing.value, CategoryEnum.Books.value,
            CategoryEnum.Food.value, CategoryEnum.Furniture.value,
            CategoryEnum.Furniture.value, CategoryEnum.Electronics.value
        ]

        self.ground_truth = Solution(
            problem=self.problem,
            enriched_train_data=gt_enriched_train_df,
            enriched_test_data=gt_enriched_test_df,
            enriched_column_names=gt_feature_column_names,
            solved_by='ground truth',
            is_ground_truth=True,
            new_feature_functions=[],  # providing no code
            feature_descriptions=gt_feature_descriptions
        )

        #
        solution_feature_descriptions = ["total amount ordered last month"]
        solution_feature_column_names = ['total_amount_ordered']

        solution_enriched_train_df = train_df
        solution_enriched_test_df = test_df
        solution_enriched_train_df['total_amount_ordered'] = [30, 346, 1, 522, 30, 61, 153]
        solution_enriched_test_df['total_amount_ordered'] = [45, 60, 12, 899, 120, 0]

        self.solution = Solution(
            problem=self.problem,
            enriched_train_data=solution_enriched_train_df,
            enriched_test_data=solution_enriched_test_df,
            enriched_column_names=solution_feature_column_names,
            solved_by='test solution',
            is_ground_truth=False,
            new_feature_functions=[],  # providing no code
            feature_descriptions=solution_feature_descriptions
        )
        pass


class TestObjectsSimple:
    problem: Problem
    ground_truth: Solution
    solution: Solution

    def __init__(self) -> None:
        train_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'date': [
                dt.strptime("2023-08-12", "%Y-%m-%d"),
                dt.strptime("2023-08-01", "%Y-%m-%d"),
                dt.strptime("2023-08-23", "%Y-%m-%d"),
                dt.strptime("2023-08-17", "%Y-%m-%d"),
                dt.strptime("2023-08-25", "%Y-%m-%d")
                ],
            're_order': [0, 1, 0, 1, 0]
        })

        test_df = pd.DataFrame({
            'id': [11, 12, 13, 14],
            'date': [
                dt.strptime("2023-09-30", "%Y-%m-%d"),
                dt.strptime("2023-09-30", "%Y-%m-%d"),
                dt.strptime("2023-09-30", "%Y-%m-%d"),
                dt.strptime("2023-09-30", "%Y-%m-%d")
            ],
            're_order': [1, 0, 0, 1]
        })

        self.problem = Problem(
            problem_domain="Retail",
            name="Orders",
            description="Find drivers for follow up orders (re-order) by customers.",
            target_column="re_order",
            train=train_df,
            test=test_df,
            comments="train data is during 08-2023, target refers to 09-2023.",
            split_method='random_split',
            split_col='n/a',
        )

        gt_feature_descriptions = [
            "number of orders in previous month",
            "most common product category last quarter"]
        gt_feature_column_names = [
            'n_orders',
            'common_category'
            ]

        gt_enriched_train_df = train_df
        gt_enriched_test_df = test_df

        gt_enriched_train_df['n_orders'] = [1, 12, 0, 7, 1]
        gt_enriched_train_df['common_category'] = [
            CategoryEnum.Electronics.value, CategoryEnum.Clothing.value,
            CategoryEnum.Food.value, CategoryEnum.Books.value, CategoryEnum.Electronics.value,
            CategoryEnum.Electronics.value, CategoryEnum.Books.value
        ]

        gt_enriched_test_df['n_orders'] = [9, 3, 0, 15]
        gt_enriched_test_df['common_category'] = [
            CategoryEnum.Clothing.value, CategoryEnum.Electronics.value,
            CategoryEnum.Food.value, CategoryEnum.Books.value,
            CategoryEnum.Electronics.value, CategoryEnum.Books.value
        ]

        self.ground_truth = Solution(
            problem=self.problem,
            enriched_train_data=gt_enriched_train_df,
            enriched_test_data=gt_enriched_test_df,
            enriched_column_names=gt_feature_column_names,
            solved_by='ground truth',
            is_ground_truth=True,
            new_feature_functions=[],  # providing no code
            feature_descriptions=gt_feature_descriptions
        )

        #
        solution_feature_descriptions = ["total amount ordered last month"]
        solution_feature_column_names = ['total_amount_ordered']

        solution_enriched_train_df = train_df
        solution_enriched_test_df = test_df

        solution_enriched_train_df['total_amount_ordered'] = [30, 360, 0, 210, 30]

        solution_enriched_test_df['total_amount_ordered'] = [45, 99, 0, 899]

        self.solution = Solution(
            problem=self.problem,
            enriched_train_data=solution_enriched_train_df,
            enriched_test_data=solution_enriched_test_df,
            enriched_column_names=solution_feature_column_names,
            solved_by='test solution',
            is_ground_truth=False,
            new_feature_functions=[],  # providing no code
            feature_descriptions=solution_feature_descriptions
        )
        pass


class TestObjectsForEvalPerformance:
    problem: Problem
    ground_truth: Solution
    solution: Solution

    def __init__(self, noise_strength: float, train_size: int, test_size: int):
        train_df, test_df, enriched_train_df, enriched_test_df, ground_truth_r2 = TestObjectsForEvalPerformance.gen_two_column_problem(train_size=train_size, test_size=test_size, noise_strength=noise_strength)

        self.noise_strength = noise_strength
        self.ground_truth_r2 = ground_truth_r2

        self.problem = Problem(
            problem_domain="Retail",
            name="Orders",
            description="Find drivers for follow up orders (re-order) by customers.",
            target_column="target",
            train=train_df,
            test=test_df,
            comments="train data is during 08-2023, target refers to 09-2023.",
            split_method='random_split',
            split_col='n/a',
        )

        enriched_feature_descriptions = ["cos(x)", "y^2"]
        enriched_column_names = ['x_cos', 'y_sqr']

        self.solution = Solution(
            problem=self.problem,
            enriched_train_data=enriched_train_df,
            enriched_test_data=enriched_test_df,
            enriched_column_names=enriched_column_names,
            solved_by='ground truth',
            is_ground_truth=True,
            new_feature_functions=[],  # providing no code
            feature_descriptions=enriched_feature_descriptions
        )

        pass

    @staticmethod
    def gen_two_column_problem(train_size: int, test_size: int, noise_strength: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
        size = train_size + test_size

        assert noise_strength >= 0

        x = np.arange(1, size + 1) / size
        y = np.arange(1, size + 1) / (size/2) - 1

        x_cos = np.cos(2*np.pi * x)
        y_sqr = y**2

        z = x_cos + y_sqr

        rms: Callable[[Any], Any] = lambda t: np.sqrt(np.mean(t**2))

        unit_noise = np.random.randn(size)  # standard Gaussian
        rms_z = rms(z)

        target = z + noise_strength * rms_z * unit_noise

        df = pd.DataFrame({
            'x': x,
            'y': y,
            'x_cos': x_cos,
            'y_sqr': y_sqr,
            'target': target
        })

        # Shuffle the rows
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train and test
        train_df = df.iloc[:train_size].copy().reset_index(drop=True)[['x', 'y', 'target']]
        test_df = df.iloc[train_size:].copy().reset_index(drop=True)[['x', 'y', 'target']]

        enriched_train_df = df.iloc[:train_size].reset_index(drop=True)[['x', 'y', 'target', 'x_cos', 'y_sqr']]
        enriched_test_df = df.iloc[train_size:].reset_index(drop=True)[['x', 'y', 'target', 'x_cos', 'y_sqr']]

        # Compute the R^2 with ground-truth knowledge
        # Z_test = Z[df.index[train_size:]]
        # target_test = target[df.index[train_size:]]
        # noise_test = unit_noise[df.index[train_size:]]

        # r-squared with perfect knowledge estimated only on the test part
        # noinspection PyTypeChecker
        ground_truth_r2 = 1 - (rms(z - target) / np.std(target)) ** 2

        return train_df, test_df, enriched_train_df, enriched_test_df, ground_truth_r2


def gen_df_for_tests(size: int) -> pd.DataFrame:
    """
    Generates a DataFrame for testing purposes with specified size.
    The DataFrame will contain various data types and a target column.
    """

    start_date = pd.to_datetime('2026-01-01')
    date_list = [(start_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(size)]

    df = pd.DataFrame({
        'id': range(1, size + 1),
        'num': [10 * i for i in range(1, size + 1)],
        'bool': [True, False] * (size // 2) + ([True] * (size % 2)),
        'float': [0.1 * i for i in range(1, size + 1)],
        'cat1': ['A', 'B', 'C', 'D', 'E'] * (size // 5) + (['A'] * (size % 5)),
        'cat2': ['A' + str(i) for i in range(1, size + 1)],
        'target': [0, 1, 0, 1, 0, 1] * (size // 6) + ([0] * (size % 6)),
        'date': date_list
    })
    return df
