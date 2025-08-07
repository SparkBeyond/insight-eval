from __future__ import annotations
from insight_eval import converter
from pathlib import Path
import numpy as np
import attrs
import json
import os


@attrs.define(eq=False, hash=False)
class PerformanceEvaluationResults:
    solution_name: str
    inclusive_performance: float
    exclusive_performance: float
    naive_performance: float


    def name(self) -> str:
        return self.solution_name

    def __str__(self) -> str:
        return (
            f"SolutionPerformanceEvaluationResults("
            f"{self.solution_name=}, "
            f"{self.inclusive_performance=}, "
            f"{self.exclusive_performance=}, "
            f"{self.naive_performance=})"
        )

    def to_directory(self, folder: Path, name_prefix: str = '') -> None:
        def convert_np_types(obj: object) -> bool:
            if isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        fn = os.path.join(folder, f'{name_prefix}performance.json')
        payload = converter.unstructure(self)
        with open(fn, 'w', encoding="utf-8") as f:
            json.dump(payload, f, default=convert_np_types)

    @staticmethod
    def from_directory(folder: Path, name_prefix: str = '') -> PerformanceEvaluationResults | None:
        fn = os.path.join(folder, f'{name_prefix}performance.json')
        if not os.path.exists(fn):
            return None
        with open(fn, 'r') as f:
            d = json.load(f)
            if 'solution_name' not in d:
                d['solution_name'] = 'n/a'
            return converter.structure(d, PerformanceEvaluationResults)


    def summary(self) -> str:
        name = self.solution_name
        return (f'\nPerformance evaluation(for solution={name}):\n   performance: {self.inclusive_performance}, naive performance: {self.naive_performance}, \n   '
                f'improvement: {self.inclusive_performance-self.naive_performance}')
