from __future__ import annotations
from attr.validators import instance_of
from insight_eval import converter
from pathlib import Path
import attrs
import json
import os


@attrs.define(eq=False, hash=False)
class CombinedEvaluationScore:
    inclusive_performance: float
    exclusive_performance: float
    coverage: float
    naive_performance: float
    target_leak: bool
    ground_truth_inclusive_performance: float
    ground_truth_exclusive_performance: float
    combined_score: float = attrs.field(init=False)
    formula: str = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self.combined_score, self.formula = self.combined_evaluation(
            self.inclusive_performance,
            self.coverage,
            self.target_leak
        )

    def to_directory(self, folder: Path, name_prefix: str = '') -> None:
        fn = os.path.join(folder, f'{name_prefix}combined_score.json')
        payload = converter.unstructure(self)
        with open(fn, 'w', encoding="utf-8") as f:
            json.dump(payload, f)

    def summary(self) -> str:
        return f'Combined score: {self.combined_score} \n ({self.formula})'

    @staticmethod
    def from_directory(folder: Path, name_prefix: str = '') -> CombinedEvaluationScore:
        fn = os.path.join(folder, f'{name_prefix}combined_score.json')
        with open(fn, 'r') as f:
            return converter.structure(json.load(f), CombinedEvaluationScore)

    @staticmethod
    def combined_evaluation(inclusive_performance: float, coverage: float, target_leak: bool) -> tuple[float, str]:
        target_leak_score = target_leak if target_leak else 0
        if coverage:
            formula = "0.5 * inclusive_performance + 0.5 * coverage - 1.0 * target_leak_score"
            combined_score = 0.5 * inclusive_performance + 0.5 * coverage - 1.0 * target_leak_score
        else:
            formula = "inclusive_performance - 1.0 * target_leak_score"
            combined_score = inclusive_performance - 1.0 * target_leak_score
        return combined_score, formula
