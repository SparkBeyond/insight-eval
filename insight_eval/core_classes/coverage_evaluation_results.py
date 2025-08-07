from __future__ import annotations
from attr.validators import optional, instance_of
from collections import UserDict
from insight_eval import converter
from pathlib import Path
from typing import Self
import numpy as np
import attrs
import json
import os


class DetailedCoveragesDictionary(UserDict[str, dict[str, float]]):
    def get_ground_truth_column_coverage(self, column_name: str) -> dict[str, float]:
        return self.get(column_name, dict())

    def get_correlation_coverage(self, ground_truth_column_name: str, solution_column_name: str) -> float:
        return self.get_ground_truth_column_coverage(ground_truth_column_name).get(solution_column_name, 0.0)


class CoveragesDictionary(UserDict[str, float]):
    def get_ground_truth_column_coverage(self, column_name: str) -> float:
        return self.data.get(column_name, 0.0)


@attrs.define(eq=False, hash=False)
class CoverageEvaluationResults:
    solution_name: str
    reference_name: str
    mean_correlation_coverage: float | None = attrs.field(default=None, validator=optional(instance_of(float)))
    mean_single_column_predictive_coverage: float | None = attrs.field(default=None, validator=optional(instance_of(float)))
    min_incremental_performance_coverage: float | None = attrs.field(default=None, validator=optional(instance_of(float)))
    mean_predictive_coverage: float | None = attrs.field(default=None, validator=optional(instance_of(float)))
    correlation_coverage_weights: CoveragesDictionary = attrs.field(factory=CoveragesDictionary)
    incremental_performance_coverages: CoveragesDictionary = attrs.field(factory=CoveragesDictionary)
    predictive_coverages: CoveragesDictionary = attrs.field(factory=CoveragesDictionary)
    predictive_coverage_weights: CoveragesDictionary = attrs.field(factory=CoveragesDictionary)
    single_column_predictive_coverage_weights: CoveragesDictionary = attrs.field(factory=CoveragesDictionary)
    single_column_predictive_coverages: DetailedCoveragesDictionary = attrs.field(factory=DetailedCoveragesDictionary)
    correlation_coverages: DetailedCoveragesDictionary = attrs.field(factory=DetailedCoveragesDictionary)

    @classmethod
    def invalid_coverage_evaluation_results(cls, solution_name: str, reference_name: str) -> Self:
        return cls(
            solution_name, reference_name,
            mean_correlation_coverage=np.nan,
            min_incremental_performance_coverage=0.0,
            mean_predictive_coverage=0.0,
            mean_single_column_predictive_coverage=0.0
        )

    def name(self) -> str:
        return self.solution_name

    def to_directory(self, folder: Path, name_prefix: str = '') -> None:
        fn = os.path.join(folder, f'{name_prefix}coverage.json')
        payload = converter.unstructure(self)
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def from_directory(folder: Path, name_prefix: str = '') -> CoverageEvaluationResults | None:
        fn = os.path.join(folder, f'{name_prefix}coverage.json')
        if not os.path.exists(fn):
            return None

        with open(fn, 'r') as f:
            data = json.load(f)
            return converter.structure(data, CoverageEvaluationResults)


    def summary(self, detailed: bool = False, names: bool = True) -> str:
        name = f'Name: {self.solution_name} ' if names else ''
        ref_name = f'reference = {self.reference_name} (GroundTruth)' if names else ''
        s1 = f'Mean Correlation Coverage: {self.mean_correlation_coverage}'
        s2 = f'Minimal Incremental Performance Coverage: {self.min_incremental_performance_coverage}'
        s3 = f'Mean Predictive Coverage: {self.mean_predictive_coverage}'
        s4 = f'Mean Single Column Predictive Coverage : {self.mean_single_column_predictive_coverage}'
        s = f'{name}\n{ref_name}\n  {s1}\n  {s2}\n  {s3}\n  {s4}'
        if detailed and self.correlation_coverages is not None:
            for k, v in self.correlation_coverages.items():
                gt_importance = self.correlation_coverage_weights[k] if k in self.correlation_coverage_weights else -1
                covered_by, coverage_strength = list(v.items())[0]   # V is a dictionary with key size of 1
                s += f'\nground_truth_feature: {k} (importance = {gt_importance}) covered by feature: {covered_by}, coverage strength = {coverage_strength}'

        return s
