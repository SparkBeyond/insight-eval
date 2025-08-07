# Evaluation Metrics for Insight Discovery

This document provides a comprehensive specification of the evaluation metrics used in the Insight Discovery Evaluation Framework. These metrics assess automated ML agents across three critical dimensions: coverage (semantic alignment with expert insights), performance (predictive capability), and proper data usage (avoiding target leakage).

## Overview

To evaluate an agent's performance on a given benchmark problem, the framework employs a three-dimensional evaluation approach that goes beyond traditional accuracy metrics to assess whether systems truly understand the patterns they discover.

### Evaluation Dimensions

**Coverage Metrics** assess the agent's ability to discover insights regarding factors that affect the target KPI, measuring semantic alignment with domain expert knowledge through four distinct approaches.

**Performance Metrics** assess the predictive power of the agent's solution using multiple baselines to understand the added value of discovered insights.

**Proper Data Usage** detects possible target leakage introduced by feature functions, ensuring temporal validity and real-world applicability.

## Coverage Metrics

Coverage metrics represent the core innovation of this evaluation framework, directly measuring the relationship between automated discoveries and expert knowledge rather than assuming that high predictive performance indicates good insight discovery.

### Incremental Performance Coverage

Incremental Performance Coverage quantifies the extent to which each ground truth insight is "covered" by the insights identified by the solving agent. It measures the marginal predictive performance that could have been gained by explicitly adding ground truth features to the agent's solution.

**Definition:**
Let `c` be a ground truth insight column, and `S` be the set of insight columns provided by the agent. The Incremental Performance Coverage of `c` given `S` is:

```
1 - max(ρ(Performance(Model(S∪{c})) - Performance(Model(S))), 0)
```

**Implementation Details:**
- **Model**: Random Forest with predefined maximum depth
- **Performance Metric**: 
  - Area Under the ROC Curve (AUC) for classification problems
  - Coefficient of determination (R²) for regression problems
- **Threshold Function**: `ρ(x) = 2*max(x-0.5, 0)`
  - Discards low-quality performance values below 0.5
  - Treats performance below 0.5 as equally poor

**Interpretation:**
If adding ground truth column `c` to the agent's insight set `S` results in measurable improvement in predictive performance (after thresholding), then solution `S` is missing that potential improvement, resulting in lower Incremental Performance Coverage for `c`. Conversely, if adding `c` does not improve performance, the column is considered "covered," yielding a coverage score of 1.

**Aggregation**: The minimum of this metric over all ground truth columns is computed to ensure that all expert insights are adequately covered.

### Predictive Coverage

The Predictive Coverage metric quantifies how well each ground truth factor can be recovered from the insight columns provided by the agent. This metric captures information not reflected by incremental performance coverage, as it focuses on predicting ground truth columns rather than the final target.

**Motivation:**
If all highly influential ground truth columns can be accurately predicted from the solution's insight columns, then the agent has effectively uncovered meaningful semantic relationships that contribute to explaining the target.

**Definition:**
Given a set of solution insight columns `S` and a ground truth insight column `c`, the Predictive Coverage of `c` is:

```
ρ(Performance(Model(S) predicting c))
```

**Average Predictive Coverage** summarizes performance across all ground truth columns using importance weighting:

```
weight(c) = ρ(Performance(Model({c}) predicting target))
```

This weighting ensures that more influential ground truth features have greater impact on the overall score.

**Note**: This metric provides detailed diagnostic information but does not participate in the Combined Coverage Score.

### Single Column Predictive Coverage

Single Column Predictive Coverage extends the predictive coverage concept by constraining the reconstruction to use only individual discovered features rather than all features combined. Due to its stricter constraint, this metric is generally more challenging than its all-column counterpart.

**Motivation:**
1. **Interpretability**: If each highly influential ground truth column `c` can be predicted with high accuracy from a single solution insight column `s∈S`, then the semantic relationships uncovered by the agent are more readily interpretable.

2. **Actionable Insights**: In real-world scenarios where users can control or influence input features, understanding that feature `s` exerts its influence on the target through a mediating factor `c` (present in ground truth) enables actionable intervention.

**Definition:**
Given a set of solution insight columns `S` and a ground truth insight column `c`, the Single Column Predictive Coverage of `c` is:

```
max ρ(Performance(Model({s}) predicting c)), for s∈S
```

**Average Single Column Predictive Coverage** uses the same importance weighting as Predictive Coverage:

```
weight(c) = ρ(Performance(Model({c}) predicting target))
```

### Correlation Coverage

Correlation Coverage serves as a computationally efficient analogue to Single Column Predictive Coverage. Rather than training predictive models, it evaluates Spearman correlation between ground truth and solution columns.

**Important Limitation:**
While high correlation indicates that solution columns are informative with respect to ground truth columns, the converse does not necessarily hold. Low correlation does not imply lack of predictive information, particularly when relationships are non-monotonic and thus not captured by Spearman correlation.

**Definition:**
Given a set of solution insight columns `S` and a ground truth insight column `c`, the Correlation Coverage of `c` is:

```
max SpearmanCorr(c,s), for s∈S
```

**Aggregation**: Weighted average over all ground truth columns, where the weight assigned to column `c` is its Spearman correlation with the target column.

**Note**: This metric provides rapid assessment but does not participate in the Combined Coverage Score due to its limitations in capturing complex relationships.

### Combined Coverage Score

The Combined Coverage Score provides a unified per-problem metric that balances different aspects of semantic alignment.

**Formula:**
```
Combined Coverage = 0.3 × Incremental Performance Coverage + 0.7 × Single Column Predictive Coverage
```

**Rationale:**
- **Incremental Performance Coverage (30%)**: Ensures the agent captures essential predictive information
- **Single Column Predictive Coverage (70%)**: Emphasizes interpretability and clear semantic relationships

This weighting reflects the framework's emphasis on interpretable insight discovery while maintaining predictive validity.

## Performance Metrics

Performance metrics assess the predictive capability of agent-generated insights using multiple baselines to understand their added value.

### Inclusive Performance

Inclusive Performance measures the predictive power using both the original problem features and the agent-generated insight columns, representing real-world deployment scenarios.

**Definition:**
```
Performance(Model(base_features + agent_features) → target)
```

**Implementation:**
- Training data consists of the union of base columns and insight columns
- Model evaluation on test data target values
- Uses Random Forest with consistent hyperparameters across all evaluations

**Interpretation**: This metric reflects the maximum predictive capability when combining human-engineered features with automated discoveries.

### Exclusive Performance

Exclusive Performance measures predictive capability using only the agent-generated insight columns, excluding original problem features.

**Rationale:**
A capable agent should identify and extract the base columns that are informative for predicting the target, effectively promoting them to insight columns. This is particularly important when predictive signals are explicit and do not require complex reasoning.

**Definition:**
```
Performance(Model(agent_features) → target)
```

**Interpretation**: This metric tests whether the agent can independently discover all relevant patterns without relying on pre-engineered features.

### Naive Performance

Naive Performance provides a baseline using only the original problem features, enabling assessment of the added value from automated insight discovery.

**Definition:**
```
Performance(Model(base_features) → target)
```

**Purpose**: Serves as a comparison baseline for measuring improvement achieved through automated insight discovery.

## Proper Data Usage Validation

The framework implements comprehensive target leakage detection to ensure that insights are valid for real-world deployment.

### Target Leakage Detection

Target leakage occurs when feature engineering functions inadvertently use information that wouldn't be available at prediction time, particularly in temporal scenarios.

**Static Analysis:**
- Examines function source code for direct target variable references
- Identifies patterns like `row['target_column']` in feature engineering code
- Flags suspicious data access patterns

**Dynamic Testing:**
- Executes functions under controlled conditions with target information masked
- Compares outputs before and after target masking
- Detects functions whose behavior changes when target access is removed

**Temporal Validation:**
- Ensures feature functions respect temporal constraints
- Validates that auxiliary data access follows proper time ordering
- Prevents use of future information relative to prediction time

### Combined Scoring Integration

Target leakage detection is integrated into the overall evaluation through a penalty mechanism:

```
Combined Score = 0.5 × Inclusive Performance + 0.5 × Coverage - 1.0 × Target Leak Penalty
```

**Target Leak Penalty**: Binary penalty (1.0) applied when target leakage is detected, ensuring that leaked insights receive appropriately low scores regardless of their apparent performance or coverage.

## Configuration and Customization

The evaluation framework provides configurable parameters to adapt to different research questions and deployment scenarios.

### Coverage Metric Weights

Default configuration emphasizes interpretable relationships while maintaining comprehensive coverage assessment:

```python
COVERAGE_METRIC_WEIGHTS = {
    'correlation_coverage': 0.0,
    'incremental_performance_coverage': 0.3,
    'predictive_coverage': 0.0,  # Diagnostic only
    'single_column_predictive_coverage': 0.7
}
```

### Performance Thresholds

**Coverage Correlation Threshold**: `0.15` - Minimum correlation required for meaningful semantic alignment
**Eligibility Threshold**: `0.0` - Ground truth feature eligibility threshold for coverage evaluation

### Model Configuration

**Random Forest Parameters**:
- `N_RF_ESTIMATORS = 100` - Number of trees for consistent evaluation
- `RANDOM_STATE = 42` - Reproducibility seed
- **Maximum Depth**: Predefined to prevent overfitting while maintaining expressiveness

### Data Processing

- **Feature Limits**: `MAX_FEATURES = 20` - Maximum features per solution to ensure focused insight discovery
- **Sampling**: `N_SAMPLES = 5000` - Sample size for fast mode during development
- **Categorical Encoding**: `MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 10` - Threshold for one-hot encoding
