# Insight Discovery Evaluation Framework

## About insight-eval

This repository contains:
- A benchmark for insight discovery with 197 realistic problems and associated ground-truth insights
- An evaluation framework and code to evaluate agents performing the insight discovery 
- This work expands on our work on benchmarking insight discovery from late 2024, explained in more details [here](https://www.sparkbeyond.ai/articles/benchmarking-llms-insight-discovery-capabilities-through-synthetic-problem-generation)

## Content

The package Insight-Eval provides a library for evaluating agents that perform insight discovery over structured data. 
The package is available from PyPI as insight-eval.

**Repository Structure:**
```
/
├── benchmark_curriculum/           # The benchmark data
│   ├── data/                       # Benchmark problems and solutions by sample agents
│   └── repo_stats_reference/       # Reference statistics
└── insight_eval/                   # Core library code
    ├── evaluation_framework/       # Evaluation algorithms
    ├── core_classes/               # Data models
    ├── readers/                    # Data loading utilities
    ├── batch_scripts/              # Usage demonstration scripts
    ├── reports/                    # Reporting utilities
```

## Core Classes

The framework provides the following core classes:

- `Problem` - Represents a prediction problem with metadata including target column, description, and domain context. Contains all information needed to understand what the automated system should predict and the business context for evaluation.

- `Solution` - Contains executable feature engineering functions that automated systems generate to create new features from raw data. Includes metadata about the solving agent, enriched column names, and human-readable feature descriptions.

- `EvaluationResults` - Complete evaluation results container that aggregates performance metrics, coverage analysis, and combined scoring. Provides a unified interface to access all evaluation outcomes for a single agent-problem pair.

- `CoverageEvaluationResults` - Semantic alignment metrics that measure how well discovered features align with expert-defined ground truth insights. Includes correlation coverage, incremental performance coverage, and predictive coverage scores.

- `PerformanceEvaluationResults` - Predictive performance metrics including inclusive performance (agent + original features), exclusive performance (agent features only), and naive performance (original features only). Enables assessment of the added value from automated insight discovery.

- `CombinedEvaluationScore` - Unified scoring mechanism that combines performance and coverage metrics with configurable weights. Includes target leakage penalty detection and provides an overall assessment of agent quality.

## Primary Function

The primary function is `evaluate()` in `evaluators.py`, which performs complete evaluation of an agent's solution against ground truth insights. This function orchestrates the entire evaluation pipeline including feature enrichment, performance measurement, coverage analysis, and target leakage detection. It returns a comprehensive `EvaluationResults` object containing all metrics needed to assess both predictive capability and semantic alignment with expert knowledge.

## Evaluation Metrics

Evaluation metrics are described in detail in the **Evaluation Metrics** documentation. The framework implements four coverage metrics (correlation, incremental performance, predictive, and single-column predictive coverage) that measure semantic alignment with expert insights, plus traditional performance metrics and target leakage detection. All the code to compute the evaluation metrics is implemented in `evaluators.py`, providing a complete suite of tools to assess whether automated systems truly understand the patterns they discover rather than simply achieving high predictive accuracy. More information about the evaluation metrics can be found [here](insight_eval/evaluation_framework/README.md)

## Batch Scripts

The `batch_scripts` directory provides scripts that demonstrate usage (names are self-explanatory):
- `evaluate_single_problem.py` - Evaluate one problem/solution pair
- `evaluate_agents.py` - Compare multiple agents across problems
- `evaluate_benchmark_curriculum.py` - Run evaluation on full benchmark
- `run_benchmark_curriculum_reports.py` - Generate comprehensive reports

## Readers Package

The `readers` package contains `read_curriculum_problems_and_solution.py` with methods for:
- `read_curriculum_problem_data_directory` - Load problem data from directory structure
- `read_problem_and_ground_truth_from_folder` - Read problem definition and ground truth
- `read_solution` - Load agent solution data

## Reports

The `reports` package provides utility functions to generate reports on the entire benchmark curriculum:
- `repo_stats.py` - Repository statistics utilities
- `repo_eval_stats.py` - Evaluation statistics utilities
- Script for running reports: `batch_scripts/run_benchmark_curriculum_reports.py`

## Configuration

Configuration parameters are set in `config.py`. The library is designed to run with a fixed set of configuration parameters. 
**Avoid changing these parameters at runtime, especially if multiple threads/processes are running.**

## Installation

**For end users** who want to integrate insight evaluation into their projects:

```bash
# Install from PyPI
pip install insight_eval
```

**For developers** who want to contribute to or extend the framework:

```bash
# Clone the repository
git clone https://github.com/SparkBeyond/insight_eval.git
cd insight_eval

# Install dependencies with Poetry
poetry install

# Or install with development dependencies
poetry install --with dev
```

## Evaluate from Command Line

These commands will help you verify your installation and understand the framework's capabilities using the provided sample data:

```bash
# Evaluate a single problem - runs complete pipeline on one problem/solution pair
# Shows all evaluation metrics and detects any target leakage issues
python insight_eval/batch_scripts/evaluate_single_problem.py

# Compare multiple agents - benchmarks different ML systems side-by-side
# Generates statistical comparisons and identifies best-performing approaches  
python insight_eval/batch_scripts/evaluate_agents.py
```

Both scripts use sample data from `insight_eval/batch_scripts/data/` to demonstrate the evaluation process.

## Basic Problem Evaluation

This example demonstrates how to evaluate a single automated ML agent's solution against expert-defined ground truth insights:

```python
from insight_eval.readers.read_curriculum_problems_and_solution import read_problem_and_ground_truth_from_folder, read_solution
from insight_eval.evaluation_framework.evaluators import evaluate
from pathlib import Path

# Load problem and ground truth
problem_path = Path("insight_eval/batch_scripts/data/problems/Ex1")
problem, ground_truth = read_problem_and_ground_truth_from_folder("Ex1", problem_path.parent)

# Load agent solution
solution_path = Path("insight_eval/batch_scripts/data/agents_solutions/agent1/Ex1") 
solution = read_solution(solution_path, problem)

# Perform complete evaluation
evaluation_results = evaluate(problem, solution, ground_truth, solution_path)

# Access results
print(f"Coverage Score: {evaluation_results.coverage_evaluation_results.mean_correlation_coverage}")
print(f"Exclusive Performance: {evaluation_results.performance_evaluation_results.exclusive_performance}")
print(f"Combined Score: {evaluation_results.combined_score.combined_score}")
```

**Note:** More information about the Problem and Solution schemas can be found [here](insight_eval/readers/README.md)

This evaluation produces three key metrics: **Inclusive Performance** (how well the agent + original features predict), **Coverage Score** (how well agent discoveries align with expert insights), and **Combined Score** (balanced assessment of both).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This framework was developed by SparkBeyond as part of their mission to advance automated insight discovery and ethical AI development. We acknowledge the contributions of the research community in developing evaluation methodologies for interpretable machine learning and automated feature engineering.

## Support

For questions, bug reports, or feature requests:
- **GitHub Issues**: [Repository Issues](https://github.com/SparkBeyond/insight_eval/issues)
- **Examples**: Usage patterns and sample data in the `insight_eval/batch_scripts/data/` directory
