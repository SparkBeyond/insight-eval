from insight_eval.evaluation_framework.evaluate_multiple_agents import evaluate_multiple_agents
from insight_eval.logging_config import loggers
from pathlib import Path


if __name__ == '__main__':
    loggers.set_logging_directory(Path('__file__').parent / 'loggers', delete_existing_results=True)

    data_path = Path(__file__).parent.parent.parent / 'benchmark_curriculum' / 'data'
    # data_path = Path.home() / 'SparkBeyond_Insight_Discovery_Benchmark-V1' / 'data'
    problems_path = data_path / 'problems'
    solutions_path = data_path / 'agents_solutions'
    evaluations_path = data_path.parent / 'agents_evaluations'
    combined_reports_path = evaluations_path / 'combined_reports'

    agents_for_combined_report = ['single step agent', 'two iterations agent']

    evaluate_multiple_agents(
        agents_for_combined_report,
        problems_path,
        solutions_path,
        evaluations_path,
        delete_existing_evaluations=True,
        problem_sets_hierarchy=False,
        combined_reports_output_path=combined_reports_path
    )
