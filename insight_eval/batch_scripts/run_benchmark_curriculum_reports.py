from insight_eval.reports.repo_eval_stats import combined_repo_eval_stats, agent_performance_by_difficulty_level
from insight_eval.reports.repo_stats import create_repo_report
from insight_eval.logging_config import loggers
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    data_path = Path(__file__).parent.parent.parent / 'benchmark_curriculum' / 'data'
    # data_path = Path.home() / 'SparkBeyond_Insight_Discovery_Benchmark-V1' / 'data'

    loggers.set_logging_directory(data_path.parent / 'loggers', delete_existing_results=True)
    reports_path = data_path.parent / 'repo_stats'

    repo_stats_df = create_repo_report(data_path / 'problems')
    detailed_eval_stats_df = pd.read_csv(data_path.parent / 'agents_evaluations' / 'combined_reports' / 'detailed_eval.csv')

    agents_for_difficulty = ['single step agent', 'two iterations agent']
    repo_eval_stats_df = combined_repo_eval_stats(repo_stats_df, detailed_eval_stats_df, agents_for_difficulty, reports_path)

    agent_performance_by_difficulty_level(detailed_eval_stats_df, repo_eval_stats_df, reports_path)

    pass
