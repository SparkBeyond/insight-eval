from insight_eval.evaluation_framework.evaluate_multiple_agents import evaluate_multiple_agents
from insight_eval.logging_config import loggers
from pathlib import Path


if __name__ == '__main__':
    loggers.set_logging_directory(Path('__file__').parent / 'loggers', delete_existing_logs=True)

    delete_existing_evaluations = True  # if True, will delete existing evaluations before running new ones
    batch_data_path = Path(__file__).parent / 'data'
    agents_for_combined_report = ['agent1', 'agent2']

    problems_path = batch_data_path / 'problems'
    solutions_path = batch_data_path / 'agents_solutions'
    evaluations_path = batch_data_path / 'agents_evaluations'

    if Path.exists(solutions_path):
        for only_common_problems in [False, True]:
            combined_reports_path = evaluations_path / ('combined_reports' + ('_only_common_problems' if only_common_problems else ''))

            evaluate_multiple_agents(agents_for_combined_report, problems_path, solutions_path, evaluations_path,
                                     delete_existing_evaluations, only_common_problems=only_common_problems,
                                     problem_sets_hierarchy=False, combined_reports_output_path=combined_reports_path)
    else:
        loggers.eval_logger.error(f'Cannot generate combined report, some solution folders do not exist')
