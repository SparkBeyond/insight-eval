
from traceback import print_exc, format_exc

from insight_eval.core_classes.solution import Solution
from insight_eval.logging_config import loggers
from insight_eval.config import EVALUATE_TARGET_LEAK

eval_logger = loggers.eval_logger


def evaluate_target_leak(solution: Solution) -> bool:
    if EVALUATE_TARGET_LEAK:
        try:
            return attempt_evaluate_target_leak(solution)
        except:
            eval_logger.warning('Failed to evaluate target leak')
            print_exc()
            return False
    else:
        return False


def attempt_evaluate_target_leak(solution: Solution) -> bool:  # true = found target leak
    if solution.new_feature_functions is None:
        raise ValueError(f'Solution {solution.name()} has no new feature functions')

    new_feature_functions = [x for x in solution.new_feature_functions]
    problem = solution.problem
    leaky_functions = []
    # static target leak check
    for f in new_feature_functions.copy():
        f.target = problem.target_column
        if f.static_target_leak_check():
            new_feature_functions.remove(f)
            leaky_functions.append(f)
            eval_logger.debug(f'Removed function {f.name}, because of target leak')

    enriched_train_data = problem.train.copy()

    pivot = int(len(enriched_train_data) // 10)
    smple_sz = min(5, len(enriched_train_data))
    smple_df = enriched_train_data[pivot:pivot + smple_sz]
    for f in new_feature_functions.copy():
        # target leak check #
        f.df_train = problem.train.copy()
        try:  # capture exceptions and collect target leaks
            f.enrich_naive(smple_df.copy(), target_leak_detection=True)
        except Exception as e:
            if 'target leak' in str(e):
                # try to salvage leaky function
                f.sample_df = smple_df.copy()
                # new_f, enriched_sample = f.cure_from_target_leak()
                new_feature_functions.remove(f)
                leaky_functions.append(f)
                eval_logger.warning(f'Removed function {f.name}, because of target leak')
            else:
                error_message = format_exc()
                eval_logger.debug(f'Failed to enrich with target leak detection for function {f.name}')
                eval_logger.debug(f'{error_message=}')
                new_feature_functions.remove(f)

    target_leak_bit = leaky_functions != []

    return target_leak_bit
