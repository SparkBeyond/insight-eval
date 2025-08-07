from insight_eval.logging_config import loggers
from traceback import format_exc
from natsort import natsorted
from pathlib import Path
import pandas as pd


readers_logger = loggers.readers_logger

total_prompt_length = 0
dirs_files_to_exclude = ['.DS_Store']


def list_directory_items(dir_path: Path, files_only: bool = False, dirs_only: bool = False,
                         exclude_list: list[str] | None = None, missing_ok: bool = False) -> list[Path]:
    if missing_ok and not Path.exists(dir_path):
        return []

    items_list = []
    exclude_list = dirs_files_to_exclude if not exclude_list else exclude_list
    sorted_iter = natsorted(Path(dir_path).iterdir())
    for item in sorted_iter:
        if ((item.is_file() and (not dirs_only)) or (item.is_dir() and not files_only)) and item.name not in exclude_list:
            items_list.append(item)

    return items_list


def is_directory_empty(directory: Path) -> bool:
    return not any(directory.iterdir())


def read_tables_from_data_dir(data_path: Path) -> dict[str, pd.DataFrame]:
    """
    Read CSV files from a specified directory into a dictionary of DataFrames.

    Args:
        data_path (Path): The path to the directory containing CSV files.

    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping file names to their corresponding DataFrames.
    """
    dataframes_dict = {}
    data_files = list_directory_items(data_path, files_only=True)

    # Filter for CSV files
    csv_files = [f for f in data_files if f.suffix.lower() == '.csv']

    for data_file in csv_files:
        try:  # recover and return empty dict
            df = pd.read_csv(data_file, low_memory=False)
            dataframes_dict[data_file.name] = df
        except Exception as e:
            error_message = format_exc()
            readers_logger.error(f'!!! *** Error reading {data_file.name}: {e}\n{error_message} *** !!!')

    return dataframes_dict
