from insight_eval import converter
from typing import Any, cast
from pathlib import Path
import json


def class_object_from_json_file[T](json_file_path: Path, cls: type[T]) -> T:
    return converter.structure(dict_from_json_file(json_file_path), cls)


def dict_from_json_file(json_file_path: Path) -> dict[str, Any]:
    try:
        with open(json_file_path, 'r') as file:
            return cast(dict[str, Any], json.load(file))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {json_file_path} not found.")


def read_schema[T](json_file_path: Path, schema: type[T]) -> T:
    return converter.structure(dict_from_json_file(json_file_path), schema)
