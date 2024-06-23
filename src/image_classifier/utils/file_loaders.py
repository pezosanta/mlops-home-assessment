import json
from pathlib import Path

import yaml


def load_json(file_path: Path) -> dict:
    with open(file=file_path, mode="r") as file:
        return json.load(fp=file)


def load_yaml(file_path: Path) -> dict:
    with open(file=file_path, mode="r") as file:
        return yaml.safe_load(stream=file)
