import json

from box import Box


def load_json_file():
    json_path = "config.json"
    with open(json_path, 'r') as f:
        result = json.load(f)
    return result


def get_config():
    return Box(load_json_file())
