import json


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def print_json(j):
    print(json.dumps(j, indent=4))
