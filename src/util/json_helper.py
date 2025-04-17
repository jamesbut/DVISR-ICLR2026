import json
import copy
from mergedeep import merge


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def print_json(j):
    print(json.dumps(j, indent=4))


# Recursively add key(s) value pair to dictionary.
# If key(s) value pair is already in dictionary it is replaced with the new
# value.
def add_to_dict(keys, value, dictionary):
    d = build_dict(copy.deepcopy(keys), value)
    merge(dictionary, d)
    return dictionary


# Build dictionary from keys and value as a nested dictionary
def build_dict(keys, value):
    keys.reverse()
    d = {}
    for i, k in enumerate(keys):
        if i == 0:
            d = {k: value}
        else:
            d = {k: d}
    return d
