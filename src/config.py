import os
from utils.json_helper import read_json


def read_config(args):

    path = os.getcwd() + '/configs/' + args.config
    return read_json(path)
