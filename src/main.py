from args import get_args_parser
from config import read_config


def main():

    # Parse args
    args = get_args_parser().parse_args()

    # Read config
    config = read_config(args)

    from utils.json_helper import print_json
    print_json(config)


if __name__ == '__main__':
    main()
