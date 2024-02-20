from args import get_args_parser
from config import read_config
from domains.domain_factory import create_domain


def main():

    # Parse args
    args = get_args_parser().parse_args()

    # Read config
    config = read_config(args)

    from utils.json_helper import print_json
    print_json(config)

    # Create domain
    domain = create_domain(config['domain'])

    domain_data = domain.create_data()
    print(domain_data)


if __name__ == '__main__':
    main()
