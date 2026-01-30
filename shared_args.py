import argparse

def add_user_limit_and_samples(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--user_limit',
        type=int,
        default=25000,
        help='max number of users to find paths for'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='number of paths to sample for each interaction (-1 means include all paths)'
    )