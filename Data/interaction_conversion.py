import pickle
import os
from pathlib import Path
import constants.consts as consts
import sys
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared_args import add_user_limit_and_samples

# Repo root resolution (works if file is in repo root or subfolder)
REPO_ROOT = Path(__file__).resolve()
REPO_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name in {"baseline", "src", "scripts", "Data"} else REPO_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Convert paths to interactions")
    add_user_limit_and_samples(parser)
    return parser.parse_args()

args = parse_args()
user_limit = args.user_limit
samples = args.samples

def p(*parts) -> Path:
    return REPO_ROOT.joinpath(*parts)


def convert_train_paths_to_interactions(file_name):
    """
    Converts train path file to list of (user, item) interaction tuples
    """
    pos_interactions = []
    neg_interactions = []

    data_path = p('Data', consts.DATASET_DOMAIN, 'processed_data',
                  consts.PATH_DATA_DIR, file_name)

    with open(data_path, 'r') as f:
        for line in f:
            interaction = eval(line.rstrip("\n"))
            marker = interaction[1]

            path_tuple = interaction[0][0]
            length = path_tuple[-1]
            user = path_tuple[0][0][0]
            item = path_tuple[0][length - 1][0]

            if marker == 1:
                pos_interactions.append((user, item))
            elif marker == 0:
                neg_interactions.append((user, item))

    return pos_interactions, neg_interactions


def convert_test_paths_to_interactions(file_name):
    """
    Converts test path file to list of (user, item) interaction tuples
    """
    pos_interactions = []
    neg_interactions = []

    data_path = p('Data', consts.DATASET_DOMAIN, 'processed_data',
                  consts.PATH_DATA_DIR, file_name)

    with open(data_path, 'r') as f:
        for line in f:
            interactions = eval(line.rstrip("\n"))
            for interaction in interactions:
                marker = interaction[1]

                path_tuple = interaction[0][0]
                length = path_tuple[-1]
                user = path_tuple[0][0][0]
                item = path_tuple[0][length - 1][0]

                if marker == 1:
                    pos_interactions.append((user, item))
                elif marker == 0:
                    neg_interactions.append((user, item))

    return pos_interactions, neg_interactions


def save_interactions(interactions, out_dir, file_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / file_name, 'wb') as f:
        pickle.dump(interactions, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    """
    Convert paths to interaction tuples for baseline evaluation
    """

    # user_limit = 1000

    test_pos, test_neg = convert_test_paths_to_interactions(
        f"test_full_{user_limit}_samples{samples}_interactions.txt"
    )

    out_dir = p('Data', 'BPR', consts.DATASET_DOMAIN)

    save_interactions(test_pos, out_dir, 'full_test_pos_interactions.txt')
    save_interactions(test_neg, out_dir, 'full_test_neg_interactions.txt')


if __name__ == "__main__":
    main()
