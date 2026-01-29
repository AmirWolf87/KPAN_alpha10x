import pickle
import numpy as np
from random import randint
import os
import constants.consts as consts
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def p(*parts) -> Path:
    """Build repo-relative paths."""
    return REPO_ROOT.joinpath(*parts)
def convert_for_bpr(pos_list, neg_list, subnetwork):
    '''
    converts pos/neg usersong pair lists into a matrix where every row contains
    101 tuples with the format ((user, song), 1 or 0)
    each row has 1 positive interactions and 100 negative interactions
    '''
    bpr_matrix = []
    total_row = len(pos_list)
    one_percent = total_row//100
    percent = 0
    pos_count = 0
    neg_count = 0
    len_neg = len(neg_list)
    # Don't pop. Just iterate.
    for tuple in pos_list:
        row = []
        for i in range(4):
            neg_interaction = neg_list[neg_count]
            row.append((neg_interaction, 0))
            neg_count += 1
            print(neg_count)
        pos_interaction = pos_list[pos_count]
        row.insert(randint(0, 99), (pos_interaction, 1)) # randomly insert the positive interaction
        bpr_matrix.append(row)
        pos_count += 1
        # if pos_count % (total_row//100) == 0:
        #     percent += 1
        #     print(percent, ' percent done')

    # pickle to python2 format
    out_dir = p('Data', 'BPR', consts.DATASET_DOMAIN)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"bpr_matrix_test_{subnetwork}.pkl", "wb") as f:
        pickle.dump(bpr_matrix, f, protocol=2)


def main():
    subnetwork = 'full'
    with open(p('Data', 'BPR', consts.DATASET_DOMAIN,
                f'{subnetwork}_test_pos_interactions.txt'), 'rb') as handle:
        test_pos_inv_org = pickle.load(handle)
    with open(p('Data', 'BPR', consts.DATASET_DOMAIN,
                f"{subnetwork}_test_neg_interactions.txt"), 'rb') as handle:
        test_neg_inv_org = pickle.load(handle)

    convert_for_bpr(test_pos_inv_org, test_neg_inv_org, subnetwork)

if __name__ == "__main__":
    main()
