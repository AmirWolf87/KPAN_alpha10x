import pickle
import numpy as np
from random import randint
import os
import consts as consts

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

    # pickle to python2 format
    pickle.dump(bpr_matrix, open(os.path.join(f'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',
                                              f"bpr_matrix_test_{subnetwork}.pkl"),"wb"), protocol=2)


def main():
    subnetwork = 'full'
    with open(os.path.join(f'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',f'{subnetwork}_test_pos_interactions.txt'), 'rb') as handle:
        test_pos_inv_org = pickle.load(handle)
    with open(os.path.join(f'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',f"{subnetwork}_test_neg_interactions.txt"), 'rb') as handle:
        test_neg_inv_org = pickle.load(handle)

    convert_for_bpr(test_pos_inv_org, test_neg_inv_org, subnetwork)

if __name__ == "__main__":
    main()
