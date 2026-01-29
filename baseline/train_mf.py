import numpy as np
from scipy.io import mmread, mmwrite
from sklearn.model_selection import train_test_split
from scipy import sparse
from baseline.bpr import *
# import cPickle
import pickle
import numpy as np
import scipy.sparse as sp
import random
import sys
import os
import argparse
import pickle
sys.path.append('..')
from tqdm import tqdm

from eval import hit_at_k,ndcg_at_k,prep_for_evaluation,aggregate_results,calc_scores_per_user
import constants.consts as consts
from pathlib import Path
import os

# baseline/train_mf.py -> repo root is one level up from baseline/
REPO_ROOT = Path(__file__).resolve().parents[1]

def p(*parts) -> Path:
    """Build paths relative to the repo root (or RUN_ROOT if set)."""
    return REPO_ROOT.joinpath(*parts)


# maps the indices in the kprn data to the matrix indices here
kprn2matrix_user = {}
kprn2matrix_song = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",
                        default=0.000001,
                        # default=0.000001
                        help="learning rate",
                        type=float)
    parser.add_argument("--num_factors",
                        default=64,
                        help="rank of the matrix decomposition",
                        type=int)
    parser.add_argument("--num_iters",
                        default=20,
                        help="number of iterations",
                        type=int)
    parser.add_argument("--max_samples",
                        default=5000,
                        help="max number of training samples in each iteration",
                        type=int)
    parser.add_argument("--k",
                        default=15,
                        help="k for hit@k and ndcg@k",
                        type=int)
    parser.add_argument("--output_dir",
                        default=str(p('Data', 'BPR', consts.DATASET_DOMAIN)),
                        help="save the trained model here",
                        type=str)
    parser.add_argument("--pretrained_dir",
                        default=".",
                        help="load the pretrained model here",
                        type=str)
    parser.add_argument('--subnetwork',
                        default='full',
                        choices=['dense', 'rs', 'full'],
                        help='the type of subnetwork to form from the full KG')
    parser.add_argument("--without_sampling",
                        default=True,
                        action="store_true")
    parser.add_argument("--load_pretrained_model",
                        action="store_true")
    parser.add_argument("--do_train",
                        default=True,
                        action="store_true")
    parser.add_argument("--do_eval",
                        default=True,
                        action="store_true")
    parser.add_argument("--eval_data",
                        default='kprn_test',
                        choices=['kprn_test', 'kprn_test_subset_1000','10users'],
                        help='Evaluation data')
    args = parser.parse_args()
    return args


def load_data(args):
    if consts.DATASET_DOMAIN == 'orgs':
        item = 'org'
    # load investor org dict and org investor dict
    with open(p('Data', consts.DATASET_DOMAIN, 'processed_data',
                consts.ITEM_IX_DATA_DIR,
                f'{args.subnetwork}_train_ix_inv_{item}.dict'), 'rb') as handle:
        train_inv_org = pickle.load(handle)
    with open(p('Data', consts.DATASET_DOMAIN, 'processed_data', consts.ITEM_IX_DATA_DIR,
                f'{args.subnetwork}_test_ix_inv_{item}.dict'), 'rb') as handle:
        test_inv_org = pickle.load(handle)
    with open(p('Data', consts.DATASET_DOMAIN,
                            'processed_data',
                            f'{consts.ITEM_IX_DATA_DIR}',
                           f'{args.subnetwork}_ix_{item}_inv.dict'), 'rb') as handle:
        full_org_inv = pickle.load(handle)
    with open(p('Data', consts.DATASET_DOMAIN,
                            'processed_data',
                            f'{consts.ITEM_IX_DATA_DIR}',
                           f'{args.subnetwork}_ix_{item}_biz.dict'), 'rb') as handle:
        full_org_biz = pickle.load(handle)
    for inv in list(train_inv_org):
        if train_inv_org[inv] == None or len(train_inv_org[inv]) == 0:
            train_inv_org.pop(inv)
            if inv in test_inv_org:
                test_inv_org.pop(inv)

    #get the index correspondence of the KPAN data and the matrix indices
    inv_ix = list(train_inv_org.keys())
    inv_ix.sort() # ascending order
    orgs_from_inv = set(full_org_inv.keys())
    orgs_from_biz = set(full_org_biz.keys())
    org_ix = list(orgs_from_inv.union(orgs_from_biz))
    org_ix.sort() # ascending order

    return train_inv_org, test_inv_org, full_org_inv, full_org_biz, inv_ix, org_ix

def prep_train_data(train_inv_org, inv_ix, org_ix):
    '''
    prepare training data in csr sparse matrix form
    '''
    num_invs = len(inv_ix)
    num_orgs = len(org_ix)

    # convert dictionary to sparse matrix
    mat = sp.dok_matrix((len(inv_ix), len(org_ix)), dtype=np.int8)

    # Iterate over train_inv_org with a progress bar
    for kprn_inv_id, kprn_org_ids in tqdm(train_inv_org.items(), desc="Processing", total=len(train_inv_org)):
        mf_inv_id = inv_ix.index(kprn_inv_id)
        for kprn_org_id in kprn_org_ids:
            mf_org_id = org_ix.index(kprn_org_id)
            mat[mf_inv_id, mf_org_id] = 1

    mat = mat.tocsr()
    print('shape of sparse matrix', mat.shape)
    print('number of nonzero entries: ', mat.nnz)

    # Save mat as a .pkl file to avoid preparing training data if already exist
    out_dir = p('Data', 'BPR', consts.DATASET_DOMAIN)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'bpr_mat.pkl', 'wb') as handle:
        pickle.dump(mat, handle)

    return mat



def prep_test_data(test_inv_org, train_inv_org, full_org_inv, full_org_biz, inv_ix, org_ix):
    '''
    for each user, for every 1 positive interaction in test data,
    randomly sample 100 negative interactions in tests data

    only evaluate on 10 users here

    convert indices to mf indices
    '''
    test_neg_inter = []
    test_pos_inter = []
    # TODO: this should be on all users
    eval_inv_ix_mf = list(range(1000))
    # test_data is a list of lists,
    # where each list is a list of 101 pairs ((u,i),tag)
    test_data = []

    print('find all pos interactions...')
    for inv_ix_mf in eval_inv_ix_mf:
        inv_ix_kprn = inv_ix[inv_ix_mf]
        for org_ix_kprn in test_inv_org[inv_ix_kprn]:
            org_ix_mf = org_ix.index(org_ix_kprn)
            test_pos_inter.append((inv_ix_mf, org_ix_mf))

    print('sample neg interactions...')
    for each_pos in test_pos_inter:
        instance = []
        instance.append((each_pos, 1))
        # append negative pairs
        inv_ix_mf = each_pos[0]
        inv_ix_kprn = inv_ix[inv_ix_mf]
        # use inv_ix_kprn to find all negative test orgs for that investor
        all_orgs = set(full_org_inv.keys()).union(set(full_org_biz.keys()))
        train_pos = set(train_inv_org[inv_ix_kprn])
        test_pos = set(test_inv_org[inv_ix_kprn])
        all_negative_orgs = all_orgs - train_pos - test_pos

        neg_samples = random.sample(all_negative_orgs, 100)
        for org_ix_kprn in neg_samples:
            org_ix_mf = org_ix.index(org_ix_kprn)
            instance.append(((inv_ix_mf, org_ix_mf), 0))
        test_data.append(instance)

    return test_data


def evaluate(args, model, inv_ix, org_ix, test_data):
    hit = 0
    ndcg = 0
    total = 0
    inv_set_mf = set()
    inv_set_kprn = set()
    rank_tuples = []
    for instance in test_data:
        # rank_tuples = []
        for i in instance:
            tag = i[1]
            print("i",i)
            #convert kprn indices to mf indices (inv and org)
            try:
                inv_ix_kprn = i[0][0]
                print("inv_ix_kprn")
                org_ix_kprn = i[0][1]
                inv_ix_mf = inv_ix.index(inv_ix_kprn)
                inv_set_mf.add(inv_ix_mf)
                inv_set_kprn.add(inv_ix_kprn)
                org_ix_mf = org_ix.index(org_ix_kprn)
                score = model.predict(inv_ix_mf, org_ix_mf)
                rank_tuples.extend([[inv_ix_kprn, org_ix_kprn, score, tag]])
            except:
                continue
    print('Total number of test cases: ', total)
    return rank_tuples


def load_test_data(args, eval_data):
    # test_data = None
    with open(p('Data', 'BPR', consts.DATASET_DOMAIN, f"bpr_matrix_test_{args.subnetwork}.pkl"), 'rb') as handle:
        test_data = pickle.load(handle)
    return test_data


def create_directory(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)


def main():
    random.seed(0)
    args = parse_args()
    create_directory(args.output_dir)
    # load data
    print('load data...')
    train_inv_org, test_inv_org, full_inv_org, full_inv_biz, \
    inv_ix, org_ix = load_data(args)

    model = None
    if args.load_pretrained_model:
        with open(args.pretrained_dir + "/mf_model.pkl", 'rb') as handle:
            model = pickle.load(handle)
    else:
        # initialize a new model
        bpra_args = BPRArgs()
        print("initializing a new model")
        bpra_args.learning_rate = args.learning_rate
        model = BPR(args.num_factors, bpra_args)

    if args.do_train:
        mat_path = p('Data', 'BPR', consts.DATASET_DOMAIN, 'bpr_mat.pkl')
        if mat_path.exists():
            # Load the file
            print('Training data already exist, no need to prepare from scratch')
            with open(mat_path, 'rb') as handle:
                train_data_mat = pickle.load(handle)
        else:
            print('prepare training data...')
            train_data_mat = prep_train_data(train_inv_org, inv_ix, org_ix)
        sample_negative_items_empirically = True
        sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
        max_samples = None if args.without_sampling else args.max_samples
        print('training...')
        print('max_samples: ', max_samples)
        train_new_model = not args.load_pretrained_model
        # data_dense = train_data_mat.toarray()

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(train_data_mat, test_size=0.2, random_state=42)
        train_matrix = sparse.csr_matrix(train_data)
        val_matrix = sparse.csr_matrix(val_data)
        # Store the validation data as a tuple
        model.train(train_matrix, sampler, args.num_iters, train_new_model, max_samples, val_matrix, early_stopping_rounds =2)
        num_5epochs = int(args.num_iters/5)
        eval_data = 'kprn_test_subset_1000'
        test_data = load_test_data(args, eval_data)
        print('test_data: ', len(test_data))
        # note that output_dir should contain information about
        # number of iterations, max sample size, num_factors, and learning rate
        bpr_out_dir = p('Data', 'BPR', consts.DATASET_DOMAIN)
        bpr_out_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(model, open(p('Data', 'BPR', consts.DATASET_DOMAIN, "mf_model_final.pkl"),"wb"), protocol=2)
        item_mapping, user_mapping = model.get_mapping_dataframe()
        pickle.dump(model.item_factors, open(p('Data', 'BPR', consts.DATASET_DOMAIN,
                                             "mf_full_6040_comp_16_items_embeddings.pkl"),"wb"), protocol=2)
        pickle.dump(model.user_factors,
                    open(p('Data', 'BPR', consts.DATASET_DOMAIN,
                                      "mf_full_6040_comp_16_users_embeddings.pkl"), "wb"), protocol=2)
        pickle.dump(user_mapping,
                    open(p('Data', 'BPR', consts.DATASET_DOMAIN,
                                      "mf_full_6040_comp_16_users_mapping.pkl"), "wb"), protocol=2)
        pickle.dump(item_mapping,
                    open(p('Data', 'BPR', consts.DATASET_DOMAIN,
                                      "mf_full_6040_comp_16_items_mapping.pkl"), "wb"), protocol=2)
        # initialize a new model
        bpra_args = BPRArgs()
        print("initializing a new model")
        bpra_args.learning_rate = args.learning_rate
        model = BPR(8, bpra_args)
        model.train(train_matrix, sampler, args.num_iters, train_new_model, max_samples, val_matrix, early_stopping_rounds =2)
        num_5epochs = int(args.num_iters / 5)
        item_mapping, user_mapping = model.get_mapping_dataframe()
        with open(p('Data', 'BPR', consts.DATASET_DOMAIN, f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_items_mapping.pkl'),
                  'wb') as handle:
            pickle.dump(item_mapping, handle)
        with open(p('Data', 'BPR', consts.DATASET_DOMAIN, f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_users_mapping.pkl'),
                  'wb') as handle:
            pickle.dump(user_mapping, handle)
        with open(p('Data', 'BPR', consts.DATASET_DOMAIN, f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_items_embeddings.pkl'),
                  'wb') as handle:
            pickle.dump(model.item_factors, handle)
        with open(p('Data', 'BPR', consts.DATASET_DOMAIN, f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_users_embeddings.pkl'),
                  'wb') as handle:
            pickle.dump(model.user_factors, handle)





        # EVALUATION
        if args.do_eval:
            print('start evaluation')
            all_results = evaluate(args, model, inv_ix, org_ix, test_data)
            df_model_results = prep_for_evaluation(all_results)
            df_scores_per_inv, mpr_score_per_inv = calc_scores_per_user(df=df_model_results, max_k=args.k,
                                                                          model_nm='model', mpr_metric=True)
            model_scores_rank_agg = aggregate_results(df=df_scores_per_inv, group_by=['model', 'rank'])
            print(f'MPR: {mpr_score_per_inv.mpr.mean()}')

            print('Save the results')
            # combine both results
            model_scores_rank_agg.rename({'rank': 'k'}, axis=1, inplace=True)
            out_results_dir = p('Results', 'Baseline', 'BPR')
            out_results_dir.mkdir(parents=True, exist_ok=True)
            model_scores_rank_agg.to_csv(
                out_results_dir / f'{consts.DATASET_DOMAIN}_{args.subnetwork}_all_model_scores.csv',
                index=False)


# save model results

if __name__=='__main__':
    main()
