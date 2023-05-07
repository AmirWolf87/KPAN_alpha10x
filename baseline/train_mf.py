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


# maps the indices in the kprn data to the matrix indices here
kprn2matrix_user = {}
kprn2matrix_song = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",
                        default=0.001,
                        # default=0.000001
                        help="learning rate",
                        type=float)
    parser.add_argument("--num_factors",
                        default=16,
                        help="rank of the matrix decomposition",
                        type=int)
    parser.add_argument("--num_iters",
                        default=10,
                        help="number of iterations",
                        type=int)
    parser.add_argument("--max_samples",
                        default=1000,
                        help="max number of training samples in each iteration",
                        type=int)
    parser.add_argument("--k",
                        default=15,
                        help="k for hit@k and ndcg@k",
                        type=int)
    parser.add_argument("--output_dir",
                        default=f"C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}",
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
    else:
        item = 'movie'
    # load investor org dict and org investor dict
    # if args.subnetwork == 'dense':
    with open(os.path.join('C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data',
                           f'{consts.DATASET_DOMAIN}',
                            'processed_data',
                            f'{consts.ITEM_IX_DATA_DIR}'
                            f'{args.subnetwork}_train_ix_inv_{item}.dict'), 'rb') as handle:
        train_inv_org = pickle.load(handle)
    with open(os.path.join('C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data',
                           f'{consts.DATASET_DOMAIN}',
                            'processed_data',
                            f'{consts.ITEM_IX_DATA_DIR}',
                           f'{args.subnetwork}_test_ix_inv_{item}.dict'), 'rb') as handle:
        test_inv_org = pickle.load(handle)
    with open(os.path.join('C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data',
                           f'{consts.DATASET_DOMAIN}',
                            'processed_data',
                            f'{consts.ITEM_IX_DATA_DIR}',
                           f'{args.subnetwork}_ix_{item}_inv.dict'), 'rb') as handle:
        full_org_inv = pickle.load(handle)
    with open(os.path.join('C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data',
                           f'{consts.DATASET_DOMAIN}',
                            'processed_data',
                            f'{consts.ITEM_IX_DATA_DIR}',
                           f'{args.subnetwork}_ix_{item}_biz.dict'), 'rb') as handle:
        full_org_biz = pickle.load(handle)
    # elif args.subnetwork == 'rs':
    #     print 'rs subnetwork'
    #     with open("../data/song_data_ix/rs_train_ix_user_song_py2.pkl", 'rb') as handle:
    #         train_user_song = cPickle.load(handle)
    #     with open("../data/song_data_ix/rs_test_ix_user_song_py2.pkl", 'rb') as handle:
    #         test_user_song = cPickle.load(handle)
    #     with open("../data/song_data_ix/rs_ix_song_user_py2.pkl", 'rb') as handle:
    #         full_song_user = cPickle.load(handle)
    #     with open("../data/song_data_ix/rs_ix_song_person_py2.pkl", 'rb') as handle:
    #         full_song_person = cPickle.load(handle)

    # get rid of investors who don't invest in any org in the subnetwork
    # get rid of them in both train and test inv org dictionaries
    # for inv in train_inv_org.keys():
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


# def prep_train_data(train_inv_org, inv_ix, org_ix):
#     '''
#     prepare training data in csr sparse matrix form
#     '''
#     num_invs = len(inv_ix)
#     num_orgs = len(org_ix)
#
#     # convert dictionary to sparse matrix
#     mat = sp.dok_matrix((len(inv_ix), len(org_ix)), dtype=np.int8)
#     for kprn_inv_id, kprn_org_ids in train_inv_org.items():
#         mf_inv_id = inv_ix.index(kprn_inv_id)
#         for kprn_org_id in kprn_org_ids:
#             mf_org_id = org_ix.index(kprn_org_id)
#             mat[mf_inv_id, mf_org_id] = 1
#     mat = mat.tocsr()
#     print('shape of sparse matrix', mat.shape)
#     print('number of nonzero entries: ', mat.nnz)
#
#     return mat


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
    with open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}', f'bpr_mat.pkl'),'wb') as handle:
        pickle.dump(mat, handle)

    return mat



def prep_test_data(test_inv_org, train_inv_org, full_org_inv, full_org_biz, inv_ix, org_ix):
    '''
    for each user, for every 1 positive interaction in test data,
    randomly sample 100 negative interactions in tests data

    only evaluate on 10 users here

    converts kprn indices to mf indices
    '''
    # both test_neg_inter and test_pos_inter are a list of (u, i) pairs
    # where u is actual user index and i is actual song index
    test_neg_inter = [] # don't exist in either train and test
    test_pos_inter = [] # exist in test
    # TODO: this should be on all users
    eval_inv_ix_mf = list(range(4))
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
            # if args.eval_data in ['kprn_test_subset_1000', 'kprn_test']:
            #convert kprn indices to mf indices (inv and org)
            inv_ix_kprn = i[0][0]
            print("inv_ix_kprn")
            org_ix_kprn = i[0][1]
            inv_ix_mf = inv_ix.index(inv_ix_kprn)
            inv_set_mf.add(inv_ix_mf)
            inv_set_kprn.add(inv_ix_kprn)
            org_ix_mf = org_ix.index(org_ix_kprn)
            # else:
            #     inv_ix_mf = i[0][0]
            #     org_ix_mf = i[0][1]
            score = model.predict(inv_ix_mf, org_ix_mf)
            # rank_tuples.append((score, tag))
            rank_tuples.extend([[inv_ix_kprn, org_ix_kprn, score, tag]])
        # sort rank tuples based on descending order of score
        # rank_tuples.sort(reverse=True)
        # hit = hit + hit_at_k(rank_tuples, args.k)
        # ndcg = ndcg + ndcg_at_k(rank_tuples, args.k)
        # total = total + 1

    print('Total number of test cases: ', total)
    # print('hit at %d: %f' % (args.k, hit/float(total)))
    # print('ndcg at %d: %f' % (args.k, ndcg/float(total)))
    return rank_tuples


def load_test_data(args, eval_data):
    # test_data = None
    with open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',
                           f"bpr_matrix_test_{args.subnetwork}.pkl"), 'rb') as handle:
        test_data = pickle.load(handle)

    # if args.subnetwork == 'dense' and eval_data in ['kprn_test_subset_1000', 'kprn_test']:
    #     with open("../data/org_test_data/bpr_matrix_test_dense_py2.pkl", 'rb') as handle:
    #         test_data = pickle.load(handle)
    # elif args.subnetwork == 'rs' and eval_data in ['kprn_test_subset_1000', 'kprn_test']:
    #     with open("../data/org_test_data/bpr_matrix_test_rs_py2.pkl", 'rb') as handle:
    #         test_data = pickle.load(handle)
    # if eval_data == 'kprn_test_subset_1000':
    #     return random.sample(test_data, 100) #TODO: change back to 1000 when needed
    return test_data


def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        os.mkdir(dir)
    except:
        print("Directory already exists")


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
    # f = open(args.output_dir + '/logfile.txt', 'w')
    # backup = sys.stdout
    # sys.stdout = Tee(sys.stdout, f)

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
        mat_path = os.path.join('C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR', consts.DATASET_DOMAIN, 'bpr_mat.pkl')
        # Check if the file exists
        if os.path.exists(mat_path):
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
        # val_samples = []
        # for u in range(val_matrix.shape[0]):
        #     for i in val_matrix[u].nonzero()[1]:
        #         val_samples.append((u, i))

        # # Store the validation data as a tuple
        # validation_data = (val_matrix, val_samples)
        model.train(train_matrix, sampler, args.num_iters, train_new_model, max_samples, val_matrix, early_stopping_rounds =2)
        num_5epochs = int(args.num_iters/5)
        eval_data = 'kprn_test_subset_1000'
        test_data = load_test_data(args, eval_data)
        print('test_data: ', len(test_data))
        # note that output_dir should contain information about
        # number of iterations, max sample size, num_factors, and learning rate
        # for i in range(num_5epochs):
        #     print('epoch: ',  (5*(i+1)))
        #     # eval and dump model every 5 epochs
        #     # evaluate(args, model, user_ix, song_ix, test_data)
        #     pickle.dump(model, open(os.path.join(f'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Models/BPR/{consts.DATASET_DOMAIN}',
        #                                          f"mf_model_epoch_{5*(i+1)}.pkl"),"wb"), protocol=2)
        #     train_new_model = False
        #     model.train(train_data_mat, sampler, 5, train_new_model, max_samples)
        # evaluate(args, model, user_ix, song_ix, test_data)
        pickle.dump(model, open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',
                                             "mf_model_final.pkl"),"wb"), protocol=2)
        item_mapping, user_mapping = model.get_mapping_dataframe()
        pickle.dump(model.item_factors, open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',
                                             "mf_full_6040_comp_16_items_embeddings.pkl"),"wb"), protocol=2)
        pickle.dump(model.user_factors,
                    open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',
                                      "mf_full_6040_comp_16_user_embeddings.pkl"), "wb"), protocol=2)
        pickle.dump(user_mapping,
                    open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',
                                      "mf_full_6040_comp_16_users_mapping.pkl"), "wb"), protocol=2)
        pickle.dump(item_mapping,
                    open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}',
                                      "mf_full_6040_comp_16_items_mapping.pkl"), "wb"), protocol=2)
        # initialize a new model
        bpra_args = BPRArgs()
        print("initializing a new model")
        bpra_args.learning_rate = args.learning_rate
        model = BPR(8, bpra_args)
        # if args.do_train:
        #     print('Now with num_components = 32')
        #     train_data_mat = prep_train_data(train_inv_org, inv_ix, org_ix)
        #     sample_negative_items_empirically = True
        #     sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
        #     max_samples = None if args.without_sampling else args.max_samples
        #     print('training...')
        #     print('max_samples: ', max_samples)
        # train_new_model = not args.load_pretrained_model
        model.train(train_data_mat, sampler, args.num_iters, train_new_model, max_samples)
        num_5epochs = int(args.num_iters / 5)
        item_mapping, user_mapping = model.get_mapping_dataframe()
        with open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}', f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_items_mapping.pkl'),
                  'wb') as handle:
            pickle.dump(item_mapping, handle)
        with open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}', f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_users_mapping.pkl'),
                  'wb') as handle:
            pickle.dump(user_mapping, handle)
        with open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}', f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_items_embeddings.pkl'),
                  'wb') as handle:
            pickle.dump(model.item_factors, handle)
        with open(os.path.join(f'C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Data/BPR/{consts.DATASET_DOMAIN}', f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_users_embeddings.pkl'),
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
            model_scores_rank_agg.to_csv(os.path.join('C:/M.Sc/Thesis/Alpha10x/KPAN_V2/Results/Baseline/BPR',
                                                      f'{consts.DATASET_DOMAIN}_{args.subnetwork}_all_model_scores.csv'), index=False)  # save model results



    # if (args.do_train or args.load_pretrained_model) and args.do_eval:
    #     print('prepare test data...')
    #     # note: the user and song indices have not been converted to the mf indices
    #     # the conversion will be done in the evaluate function
    #     if args.eval_data in ['kprn_test_subset_1000', 'kprn_test']:
    #         test_data = load_test_data(args, args.eval_data)
    #     elif args.eval_data == '10users':
    #         test_data = prep_test_data(test_user_song, train_user_song, \
    #                                    full_song_user, full_song_person, \
    #                                    user_ix, song_ix)
    #     print('evaluating...')
    #     print('test_data: ', len(test_data))
    #     evaluate(args, model, user_ix, song_ix, test_data)

if __name__=='__main__':
    main()
