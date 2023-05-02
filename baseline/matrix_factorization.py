import pickle
import argparse
from tqdm import tqdm
from bpr import *
import pandas as pd
import numpy as np
import mmap
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from os import mkdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fastai.collab import *
from fastai.tabular.all import *
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from pathlib import Path
import consts as consts
from eval import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline',
                        type=str,
                        default='mf',
                        help='name of the baseline')
    parser.add_argument('--kg_path_file',
                        type=str,
                        default='interactions.txt',
                        help='file name to store/load train/test paths')
    parser.add_argument('--subnetwork',
                        default='full',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to load data from')
    parser.add_argument('--user_limit',
                        type=int,
                        default=40000,
                        help='max number of users to find paths for')
    parser.add_argument('--num_of_components',
                        type=int,
                        default=16,
                        help='Number of components for NMF_model')
    parser.add_argument('--samples',
                        type=int,
                        default=100,
                        help='Number of paths samples')
    return parser.parse_args()


class MatrixFactorization:

    def __init__(self, data_ix_path):
        self.data = self.prep_from_dict(data_ix_path)
        self.users_embeddings = None
        self.items_embeddings = None
        self.users_mapping = None
        self.items_mapping = None

    def prep_from_dict(self, data_ix_path, train_or_test='train'):
        with open(data_ix_path + f'_{train_or_test}_ix_{consts.ITEM_USER_DICT}', 'rb') as handle:
            item_user_train = pickle.load(handle)


        # keys are organizations and values are investors of organizations
        user_item_pairs = [(i, v) for v, u in item_user_train.items() for i in u]
        # create df of investor-org
        data = pd.DataFrame(user_item_pairs, columns=['user', 'item'])
        data['interact'] = 1

        # with open(data_ix_path, 'rb') as handle:
        #     data = pickle.load(handle)
        # data = data.rename({'label':'interact'},axis=1)

        return data

    def create_embeddings(self, num_of_components):
        # mapping the ids
        self.items_mapping = self.data['item'].drop_duplicates().reset_index(drop=True).reset_index().rename(
            columns={'index': 'item_idx', 0: 'item'})
        self.users_mapping = self.data['user'].drop_duplicates().reset_index(drop=True).reset_index().rename(
            columns={'index': 'user_idx'})
        # the interactions are binary. so there is no meaning for sum/exp smoothing
        user_items_matrix = self._matrix_create()
        self.users_embeddings, self.items_embeddings = self._factorize_mat(user_items_matrix, num_of_components)


    def _matrix_create(self):
        self.data = pd.merge(self.data, self.users_mapping, on='user')
        self.data = pd.merge(self.data, self.items_mapping, on='item')

        user_item_matrix = sparse.coo_matrix(
            (self.data['interact'], (self.data['user_idx'], self.data['item_idx'])),
            shape=(len(self.users_mapping), len(self.items_mapping)))

        return user_item_matrix

    def predict(self, items_users_for_pred, similarity='dot'):
        # Add users and items mapping, in order to use the trained embeddings:
        items_users_for_pred = pd.merge(items_users_for_pred, self.users_mapping, on='user', how='left')
        items_users_for_pred = pd.merge(items_users_for_pred, self.items_mapping, on='item', how='left')
        # Take for prediction by the model, only items and users which were part of the training phase:
        item_ids_for_pred = items_users_for_pred['item_idx'].drop_duplicates()[
            items_users_for_pred['item_idx'].drop_duplicates().notna()].astype(int)
        user_ids_for_pred = items_users_for_pred['user_idx'].drop_duplicates()[
            items_users_for_pred['user_idx'].drop_duplicates().notna()].astype(int)
        if similarity == 'cosine':
            scores_matrix = cosine_similarity(self.users_embeddings[user_ids_for_pred, :],
                                              self.items_embeddings[:, item_ids_for_pred])
        else:  # similarity == 'dot':
            scores_matrix = self.users_embeddings[user_ids_for_pred, :].dot(self.items_embeddings[:,item_ids_for_pred])
        # convert the mapping to its original idx
        scores_df = pd.DataFrame(scores_matrix,
                                 index=pd.merge(user_ids_for_pred, self.users_mapping, on='user_idx', how='left')[
                                     'user'],
                                 columns=pd.merge(item_ids_for_pred, self.items_mapping, on='item_idx', how='left')[
                                     'item']).stack().reset_index().rename(
            columns={0: 'y_pred'})

        # Now add 'cold' users, which will get 0 as y_pred:
        users_items_df = pd.merge(items_users_for_pred, scores_df, on=['user', 'item'], how='left')
        users_items_df['y_pred'] = users_items_df['y_pred'].fillna(0)
        # users_items_df['y_pred'] = users_items_df['y_pred'].fillna(pd.Series(np.random.rand(users_items_df.shape[0])))
        return users_items_df[['user', 'item', 'y_pred']]

    def train_preparation(self, train_path, data_dir):

        file_path = os.path.join(data_dir, consts.PATH_DATA_DIR, train_path)

        users = []
        items = []
        labels = []
        with open(file_path, 'r') as file:
            for line in tqdm(file, total=self._get_num_lines(file_path)):
                test_interactions = eval(line.rstrip("\n"))

                # for path_obj,label in test_interactions:
                actual_label = test_interactions[1]
                random_path = test_interactions[0][0]
                # len of the path to know where is the end item
                len_path = random_path[1]
                init_user = random_path[0][0][0]
                end_item = random_path[0][len_path - 1][0]

                # add to lists
                users.append(init_user)
                items.append(end_item)
                labels.append(actual_label)

        processed_test = pd.DataFrame(list(zip(users, items, labels)),
                                      columns=['user', 'item', 'label'])
        return processed_test

    def test_preparation(self, test_path, data_dir):

        file_path = os.path.join(data_dir, consts.PATH_DATA_DIR, test_path)

        users = []
        items = []
        labels = []
        with open(file_path, 'r') as file:
            for line in tqdm(file, total=self._get_num_lines(file_path)):
                test_interactions = eval(line.rstrip("\n"))

                for path_obj in test_interactions:
                    actual_label = path_obj[1]
                    random_path = path_obj[0][0]
                    # len of the path to know where is the end item
                    len_path = random_path[1]
                    init_user = random_path[0][0][0]
                    end_item = random_path[0][len_path - 1][0]

                    # add to lists
                    users.append(init_user)
                    items.append(end_item)
                    labels.append(actual_label)

        processed_test = pd.DataFrame(list(zip(users, items, labels)),
                                      columns=['user', 'item', 'label'])
        return processed_test

    @staticmethod
    def _factorize_mat(matrix, num_of_components, early_stopping=True, test_size=0.2, patience=5):
        df_matrix = pd.DataFrame.sparse.from_spmatrix(matrix)
        # reset the index to create a column for user ids
        df_matrix = df_matrix.reset_index().rename(columns={'index': 'user'})
        # convert the dataframe to a long format with 3 columns
        df_long = df_matrix.set_index('user').stack().reset_index()
        df_long.columns = ['user', 'item', 'rating']
        # Create DataLoaders
        dls = CollabDataLoaders.from_df(df_long, bs=256)


        # Create model
        learn = collab_learner(dls, n_factors=num_of_components, y_range=(0, 1), wd=1e-1)

        # Train the model
        learn.fit_one_cycle(5, 5e-3, cbs=[EarlyStoppingCallback(patience=2)])
        # Extract the user and item embeddings
        users = learn.model.u_weight.weight.data.cpu().numpy()
        items = learn.model.i_weight.weight.data.cpu().numpy()
        return users, items.T


    @staticmethod
    def matrix_to_dataframe(matrix, items_mapping, users_mapping):
        return pd.DataFrame.sparse.from_spmatrix(matrix, columns=items_mapping['item'], index=users_mapping['user'])
    @staticmethod
    def _get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines


def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")


def main():
    # data_root = r'/data2/Leigh' # Game-01
    data_root = Path(os.getcwd()).parent
    files_root = Path(os.getcwd()).parent#.parents[2] # local - parents[2]
    print(files_root)
    PROCESSED_DATA_DIR = os.path.join(data_root, 'Data', consts.DATASET_DOMAIN,
                                      'processed_data')
    output_path = os.path.join(PROCESSED_DATA_DIR, 'mf', 'mf_initial')
    # output_path = os.path.join(files_root, 'Results', 'Baseline', 'MF', consts.DATASET_DOMAIN)
    create_directory(dir=output_path)

    args = parse_args()
    data_ix_path = os.path.join(PROCESSED_DATA_DIR, consts.ITEM_IX_DATA_DIR + args.subnetwork)
    # data_ix_path = os.path.join(PROCESSED_DATA_DIR, 'mf',f'processed_train_{args.subnetwork}_{str(args.user_limit)}.pkl')

    mf = MatrixFactorization(data_ix_path)
    processed_train = mf.train_preparation(train_path='train_full_40000_samples100_interactions.txt', #train_full_10000_interactions
                                          data_dir=PROCESSED_DATA_DIR)
    with open(os.path.join(PROCESSED_DATA_DIR, 'mf', f'processed_train_w_negatives_{args.subnetwork}_{str(args.user_limit)}.pkl'), #PROCESSED_DATA_DIR= "Data/Orgs/processed_data
              'wb') as handle:
        pickle.dump(processed_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

#    TODO: for running MF - uncomment
    print('create embeddings')
    mf.create_embeddings(num_of_components=args.num_of_components)
    # save embeddings and mappings for model exploration (below Data)
    with open(os.path.join(PROCESSED_DATA_DIR,'mf', 'mf_initial', f'mf_full_6040_comp_{args.num_of_components}_items_mapping.pkl'), 'wb') as handle:
        pickle.dump(mf.items_mapping, handle)
    with open(os.path.join(PROCESSED_DATA_DIR,'mf', 'mf_initial', f'mf_full_6040_comp_{args.num_of_components}_users_mapping.pkl'), 'wb') as handle:
        pickle.dump(mf.users_mapping, handle)
    with open(os.path.join(PROCESSED_DATA_DIR,'mf', 'mf_initial', f'mf_full_6040_comp_{args.num_of_components}_items_embeddings.pkl'), 'wb') as handle:
        pickle.dump(mf.items_embeddings, handle)
    with open(os.path.join(PROCESSED_DATA_DIR,'mf', 'mf_initial', f'mf_full_6040_comp_{args.num_of_components}_users_embeddings.pkl'), 'wb') as handle:
        pickle.dump(mf.users_embeddings, handle)


    # prepare for predict
    print('prepare testset for predict')
    if args.samples == -1:
        args.kg_path_file = args.subnetwork + '_' + str(
            args.user_limit) + '_' + args.kg_path_file
    else:
        args.kg_path_file = args.subnetwork + '_' + str(args.user_limit) + f'_samples{args.samples}_' + args.kg_path_file
    test_path = 'test_' + args.kg_path_file

    file_exist = False
    if os.path.isfile(os.path.join(PROCESSED_DATA_DIR,'mf',f'processed_test_{args.subnetwork}_{str(args.user_limit)}.pkl')):
        with open(os.path.join(PROCESSED_DATA_DIR,'mf',f'processed_test_{args.subnetwork}_{str(args.user_limit)}.pkl'), 'rb') as handle:
            processed_test = pickle.load(handle)
            file_exist = True
    else: # create the file
        print('create test set')
        processed_test = mf.test_preparation(test_path=test_path, data_dir=PROCESSED_DATA_DIR)

    # predict
    print('predict')
    user_item_for_eval = processed_test.copy()

    # predictions = mf.predict(items_users_for_pred=processed_test[['user', 'item']].drop_duplicates())
    predictions = mf.predict(items_users_for_pred=user_item_for_eval[['user', 'item']].drop_duplicates())
    predictions_save = user_item_for_eval.merge(predictions,on=['user', 'item'])
    predictions_save.to_csv(os.path.join(output_path,f"mf_predictions_{args.subnetwork}_{str(args.user_limit)}_{str(args.num_of_components)}.csv"))
    with open(os.path.join(output_path,
                           f'mf_predictions_{args.subnetwork}_{str(args.user_limit)}_{str(args.num_of_components)}.pkl'), 'wb') as handle:
        pickle.dump(predictions_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # evaluation
    print('Evaluation')
    max_k = 20
    df_for_evaluation = user_item_for_eval.merge(predictions, on=['user', 'item'])
    df_for_evaluation = df_for_evaluation[['user', 'item', 'y_pred', 'label']]
    df_mf_results = prep_for_evaluation(df_for_evaluation)
    df_mf_scores_per_user,mf_mpr_per_user = calc_scores_per_user(df=df_mf_results, max_k=max_k, model_nm='mf',mpr_metric=True)
    mf_scores_rank_agg = aggregate_results(df=df_mf_scores_per_user, group_by=['model', 'rank'])

    print('save the results')
    # save processed_test to Data
    if not file_exist:
        with open(os.path.join(PROCESSED_DATA_DIR,'mf',f'processed_test_{args.subnetwork}_{str(args.user_limit)}.pkl'), 'wb') as handle:
            pickle.dump(processed_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # processed_test.to_csv(os.path.join(output_path,f'prcessed_test_{args.subnetwork}_{str(args.user_limit)}.csv'))
    mf_scores_rank_agg.to_csv(os.path.join(output_path,
                                           f'mf_scores_{args.subnetwork}_{str(args.user_limit)}_samples{str(args.samples)}_{str(args.num_of_components)}_components.csv'),
                              index=False)
    mf_mpr_per_user.to_csv(os.path.join(output_path,
                                           f'mf_mpr_{args.subnetwork}_{str(args.user_limit)}_samples{str(args.samples)}_{str(args.num_of_components)}_components.csv'),
                              index=False)
    print(f'MPR: {mf_mpr_per_user.mpr.mean()}')


if __name__ == "__main__":
    main()
