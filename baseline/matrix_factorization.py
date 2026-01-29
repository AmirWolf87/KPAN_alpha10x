import pickle
import argparse

import torch
import optuna
from tqdm import tqdm
from bpr import *
import pandas as pd
import numpy as np
import mmap
import sklearn
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from os import mkdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from fastai.collab import *
# from fastai.tabular.all import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from pathlib import Path
import constants.consts as consts
from eval import *
from torch.optim import AdamW


# If this file is in <repo>/baseline/, parents[1] is <repo>
REPO_ROOT = Path(__file__).resolve().parents[1]

def p(*parts) -> Path:
    return REPO_ROOT.joinpath(*parts)

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
                        default=25000,
                        help='max number of users to find paths for')
    parser.add_argument('--num_of_components',
                        type=int,
                        default=2048,
                        help='Number of components for NMF_model')
    parser.add_argument('--samples',
                        type=int,
                        default=10,
                        help='Number of paths samples')
    # -- add new arguments here --
    parser.add_argument('--min_user_interactions',
                        type=int,
                        default=5,
                        help='Remove users with fewer than this many interactions')
    parser.add_argument('--min_item_interactions',
                        type=int,
                        default=5,
                        help='Remove items with fewer than this many interactions')
    parser.add_argument('--alpha_W',
                        type=float,
                        default=0.00018400837977918823, #<-- Optimal from Optuna was 0.001
                        help='Regularization strength for user factors (W)')
    parser.add_argument('--alpha_H',
                        type=float,
                        default=0.00010359148715265702,#<-- Optimal from Optuna was 0.001
                        help='Regularization strength for item factors (H)')
    parser.add_argument('--l1_ratio',
                        type=float,
                        default=0.12117766840473299,#<-- Optimal from Optuna was 0.0
                        help='0 for L2, 1 for L1 mixture of L1 and L2')
    parser.add_argument('--max_iter',
                        type=int,
                        default=160, #<-- Optimal from Optuna was 200
                        help='Max iterations for NMF')
    parser.add_argument('--init_method',
                        type=str,
                        default='nndsvda',
                        help='Initialization method for NMF')
    parser.add_argument('--partial_fit_cycles',
                        type=int,
                        default=1,
                        help='Number of times to re-fit or monitor the model')
    # NEW: add flag to optionally run hyperparameter tuning with Optuna
    parser.add_argument('--do_tuning',
                        action='store_true',
                        help='Run Optuna tuning')
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
        # create DataFrame of user-item interactions and add the "label" column
        data = pd.DataFrame(user_item_pairs, columns=['user', 'item'])
        data['label'] = 1  # using "label" instead of "interact"
        return data

    def create_embeddings(self, num_of_components, alpha_W=0.001, alpha_H=0.001, l1_ratio=0.0,
                          max_iter=200, init='nndsvda', partial_fit_cycles=1):
        # Build mappings using DataFrame so that merge works correctly:
        self.users_mapping = self.data[['user']].drop_duplicates().reset_index(drop=True)
        self.users_mapping = self.users_mapping.reset_index().rename(columns={'index': 'user_idx'})
        self.items_mapping = self.data[['item']].drop_duplicates().reset_index(drop=True)
        self.items_mapping = self.items_mapping.reset_index().rename(columns={'index': 'item_idx'})

        # Debug print mappings:
        print("Users mapping head:\n", self.users_mapping.head())
        print("Items mapping head:\n", self.items_mapping.head())

        # Create the sparse user-item matrix
        user_items_matrix = self._matrix_create()
        print("Shape:", user_items_matrix.shape)
        density = user_items_matrix.nnz / (user_items_matrix.shape[0] * user_items_matrix.shape[1])
        print("Density:", density)

        # Partial-fit loop: Run factorization multiple cycles for monitoring
        for cycle in range(1, partial_fit_cycles + 1):
            print(f"\n=== Factorization Cycle {cycle}/{partial_fit_cycles} ===")
            self.users_embeddings, self.items_embeddings = self._factorize_mat(
                matrix=user_items_matrix,
                num_of_components=num_of_components,
                alpha_W=alpha_W,
                alpha_H=alpha_H,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                init_method=init  # changed argument name to 'init_method' for clarity
            )
            # Inspect factor matrix statistics after each cycle
            print(f"W stats: min={self.users_embeddings.min():.4f}, max={self.users_embeddings.max():.4f}")
            print(f"H stats: min={self.items_embeddings.min():.4f}, max={self.items_embeddings.max():.4f}")

    def _matrix_create(self):
        # Merge the original data with the user and item mappings.
        merged_data = pd.merge(self.data, self.users_mapping, on='user', how='left')
        merged_data = pd.merge(merged_data, self.items_mapping, on='item', how='left')

        # Debug: Print the columns after merging to verify 'user_idx' and 'item_idx' exist
        print("Columns after merge:", merged_data.columns.tolist())

        # Use the "label" column for the interaction values.
        user_item_matrix = sparse.coo_matrix(
            (merged_data["label"], (merged_data['user_idx'], merged_data['item_idx'])),
            shape=(len(self.users_mapping), len(self.items_mapping))
        )
        return user_item_matrix

    def predict(self, items_users_for_pred, similarity='dot'):
        # Add users and items mapping, in order to use the trained embeddings:
        items_users_for_pred = pd.merge(items_users_for_pred, self.users_mapping, on='user', how='left')
        items_users_for_pred = pd.merge(items_users_for_pred, self.items_mapping, on='item', how='left')
        # Take for prediction by the model, only items and users which were part of the training phase:
        item_ids_for_pred = items_users_for_pred['item_idx'].drop_duplicates()[items_users_for_pred['item_idx'].drop_duplicates().notna()].astype(int)
        user_ids_for_pred = items_users_for_pred['user_idx'].drop_duplicates()[items_users_for_pred['user_idx'].drop_duplicates().notna()].astype(int)
        if similarity == 'cosine':
            scores_matrix = cosine_similarity(self.users_embeddings[user_ids_for_pred, :],
                                              self.items_embeddings[:, item_ids_for_pred].T)
        else:  # similarity == 'dot'
            # Convert numpy arrays to torch tensors
            users_tensor = torch.tensor(self.users_embeddings, dtype=torch.float)
            items_tensor = torch.tensor(self.items_embeddings, dtype=torch.float)
            user_ids = torch.tensor(user_ids_for_pred.values, dtype=torch.long)
            item_ids = torch.tensor(item_ids_for_pred.values, dtype=torch.long)
            users_norm = torch.nn.functional.normalize(users_tensor[user_ids, :], p=2, dim=1)
            items_slice = items_tensor[:, item_ids].T
            items_norm = torch.nn.functional.normalize(items_slice, p=2, dim=1)
            scores_matrix = torch.mm(users_norm, items_norm.T)

        scores_df = pd.DataFrame(scores_matrix,
                                 index=pd.merge(user_ids_for_pred, self.users_mapping, on='user_idx', how='left')['user'],
                                 columns=pd.merge(item_ids_for_pred, self.items_mapping, on='item_idx', how='left')['item']
                                ).stack().reset_index().rename(columns={0: 'y_pred'})
        users_items_df = pd.merge(items_users_for_pred, scores_df, on=['user', 'item'], how='left')
        users_items_df['y_pred'] = users_items_df['y_pred'].fillna(0)
        return users_items_df[['user', 'item', 'y_pred']]

    def train_preparation(self, train_path, data_dir):
        file_path = Path(data_dir) / consts.PATH_DATA_DIR / train_path
        users = []
        items = []
        labels = []
        with open(file_path, 'r') as file:
            for line in tqdm(file, total=self._get_num_lines(file_path)):
                test_interactions = eval(line.rstrip("\n"))
                actual_label = test_interactions[1]
                random_path = test_interactions[0][0]
                len_path = random_path[1]
                init_user = random_path[0][0][0]
                end_item = random_path[0][len_path - 1][0]
                users.append(init_user)
                items.append(end_item)
                labels.append(actual_label)
        processed_test = pd.DataFrame(list(zip(users, items, labels)),
                                      columns=['user', 'item', 'label'])
        return processed_test

    def test_preparation(self, test_path, data_dir):
        file_path = Path(data_dir) / consts.PATH_DATA_DIR / test_path
        users = []
        items = []
        labels = []
        with open(file_path, 'r') as file:
            for line in tqdm(file, total=self._get_num_lines(file_path)):
                test_interactions = eval(line.rstrip("\n"))
                for path_obj in test_interactions:
                    actual_label = path_obj[1]
                    random_path = path_obj[0][0]
                    len_path = random_path[1]
                    init_user = random_path[0][0][0]
                    end_item = random_path[0][len_path - 1][0]
                    users.append(init_user)
                    items.append(end_item)
                    labels.append(actual_label)
        processed_test = pd.DataFrame(list(zip(users, items, labels)),
                                      columns=['user', 'item', 'label'])
        return processed_test

    @staticmethod
    def _factorize_mat(matrix, num_of_components, alpha_W, alpha_H, l1_ratio, max_iter, init_method):
        print(f"Running NMF with n_components={num_of_components}, "
              f"alpha_W={alpha_W}, alpha_H={alpha_H}, l1_ratio={l1_ratio}, "
              f"init='{init_method}', max_iter={max_iter}")
        nmf = NMF(
            n_components=num_of_components,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            init=init_method,
            verbose=True,
            random_state=42
        )
        W = nmf.fit_transform(matrix)
        H = nmf.components_
        print(f"Final reconstruction error: {nmf.reconstruction_err_}")
        return W, H

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

    @staticmethod
    def objective(trial, user_item_matrix):
        n_components = trial.suggest_categorical("n_components", [64])#16, 32, 64, 128, 256, 512, 1024, 2048])
        alpha_W = trial.suggest_float("alpha_W", 1e-4, 1e-2, log=True)
        alpha_H = trial.suggest_float("alpha_H", 1e-4, 1e-2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 0.5)
        max_iter = trial.suggest_int("max_iter", 100, 300)
        init_method = trial.suggest_categorical("init_method", ["nndsvda", "nndsvd"])
        nmf = NMF(
            n_components=n_components,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            init=init_method,
            verbose=False,
            random_state=42
        )
        W = nmf.fit_transform(user_item_matrix)
        error = nmf.reconstruction_err_
        return error

    @staticmethod
    def tune_hyperparameters(user_item_matrix, n_trials=5): # was 50
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: MatrixFactorization.objective(trial, user_item_matrix), n_trials=n_trials)
        return study.best_params, study.best_value

# def create_directory(dir):
#     print("Creating directory %s" % dir)
#     try:
#         mkdir(dir)
#     except FileExistsError:
#         print("Directory already exists")


def main():
    print(sklearn.__version__)
    PROCESSED_DATA_DIR = p('Data', consts.DATASET_DOMAIN, 'processed_data')
    output_path = PROCESSED_DATA_DIR / 'mf' / 'mf_initial'
    output_path.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    data_ix_path = PROCESSED_DATA_DIR / (consts.ITEM_IX_DATA_DIR + args.subnetwork)
    data_ix_path = str(data_ix_path)

    mf = MatrixFactorization(data_ix_path)
    train_path = f"train_full_{args.user_limit}_samples{args.samples}_interactions.txt"
    processed_train = mf.train_preparation(train_path=train_path, data_dir=PROCESSED_DATA_DIR)

    # Filter out sparse users/items:
    user_counts = processed_train.groupby('user').size()
    valid_users = user_counts[user_counts >= args.min_user_interactions].index
    item_counts = processed_train.groupby('item').size()
    valid_items = item_counts[item_counts >= args.min_item_interactions].index
    processed_train = processed_train[
        (processed_train['user'].isin(valid_users)) &
        (processed_train['item'].isin(valid_items))
    ]
    print("Filtered training data shape:", processed_train.shape)
    mf.data = processed_train  # Update the internal data

    # Build mappings for users and items
    mf.users_mapping = mf.data[['user']].drop_duplicates().reset_index(drop=True)
    mf.users_mapping = mf.users_mapping.reset_index().rename(columns={'index': 'user_idx'})
    mf.items_mapping = mf.data[['item']].drop_duplicates().reset_index(drop=True)
    mf.items_mapping = mf.items_mapping.reset_index().rename(columns={'index': 'item_idx'})

    # Save the filtered training data for later use
    train_out = PROCESSED_DATA_DIR / 'mf'
    train_out.mkdir(parents=True, exist_ok=True)
    with open(train_out / f'processed_train_w_negatives_{args.subnetwork}_{str(args.user_limit)}.pkl', 'wb') as handle:
        pickle.dump(processed_train, handle, protocol=pickle.HIGHEST_PROTOCOL)


    user_items_matrix = mf._matrix_create()
    print("Matrix shape for tuning:", user_items_matrix.shape)
    # --- Hyperparameter Tuning Stage OPTIONAL---
    if args.do_tuning:
        best_params, best_error = MatrixFactorization.tune_hyperparameters(user_items_matrix, n_trials=50)
        print("Best hyperparameters:", best_params)
        print("Best reconstruction error:", best_error)
    # --- End of Hyperparameter Tuning Stage ---

    else:
    # Use default parameters if tuning is not enabled
        best_params = {
            "n_components": args.num_of_components,
            "alpha_W": args.alpha_W,
            "alpha_H": args.alpha_H,
            "l1_ratio": args.l1_ratio,
            "max_iter": args.max_iter,
            "init_method": args.init_method
        }
        print("Using default hyperparameters ( No Further HPT ) :", best_params)
    print('create embeddings')
    mf.create_embeddings(num_of_components=best_params["n_components"],
                         alpha_W=best_params["alpha_W"],
                         alpha_H=best_params["alpha_H"],
                         l1_ratio=best_params["l1_ratio"],
                         max_iter=best_params["max_iter"],
                         init='nndsvda',
                         partial_fit_cycles=args.partial_fit_cycles)

    mf_init_dir = PROCESSED_DATA_DIR / 'mf' / 'mf_initial'
    mf_init_dir.mkdir(parents=True, exist_ok=True)
    with open(mf_init_dir / f'mf_full_6040_comp_{best_params["n_components"]}_items_mapping.pkl', 'wb') as handle:
        pickle.dump(mf.items_mapping, handle)

    with open(mf_init_dir / f'mf_full_6040_comp_{best_params["n_components"]}_users_mapping.pkl', 'wb') as handle:
        pickle.dump(mf.users_mapping, handle)

    with open(mf_init_dir / f'mf_full_6040_comp_{best_params["n_components"]}_items_embeddings.pkl', 'wb') as handle:
        pickle.dump(mf.items_embeddings, handle)

    with open(mf_init_dir / f'mf_full_6040_comp_{best_params["n_components"]}_users_embeddings.pkl', 'wb') as handle:
        pickle.dump(mf.users_embeddings, handle)

    # Prepare test set for predict
    print('prepare testset for predict')
    if args.samples == -1:
        args.kg_path_file = args.subnetwork + '_' + str(args.user_limit) + '_' + args.kg_path_file
    else:
        args.kg_path_file = args.subnetwork + '_' + str(args.user_limit) + f'_samples{args.samples}_' + args.kg_path_file
    test_path = 'test_' + args.kg_path_file

    file_exist = False
    test_file = PROCESSED_DATA_DIR / 'mf' / f'processed_test_{args.subnetwork}_{str(args.user_limit)}.pkl'
    if test_file.is_file():
        with open(test_file, 'rb') as handle:
            processed_test = pickle.load(handle)
            file_exist = True
    else:
        print('create test set')
        processed_test = mf.test_preparation(test_path=test_path, data_dir=PROCESSED_DATA_DIR)

    print('predict')
    user_item_for_eval = processed_test.copy()
    predictions = mf.predict(items_users_for_pred=user_item_for_eval[['user', 'item']].drop_duplicates())
    predictions_save = user_item_for_eval.merge(predictions, on=['user', 'item'])
    pred_csv = output_path / f'mf_predictions_{args.subnetwork}_{str(args.user_limit)}_{str(best_params["n_components"])}.csv'
    pred_pkl = output_path / f'mf_predictions_{args.subnetwork}_{str(args.user_limit)}_{str(best_params["n_components"])}.pkl'

    predictions_save.to_csv(pred_csv, index=False)
    with open(pred_pkl, 'wb') as handle:
        pickle.dump(predictions_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Evaluation')
    max_k = 20
    df_for_evaluation = user_item_for_eval.merge(predictions, on=['user', 'item'])
    df_for_evaluation = df_for_evaluation[['user', 'item', 'y_pred', 'label']]
    df_mf_results = prep_for_evaluation(df_for_evaluation)
    df_mf_scores_per_user, mf_mpr_per_user = calc_scores_per_user(df=df_mf_results, max_k=max_k, model_nm='mf', mpr_metric=True)
    mf_scores_rank_agg = aggregate_results(df=df_mf_scores_per_user, group_by=['model', 'rank'])

    print('save the results')
    if not file_exist:
        print('processed_test not existing - now creating it')
        with open(test_file, 'wb') as handle:
            pickle.dump(processed_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mf_scores_rank_agg.to_csv(
        output_path / f'mf_scores_{args.subnetwork}_{str(args.user_limit)}_samples{str(args.samples)}_{str(best_params["n_components"])}_components.csv',
        index=False
    )

    mf_mpr_per_user.to_csv(
        output_path / f'mf_mpr_{args.subnetwork}_{str(args.user_limit)}_samples{str(args.samples)}_{str(best_params["n_components"])}_components.csv',
        index=False
    )
    print(f'MPR: {mf_mpr_per_user.mpr.mean()}')

if __name__ == "__main__":
    main()
