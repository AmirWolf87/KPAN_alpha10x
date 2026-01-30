import pickle
import pandas as pd
from sklearn.decomposition import NMF
from scipy import sparse
import os
import constants.consts as consts
from pathlib import Path


class MatrixFactorization:

    def __init__(self, data_ix_path):
        self.data = self.prep_from_dict(data_ix_path)
        self.users_embeddings = None
        self.items_embeddings = None
        self.users_mapping = None
        self.items_mapping = None

    def prep_from_dict(self, data_ix_path, train_or_test='train'):
        data_ix_path = Path(data_ix_path)
        with open(str(data_ix_path) + f'_{train_or_test}_ix_{consts.ITEM_USER_DICT}', 'rb') as handle:
            item_user_train = pickle.load(handle)

        user_item_pairs = [(i, v) for v, u in item_user_train.items() for i in u]
        # create df of user-item
        data = pd.DataFrame(user_item_pairs, columns=['user', 'item'])
        data['interact'] = 1
        return data

    def create_embeddings(self, num_of_components):
        # mapping the ids
        self.items_mapping = self.data['item'].drop_duplicates().reset_index(drop=True).reset_index().rename(
            columns={'index': 'item_idx'})
        self.users_mapping = self.data['user'].drop_duplicates().reset_index(drop=True).reset_index().rename(
            columns={'index': 'user_idx'})

        # the interactions are binary. so there is no meaning for sum/exp smoothing
        user_items_matrix = self.matrix_create()
        self.users_embeddings, self.items_embeddings = self.factorize_mat(user_items_matrix, num_of_components)

    def matrix_create(self):
        self.data = pd.merge(self.data, self.users_mapping, on='user')
        self.data = pd.merge(self.data, self.items_mapping, on='item')

        user_item_matrix = sparse.coo_matrix(
            (self.data['interact'], (self.data['user_idx'], self.data['item_idx'])),
            shape=(len(self.users_mapping), len(self.items_mapping)))

        return user_item_matrix

    def factorize_mat(self,matrix, num_of_components):
        nmf = NMF(num_of_components, verbose=False)
        W = nmf.fit_transform(matrix)
        H = nmf.components_

        return W, H


class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "MatrixFactorization"
        return super().find_class(module, name)

if __name__ == "__main__":
    # Repo-root relative paths (portable)
    REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust if this file is at repo root -> .parent
    def p(*parts) -> Path:
        return REPO_ROOT.joinpath(*parts)

    PROCESSED_DATA_DIR = p('Data', consts.DATASET_DOMAIN, 'processed_data')
    output_path = PROCESSED_DATA_DIR / 'mf' / 'mf_initial'
    output_path.mkdir(parents=True, exist_ok=True)

    # Index path prefix (do NOT rely on trailing slashes)
    data_ix_path = PROCESSED_DATA_DIR / consts.ITEM_IX_DATA_DIR / 'full'


    mf = MatrixFactorization(data_ix_path)
    print('create embeddings')

    # save embeddings and mappings
    ### DIM = 64
    mf.create_embeddings(num_of_components=consts.ENTITY_EMB_DIM)
    with open(output_path / f'mf_full_6040_comp_{consts.ENTITY_EMB_DIM}_items_mapping.pkl', 'wb') as handle:
        pickle.dump(mf.items_mapping, handle)
    print(f' done creating mf_full_6040_{consts.ENTITY_EMB_DIM}_items_mapping.pkl in {output_path}')
    with open(output_path / f'mf_full_6040_comp_{consts.ENTITY_EMB_DIM}_users_mapping.pkl', 'wb') as handle:
        pickle.dump(mf.users_mapping, handle)
    with open(output_path / f'mf_full_6040_comp_{consts.ENTITY_EMB_DIM}_items_embeddings.pkl', 'wb') as handle:
        pickle.dump(mf.items_embeddings, handle)
    with open(output_path / f'mf_full_6040_comp_{consts.ENTITY_EMB_DIM}_users_embeddings.pkl', 'wb') as handle:
        pickle.dump(mf.users_embeddings, handle)

    ### DIM = 32
    mf = MatrixFactorization(data_ix_path)
    print('create embeddings')
    mf.create_embeddings(num_of_components=consts.TYPE_EMB_DIM)
    with open(output_path / f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_items_mapping.pkl', 'wb') as handle:
        pickle.dump(mf.items_mapping, handle)
    with open(output_path / f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_users_mapping.pkl', 'wb') as handle:
        pickle.dump(mf.users_mapping, handle)
    with open(output_path / f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_items_embeddings.pkl', 'wb') as handle:
        pickle.dump(mf.items_embeddings, handle)
    with open(output_path / f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_users_embeddings.pkl', 'wb') as handle:
        pickle.dump(mf.users_embeddings, handle)
