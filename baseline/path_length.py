from KPRN.eval import calc_scores_per_user, prep_for_evaluation, aggregate_results
import pickle
from os import mkdir, path
import pandas as pd
import numpy as np
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import KPRN.constants.consts as consts



output_path = r'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Results/Baseline/path_length'
if consts.DATASET_DOMAIN == 'movielens':
    test_set_path = rf'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Data/{consts.DATASET_DOMAIN}/processed_data/mf/processed_test_full_6040.pkl'
    path_file = r'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Data/movielens/processed_data/path_data/test_path_length_6040.pkl'
else:
    test_set_path = rf'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Data/{consts.DATASET_DOMAIN}/processed_data/mf/processed_test_full_10000.pkl'
    path_file = path_file = r'C:/Users/netac/Downloads/Amir/MSc/Thesis/alpha10x/KPAN10x/KPAN_V2/Data/songs/processed_data/path_data/test_path_length_10000.pkl'

with open(test_set_path, 'rb') as handle:
    test_set = pickle.load(handle)

with open(path_file, 'rb') as handle:
    path_length = pickle.load(handle)

path_length_pred = pd.Series(path_length).reset_index()
path_length_pred.columns = ['user','item','y_pred']
df_for_evaluation = test_set.merge(path_length_pred,on=['user','item'],how='left')

# evaluation
print('Evaluation')
max_k = 20
df_for_evaluation = df_for_evaluation[['user', 'item', 'y_pred', 'label']]
df_rkge_results = prep_for_evaluation(df_for_evaluation)
df_rkge_scores_per_user,mpr_per_user = calc_scores_per_user(df=df_rkge_results, max_k=max_k, model_nm='path_length',mpr_metric=True)
rkge_scores_rank_agg = aggregate_results(df=df_rkge_scores_per_user, group_by=['model', 'rank'])

print('save the results')
df_for_evaluation.to_csv(os.path.join(output_path,f'{consts.DATASET_DOMAIN}_path_length_predictions.csv'),index=False)
rkge_scores_rank_agg.to_csv(os.path.join(output_path,f'{consts.DATASET_DOMAIN}_path_length_scores_k_20.csv'),index=False)
mpr_per_user.to_csv(os.path.join(output_path,f'{consts.DATASET_DOMAIN}_path_length_mpr_k_20.csv'),index=False)
print(f'MPR: {mpr_per_user.mpr.mean()}')
