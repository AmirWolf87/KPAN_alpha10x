from pathlib import Path
import os
import pickle
import pandas as pd

from eval import calc_scores_per_user, prep_for_evaluation, aggregate_results
import constants.consts as consts

# Repo root: adjust parents[1] if this script sits in a subfolder like baseline/
REPO_ROOT = Path(__file__).resolve().parents[1]

def p(*parts) -> Path:
    return REPO_ROOT.joinpath(*parts)


output_dir = p('Results', 'Baseline', 'path_length')
output_dir.mkdir(parents=True, exist_ok=True)

test_set_path = p('Data', consts.DATASET_DOMAIN, 'processed_data', 'mf', 'processed_test_full_100.pkl')
path_file = p('Data', consts.DATASET_DOMAIN, 'processed_data', 'path_data', 'test_path_length_100.pkl')

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
df_for_evaluation.to_csv(output_dir / f'{consts.DATASET_DOMAIN}_path_length_predictions.csv', index=False)
rkge_scores_rank_agg.to_csv(output_dir / f'{consts.DATASET_DOMAIN}_path_length_scores_k_20.csv', index=False)
mpr_per_user.to_csv(output_dir / f'{consts.DATASET_DOMAIN}_path_length_mpr_k_20.csv', index=False)
print(f'MPR: {mpr_per_user.mpr.mean()}')
