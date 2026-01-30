import pickle
import time
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
import argparse
import random
import mmap
from tqdm import tqdm
from collections import defaultdict
from os import mkdir, path
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

import constants.consts as consts
from model import KPRN, train, predict, KPAN
from baseline.popularity import PopularityBL
from Data.format import format_paths
from Data.path_extraction import find_paths_user_to_items
from eval import precision_at_k, ndcg_at_k, calc_scores_per_user, hit_at_k, prep_for_evaluation, aggregate_results
from model.validation import ValidationData
from model.predictor import *
from shared_args import add_user_limit_and_samples

def parse_args():
    parser = argparse.ArgumentParser()
    add_user_limit_and_samples(parser)
    parser.add_argument('--train',
                        default=True,
                        action='store_true',
                        help='whether to train the model')
    parser.add_argument('--validation',
                        default=False,
                        help='use validation set')
    parser.add_argument('--eval',
                        default=True,
                        action='store_true',
                        help='whether to evaluate the model')
    parser.add_argument('--find_paths',
                        default=True,
                        action='store_true',
                        help='whether to find paths (otherwise load from disk)')
    parser.add_argument('--subnetwork',
                        default= 'full',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to load data from')
    parser.add_argument('--mode',
                        default='client')
    parser.add_argument('--model',
                        type=str,
                        default='model.pt',
                        help='name to save or load model from')
    parser.add_argument('--model_name',
                        type=str,
                        default='KPAN',
                        choices=['KPRN','KPAN'],
                        help='name to model type')
    parser.add_argument('--path_agg_method',
                        type=str,
                        default='attention',
                        choices=['attention','weighted_pooling'],
                        help='how to aggregate the paths')
    parser.add_argument('--load_checkpoint',
                        default=False,
                        action='store_true',
                        help='whether to load the current model state before training ')
    parser.add_argument('--kg_path_file',
                        type=str,
                        default='interactions.txt',
                        help='file name to store/load train/test paths')
    # parser.add_argument('--user_limit',
    #                     type=int,
    #                     default=25000,#40000 takes too much time
    #                     help='max number of users to find paths for')
    parser.add_argument('-e','--epochs',
                        type=int,
                        default=1,#5,25
                        help='number of epochs for training model')
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=64,#64
                        help='batch_size')
    parser.add_argument('--not_in_memory',
                        default=False,
                        action='store_true',
                        help='denotes that the path data does not fit in memory')
    parser.add_argument('--lr',
                        type=float,
                        default=0.002,#.002,
                        help='learning rate')
    parser.add_argument('--l2_reg',
                        type=float,
                        default=0.0001,#.0001,
                        help='l2 regularization coefficient')
    parser.add_argument('--gamma',
                        type=float,
                        default=1,#1,
                        help='gamma for weighted pooling')
    parser.add_argument('--nhead',
                        type=int,
                        default=8,
                        help='number of heads for MultiHead Attention')
    parser.add_argument('--path_nhead',
                        type=int,
                        default=1,
                        help='number of heads for MultiHead Attention in path aggregation')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.01,#0.01
                        help='dropout for Transformer')
    parser.add_argument('--entity_agg',
                        type=str,
                        default='max',
                        help='aggregation method for Transformer')
    parser.add_argument('--no_rel',
                        default=False,
                        action='store_true',
                        help='Run the model without relation if True')
    parser.add_argument('--np_baseline',
                        default=False,
                        action='store_true',
                        help='Run the model with the number of path baseline if True')
    # parser.add_argument('--samples',
    #                     type=int,
    #                     default=10,#10#100
    #                     help='number of paths to sample for each interaction (-1 means include all paths)')
    parser.add_argument('--random_samples',
                        default=False,
                        help='sampling paths while training')
    parser.add_argument('--long_paths',
                        default=False,
                        help='sampling paths in length 6')
    parser.add_argument('--init_mf_embeddings',
                        default=True,
                        help='initialized model embeddings with MF embeddings')
    parser.add_argument('--add_path_length',
                        default=False,
                        help='Add number of paths as scalar to entity embeddings')
    parser.add_argument('--item_to_item',
                        default=False,
                        help='Inference on task of item-to-item')
    # parser.add_argument('--data_masking',
    #                     default=False,
    #                     help='Masking entities data to the model in training. remeber to delete org_full_{user_limit}_model.pt and Data\orgs\processed_data\path_data --> train_full_{user_limit}_samples{samples}}_interactions.txt in any change ')
    return parser.parse_args()


def create_directory(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def sample_paths(paths, samples):
    index_list = list(range(len(paths)))
    random.shuffle(index_list)
    indices = index_list[:samples]
    return [paths[i] for i in indices]

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_data(item_person, person_item, user_item_all, item_user_all, item_user_split, user_item_split, neg_samples,
              e_to_ix, t_to_ix, r_to_ix, path_dir, kg_path_file, subnetwork, len_3_branch, len_5_branch, limit=10,
              version="train", samples=-1, save_dicts=False, long_paths=True, random_sample=True):
    '''
    Constructs paths for train/test data,

    For training, we write each formatted interaction to a file as we find them
    For testing, for each combo of a pos paths and 100 neg paths we store these in a single line in the file
    '''
    global item_to_paths_len5
    create_directory(path_dir)
    print("Directory created at: ", path_dir)
    path_file = open(os.path.join(path_dir, kg_path_file), 'w')
    print("Path file:", os.path.join(path_dir, kg_path_file))
    # save all paths
    if save_dicts:
        create_directory(os.path.join(path_dir, 'users_path_dicts', subnetwork))

    # trackers for statistics
    pos_paths_not_found = 0
    total_pos_interactions = 0
    num_neg_interactions = 0
    avg_num_pos_paths, avg_num_neg_paths = 0, 0
    print("path_dir is:" , path_dir)
    if path.exists(os.path.join(path_dir, f'test_path_length_{limit}.pkl')):
        print('file already exist')
        calc_all_paths = False
    else:
        print( 'file not existing yet')
        calc_all_paths = True
        if version == 'test': # create dict with key: (user,item) and value: # of paths
            test_num_paths = {}
    # initial users is limited (= limit)
    for user,pos_items in tqdm(list(user_item_split.items())[:limit]):
        total_pos_interactions += len(pos_items)
        item_to_paths, neg_items_with_paths = None, None
        cur_index = 0 # current index in negative list for adding negative interactions

        for pos_item in pos_items:
            interactions = [] # just used with finding test paths
            if item_to_paths is None:
                if version == "train": # initialize paths only for the first pos_org?
                    print("starting find_paths_user_to_items")
                    item_to_paths = find_paths_user_to_items(user, item_person, person_item, item_user_split,
                                                             user_item_split, 3, len_3_branch,
                                                             # data_masking
                                                             )
                    if long_paths:
                        item_to_paths_len5 = find_paths_user_to_items(user, item_person, person_item, item_user_split,
                                                                      user_item_split, 5, len_5_branch)
                        # item_to_paths_len5 = find_paths_user_to_items(user, item_person, person_item, item_user_split,
                        #                                               user_item_split, 5, len_5_branch,data_masking)
                else: # for testing we use entire org_user and user_org dictionaries
                    item_to_paths = find_paths_user_to_items(user, item_person, person_item, item_user_all,
                                                             user_item_all, 3, len_3_branch)
                    # item_to_paths = find_paths_user_to_items(user, item_person, person_item, item_user_all,
                    #                                          user_item_all, 3, len_3_branch,data_masking)
                    if long_paths:
                        item_to_paths_len5 = find_paths_user_to_items(user, item_person, person_item, item_user_all,
                                                                      user_item_all, 5, len_5_branch)
                        # item_to_paths_len5 = find_paths_user_to_items(user, item_person, person_item, item_user_all,
                        #                                               user_item_all, 5, len_5_branch,data_masking)
                # org_to_path includes both paths of 3 and 5
                if long_paths:
                    for org in item_to_paths_len5.keys():
                        item_to_paths[org].extend(item_to_paths_len5[org])

                # save dicts
                if save_dicts:
                    # create_directory(os.path.join(path_dir,'users_path_dicts',subnetwork))
                    users_dicts_dir = os.path.join(path_dir,'users_path_dicts',subnetwork,version)
                    create_directory(users_dicts_dir)
                    with open(os.path.join(users_dicts_dir,f'user_{user}.dict'), 'wb') as handle:
                        pickle.dump(item_to_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)


                # select negative paths - per user (out of all datasets)
                all_pos_items = set(user_item_all[user])
                items_with_paths = set(item_to_paths.keys())
                neg_items_with_paths = list(items_with_paths.difference(all_pos_items))
                # print("neg_items_with_paths" , neg_items_with_paths)

                top_neg_items = neg_items_with_paths.copy()
                random.shuffle(top_neg_items)

            # add paths for positive interaction (according train-test split)
            pos_paths = item_to_paths[pos_item]
            if len(pos_paths) > 0:
                if (samples != -1) and (random_sample is False):
                    if version == 'test' and calc_all_paths is True:
                        test_num_paths[(pos_paths[0][0][0], pos_paths[0][-1][0])]=test_num_paths.get(
                            (pos_paths[0][0][0], pos_paths[0][-1][0]),len(pos_paths))
                    pos_paths = sample_paths(pos_paths, samples)


                # reformat the paths, as such each path is a tuple of: (path, len(path))
                # 1 at the end represent it is positive
                interaction = (format_paths(pos_paths, e_to_ix, t_to_ix, r_to_ix), 1)
                if version == "train":
                    path_file.write(repr(interaction) + "\n")
                else:
                    interactions.append(interaction)
                avg_num_pos_paths += len(pos_paths)
            else:
                pos_paths_not_found += 1
                continue

            # add negative interactions that have paths
            found_all_samples = True
            for i in range(neg_samples):
                # check if not enough neg paths to complete
                if cur_index >= len(top_neg_items):
                    print("not enough neg paths, only found:", str(i))
                    found_all_samples = False
                    break
                neg_org = top_neg_items[cur_index]
                neg_paths = item_to_paths[neg_org]

                if (samples != -1) and (random_sample is False):
                    if version == 'test' and calc_all_paths is True:
                        test_num_paths[(neg_paths[0][0][0], neg_paths[0][-1][0])]=test_num_paths.get(
                            (neg_paths[0][0][0], neg_paths[0][-1][0]),len(neg_paths))
                    neg_paths = sample_paths(neg_paths, samples)
                # 0 at the end represent it is negative
                interaction = (format_paths(neg_paths, e_to_ix, t_to_ix, r_to_ix), 0)
                if version == "train":
                    path_file.write(repr(interaction) + "\n")
                else:
                    interactions.append(interaction)

                avg_num_neg_paths += len(neg_paths)
                num_neg_interactions += 1
                cur_index += 1

            if found_all_samples and version == "test":
                print('printed test_full_{user_limit}_samples{args.samples}_interactions.txt')
                path_file.write(repr(interactions) + "\n")
    avg_num_neg_paths = avg_num_neg_paths / num_neg_interactions
    avg_num_pos_paths = avg_num_pos_paths / (total_pos_interactions - pos_paths_not_found)

    print("number of pos paths attempted to find:", total_pos_interactions)
    print("number of pos paths not found:", pos_paths_not_found)
    print("avg num paths per positive interaction:", avg_num_pos_paths)
    print("avg num paths per negative interaction:", avg_num_neg_paths)

    if version == 'test' and calc_all_paths is True:
        with open(os.path.join(path_dir,f'test_path_length_{limit}.pkl'), 'wb') as handle:
            pickle.dump(test_num_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path_file.close()
    return


def load_string_to_ix_dicts(data_path):
    '''
    Loads the dictionaries mapping entity, relation, and type to id
    '''

    with open(data_path + f'_{consts.TYPE_TO_IX}', 'rb') as handle: # type_to_ix.dict
        type_to_ix = pickle.load(handle)
    with open(data_path + f'_{consts.RELATION_TO_IX}', 'rb') as handle: # relation_to_ix.dict
        relation_to_ix = pickle.load(handle)
    with open(data_path + f'_{consts.ENTITY_TO_IX}', 'rb') as handle: # entity_to_ix.dict
        entity_to_ix = pickle.load(handle)

    return type_to_ix, relation_to_ix, entity_to_ix


def load_rel_ix_dicts(data_path):
    '''
    Loads the relation dictionaries
    '''
    print(data_path)
    with open(data_path + f'_ix_{consts.ITEM_PERSON_DICT}', 'rb') as handle: # org_biz.dict
        org_biz = pickle.load(handle)
    with open(data_path + f'_ix_{consts.PERSON_ITEM_DICT}', 'rb') as handle: # biz_org.dict
        biz_org = pickle.load(handle)
    with open(data_path + f'_ix_{consts.ITEM_USER_DICT}', 'rb') as handle: # org_inv.dict
        org_inv = pickle.load(handle)
    with open(data_path + f'_ix_{consts.USER_ITEM_DICT}', 'rb') as handle: # inv_org.dict
        inv_org = pickle.load(handle)

    return org_biz, biz_org, org_inv, inv_org


def main(files_loc, params={}, data_loc=None):
    '''
    Main function for kprn model testing and training
    '''
    print("Main Loaded")
    print(f'Dataset: {consts.DATASET_DOMAIN}')
    print(f'files loc: {files_loc}')
    print(f'data loc: {data_loc}')
    random.seed(1)
    args = parse_args()
    if data_loc is None:
        data_loc = files_loc

    # overwrite args values (in case of HPT)
    if len(params) > 0:
        for k, v in params.items():
            setattr(args, k, v)
    print(f'Item-2-item: {args.item_to_item}')

    PROCESSED_DATA_DIR = os.path.join(data_loc, 'Data', consts.DATASET_DOMAIN,
                                      'processed_data')
    # rename model name
    args.model = consts.DATASET_DOMAIN + '_' + args.subnetwork + '_' + str(args.user_limit) + '_' + args.model
    model_path = os.path.join(files_loc, 'model', args.model_name, args.model)
    print("model path: ", model_path)

    t_to_ix, r_to_ix, e_to_ix = load_string_to_ix_dicts(data_path=os.path.join(PROCESSED_DATA_DIR,
                                                                               consts.ITEM_IX_MAPPING_DIR,
                                                                               args.subnetwork))
    item_person, person_item, item_user, user_item = load_rel_ix_dicts(data_path=os.path.join(PROCESSED_DATA_DIR,
                                                                                              consts.ITEM_IX_DATA_DIR,
                                                                                              args.subnetwork))

    with open(os.path.join(PROCESSED_DATA_DIR, consts.ITEM_IX_DATA_DIR, 'items_for_pred.pkl'), 'rb') as handle:
        items_for_pred = pickle.load(handle)

    if args.model_name == 'KPRN':
        model = KPRN(e_emb_dim=consts.ENTITY_EMB_DIM, t_emb_dim=consts.TYPE_EMB_DIM, r_emb_dim=consts.REL_EMB_DIM,
                     hidden_dim=consts.HIDDEN_DIM,e_vocab_size=len(e_to_ix),t_vocab_size=len(t_to_ix),
                     r_vocab_size=len(r_to_ix),tagset_size=consts.TAG_SIZE,no_rel=args.no_rel)
    else: # KPAN
        print(f"sending to KPAN, init_mf_embeddings is set to {args.init_mf_embeddings}")
        model = KPAN(e_emb_dim=consts.ENTITY_EMB_DIM, t_emb_dim=consts.TYPE_EMB_DIM,
                     r_emb_dim=consts.REL_EMB_DIM, hidden_dim=consts.HIDDEN_DIM, e_vocab_size=len(e_to_ix),
                     t_vocab_size=len(t_to_ix), r_vocab_size=len(r_to_ix), tagset_size=consts.TAG_SIZE,
                     no_rel=args.no_rel, person_item_dict=person_item, dropout=args.dropout,
                     nhead=args.nhead, nlayers=1, entities_agg=args.entity_agg,
                     init_mf_embedding=args.init_mf_embeddings, add_path_length=args.add_path_length,
                     path_agg=args.path_agg_method, path_nhead=args.path_nhead,
                     mf_path=os.path.join(PROCESSED_DATA_DIR, r'mf', r'mf_initial')
                     )

    data_ix_path = os.path.join(PROCESSED_DATA_DIR, consts.ITEM_IX_DATA_DIR + args.subnetwork)
    if args.random_samples:
        args.kg_path_file = args.subnetwork + '_' + str(args.user_limit) + '_' + args.kg_path_file
    # below is the file name when we want same paths for all epochs
    else:
        args.kg_path_file = args.subnetwork + '_' + str(args.user_limit) + '_samples' + str(
            args.samples) + '_' + args.kg_path_file
    train_losses = [10000]

    if args.train:
        print(f"Training Starting - {args.model_name}")
        train_path = 'train_' + args.kg_path_file # separate interactions.txt

        # check if paths were already found
        if path.exists(os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR, train_path)):
            args.find_paths = False
        else:
            args.find_paths = True

        # either load interactions from disk, or run path extraction algorithm
        if args.find_paths:
            print("Finding paths")
            print("Directory path: ", data_ix_path)
            with open(data_ix_path + f'_train_ix_{consts.USER_ITEM_DICT}', 'rb') as handle:
                user_item_train = pickle.load(handle)
            with open(data_ix_path + f'_train_ix_{consts.ITEM_USER_DICT}', 'rb') as handle:
                item_user_train = pickle.load(handle)
            print("Load data")
            load_data(item_person=item_person, person_item=person_item, user_item_all=user_item,
                      item_user_all=item_user, item_user_split=item_user_train, user_item_split=user_item_train,
                      neg_samples=consts.NEG_SAMPLES_TRAIN, e_to_ix=e_to_ix, t_to_ix=t_to_ix, r_to_ix=r_to_ix,
                      path_dir=os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR), kg_path_file=train_path,
                      subnetwork=args.subnetwork, len_3_branch=consts.LEN_3_BRANCH,
                      len_5_branch=consts.LEN_5_BRANCH_TRAIN, limit=args.user_limit, version="train",
                      samples=args.samples, save_dicts=False, long_paths=args.long_paths,
                      random_sample=args.random_samples,
                      # data_masking=args.data_masking
                      )

        if args.validation:
            if args.random_samples:
                path_split = args.kg_path_file.split('_')
                path_split.insert(2, f'samples{args.samples}')
                kg_path = '_'.join(path_split)
                test_path = 'test_' + kg_path
            else:
                test_path = 'test_' + args.kg_path_file
            validation_path = os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR, test_path)
            val_dataset = ValidationData(path_file=validation_path, user_limit=args.user_limit)
            val_dataloader = DataLoader(dataset=val_dataset, shuffle=True)
        else:
            print("no validation run")
            val_dataloader = None
        print(os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR, train_path))
        model,train_losses,val_losses = train(model, os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR, train_path),
                                       args.batch_size, args.epochs, model_path, args.load_checkpoint, args.not_in_memory,
                                       args.lr, args.l2_reg, args.gamma, args.no_rel, args.samples, args.random_samples,
                                       args.model_name, args.validation, val_dataloader,args.path_agg_method)

        if np.isnan(train_losses[-1]):
            args.eval = False
            print( "Unsuccessful training" )

    if args.eval:
        print("Evaluation Starting")
        if not args.item_to_item:
            if not args.validation:
                if args.random_samples:
                    print("1111")
                    path_split = args.kg_path_file.split('_')
                    print(f'args.kg_path_file is now equal : {args.kg_path_file}')
                    path_split.insert(2, f'samples{args.samples}')
                    kg_path = '_'.join(path_split)
                    test_path = 'test_' + kg_path
                    print(f'random_samples = True --> test_path is now equal {test_path}')
                else:
                    test_path = 'test_' + args.kg_path_file
        else:
            test_path = f'test_{args.subnetwork}_item_to_item_interactions.txt'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device is", device)
        print(model_path)
        print("find_paths :" , args.find_paths)

        try:
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print('Untrained model')
            return train_losses[-1]
        finally:
            # Save scores + args
            print('save args')
            output_path = os.path.join(files_loc, 'Results', args.model_name, consts.DATASET_DOMAIN,
                                       args.subnetwork)
            create_directory(dir=output_path)
            # create folder per experiment
            experiments = [x for x in os.listdir(output_path)]
            if len(experiments) == 0:
                exp_num = 0
            else:
                exp_num = max([int(x.split('_')[1]) for x in experiments]) + 1
            output_path_exp = os.path.join(output_path, f'exp_{exp_num}')
            create_directory(dir=output_path_exp)
            # save args
            with open(os.path.join(output_path_exp, 'model_args.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

        model.eval()
        if args.path_agg_method == 'attention':
            for param in model.attention_model.parameters():
            # for param in model.path_attention.parameters():
                param.requires_grad = False
        if args.path_agg_method == 'statistics':
            for param in model.linear_score.parameters():
                param.requires_grad = False
        model = model.to(device)

        # check if paths were already found
        if path.exists(os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR, test_path)):
            args.find_paths = False
        else:
            args.find_paths = True

        if args.find_paths:
            print("Finding Paths")

            with open(data_ix_path + f'_test_ix_{consts.USER_ITEM_DICT}', 'rb') as handle:
                user_item_test = pickle.load(handle)
            with open(data_ix_path + f'_test_ix_{consts.ITEM_USER_DICT}', 'rb') as handle:
                item_user_test = pickle.load(handle)

            # load_data(item_person=item_person, person_item=person_item, user_item_all=user_item,
            #           item_user_all=item_user, item_user_split=item_user_test, user_item_split=user_item_test,
            #           neg_samples=consts.NEG_SAMPLES_TEST, e_to_ix=e_to_ix, t_to_ix=t_to_ix, r_to_ix=r_to_ix,
            #           path_dir=os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR), kg_path_file=test_path,
            #           subnetwork=args.subnetwork, len_3_branch=consts.LEN_3_BRANCH,
            #           len_5_branch=consts.LEN_5_BRANCH_TEST, limit=args.user_limit, version="test",
            #           samples=args.samples, long_paths=args.long_paths, random_sample=False,
            #           items_for_pred=items_for_pred)
            load_data(item_person=item_person, person_item=person_item, user_item_all=user_item,
                      item_user_all=item_user, item_user_split=item_user_test, user_item_split=user_item_test,
                      neg_samples=consts.NEG_SAMPLES_TEST, e_to_ix=e_to_ix, t_to_ix=t_to_ix, r_to_ix=r_to_ix,
                      path_dir=os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR), kg_path_file=test_path,
                      subnetwork=args.subnetwork, len_3_branch=consts.LEN_3_BRANCH,
                      len_5_branch=consts.LEN_5_BRANCH_TEST, limit=args.user_limit, version="test",
                      samples=args.samples, long_paths=args.long_paths, random_sample=False,
                     )
            # Random sample is False in test for all, since the data for test must prepare here.
            # Therefore, we want the sampling in any case we don't have -1

        if args.np_baseline:
            path_length_baseline_results = []
            # calculate popularity score (y_pred)
            popularity = PopularityBL()
            popularity.calculate_popularity(data_ix_path=PROCESSED_DATA_DIR, user_limit=args.user_limit)
            popularity_baseline_results = []

        if not args.validation:
            start_val = time.time()
            print ( "Valid path = ", os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR, test_path) )
            val_dataset = ValidationData(path_file=os.path.join(PROCESSED_DATA_DIR, consts.PATH_DATA_DIR, test_path),
                                         user_limit=args.user_limit)
            print(len(val_dataset))
            val_dataloader = DataLoader(dataset=val_dataset, shuffle=True)#, batch_size=6)
            print(f'Finish load test set - {timedelta(seconds=(time.time() - start_val))}')

        max_k = 10
        file_path = os.path.join(PROCESSED_DATA_DIR,consts.PATH_DATA_DIR,test_path)

        # EVALUATION with dataloader
        print('start predict')
        start_pred = time.time()
        if args.np_baseline:
            all_results, path_length_baseline_results, popularity_baseline_results, \
            hit_at_k_scores, ndcg_at_k_scores = predict_dataloader(model, val_dataloader, device, args.no_rel,
                                                                   args.gamma, args.np_baseline, args.path_agg_method,
                                                                   False,args.item_to_item)
        else:
            all_results = predict_dataloader(model, val_dataloader, device, args.no_rel, args.gamma, args.np_baseline,
                                             args.path_agg_method)
        print(f'Finish predict - {timedelta(seconds=(time.time() - start_pred))}')

        if args.item_to_item:
            print('Evaluation is not needed for item to item eval')
            return min(train_losses)

        ### EVALUATION PER USER ###
        df_model_results = prep_for_evaluation(all_results)
        df_scores_per_user, mpr_score_per_user = calc_scores_per_user(df=df_model_results, max_k=max_k,
                                                                      model_nm='model',mpr_metric=True)
        model_scores_rank_agg = aggregate_results(df=df_scores_per_user,group_by=['model','rank'])

        if args.np_baseline:
            # path length
            df_bl_length_results = prep_for_evaluation(path_length_baseline_results)
            df_bl_length_scores_per_user = calc_scores_per_user(df=df_bl_length_results, max_k=max_k,
                                                                model_nm='path_length')
            bl_length_scores_rank_agg = aggregate_results(df=df_bl_length_scores_per_user,group_by=['model','rank'])

            # popularity
            bl_popularity_y_pred = popularity.predict(processed_test=pd.DataFrame(popularity_baseline_results,
                                                                                      columns=['user','item','label']))
            df_bl_popularity_results = prep_for_evaluation(bl_popularity_y_pred)
            df_bl_popularity_scores_per_user = calc_scores_per_user(df=df_bl_popularity_results, max_k=max_k,
                                                                    model_nm='popularity')
            bl_popularity_scores_rank_agg = aggregate_results(df=df_bl_popularity_scores_per_user, group_by=['model', 'rank'])

        for idx, row in model_scores_rank_agg.iterrows():
            print(f"Average hit@K for k={int(row['rank'])} is {round(row['hit'],4)}")
            print(f"Average ndcg@K for k={int(row['rank'])} is {round(row['ndcg'],4)}")

        if args.np_baseline:
            scores_rank_agg = pd.concat([model_scores_rank_agg, bl_popularity_scores_rank_agg, bl_length_scores_rank_agg])
        else:
            scores_rank_agg = model_scores_rank_agg.copy()


        print('Save the results')
        # combine both results
        scores_rank_agg.rename({'rank':'k'},axis=1,inplace=True)
        all_scores = scores_rank_agg.copy()
        all_scores.to_csv(os.path.join(output_path_exp,'all_model_scores.csv'), index=False) # save model results
        pd.DataFrame(all_results, columns=['user', 'item', 'y_pred', 'label']).to_csv(os.path.join(output_path_exp,'model_predictions.csv'), index=False) # save model predictions
        mpr_score_per_user.to_csv(os.path.join(output_path_exp,'mpr_per_user.csv'), index=False)
        print(f'MPR: {mpr_score_per_user.mpr.mean()}')

        if args.train:
            if args.validation:
                pd.DataFrame({'train':train_losses, 'val':val_losses}).to_csv((os.path.join(output_path_exp,'train_loss.csv')))
            else:
                pd.DataFrame({'train': train_losses}).to_csv((os.path.join(output_path_exp, 'train_loss.csv')))

    return min(train_losses)

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    main(files_loc=repo_root, data_loc=repo_root)