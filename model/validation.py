import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from pathlib import Path
import pickle


class ValidationData(Dataset):
    def __init__(self, path_file, user_limit, batch_size=5):
        self.data = []
        data_dic = {}
        num_batch = -1
        with open(path_file, 'r') as file:
            for line in file:
                self.data.append(eval(line.rstrip("\n")))
            self.num_interactions = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def val_sort_batch(batch, indexes, lengths):
    '''
    sorts a batch of paths by path length, in decreasing order
    return: ordered paths, ordered object indices, ordered lengths
    '''
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    indexes_tensor = indexes[perm_idx]
    return seq_tensor, indexes_tensor, seq_lengths

def val_my_collate(batch):
    '''
    Custom dataloader collate function since we have tuples of lists of paths
    '''
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def validate(model, val_dataloader, device, no_rel, gamma, path_aggregation):
    '''
    -outputs predicted scores for the input test data
    -formatted_data is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded to ensure same overall length
    -Since we are evaluating we ignore the tag here
    '''

    prediction_scores = None
    targets = []

    for paths, lengths, inter_id, target in val_dataloader:
        t_paths = torch.tensor(paths, dtype=torch.long)
        # t_lengths = torch.tensor(lengths, dtype=torch.long)
        t_lengths = torch.tensor([lengths] * len(paths), dtype=torch.long)
        t_inter_ids = torch.tensor([inter_id] * len(paths), dtype=torch.long)

        # sort based on path lengths, largest first, so that we can pack paths
        s_path_batch, s_inter_ids, s_lengths = val_sort_batch(t_paths, t_inter_ids, t_lengths)
        tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel)
        targets.append(target)

        # weighted pooled scores for this interaction - tag scores is for 1 user-item
        if path_aggregation == 'weighted_pooling':
            agg_score = model.weighted_pooling(tag_scores, gamma=gamma)
        else:
            agg_score,_ = model.calc_paths_attention(tag_scores)

        # unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
        pooled_scores = agg_score.unsqueeze(0)

        if path_aggregation == 'weighted_pooling':
            pooled_scores = F.log_softmax(pooled_scores, dim=1)

        if prediction_scores is None:
            prediction_scores = pooled_scores
        else:
            prediction_scores = torch.cat((prediction_scores, pooled_scores))

    return prediction_scores,targets

class ValInteractionData(Dataset):
    def __init__(self, formatted_data):
        self.data = formatted_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def validate_B(model,formatted_data,batch_size,device,no_rel,gamma):
        prediction_scores = None
        interaction_data = ValInteractionData(formatted_data)
        # shuffle false since we want data to remain in order for comparison
        val_loader = DataLoader(dataset=interaction_data, collate_fn=val_my_collate, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (interaction_batch, _) in val_loader:
                # construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
                paths = []
                lengths = []
                inter_ids = []
                num_of_paths = []
                for inter_id, interaction_paths in enumerate(interaction_batch):
                    for path, length in interaction_paths:
                        paths.append(path)
                        lengths.append(length)
                    inter_ids.extend([inter_id for i in range(len(interaction_paths))])

                inter_ids = torch.tensor(inter_ids, dtype=torch.long)
                paths = torch.tensor(paths, dtype=torch.long)
                lengths = torch.tensor(lengths, dtype=torch.long)

                # sort based on path lengths, largest first, so that we can pack paths
                s_path_batch, s_inter_ids, s_lengths = val_sort_batch(paths, inter_ids, lengths)

                tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel, num_of_paths)


                # Get weighted pooling of scores over interaction id groups
                start = True
                for i in range(len(interaction_batch)):
                    # get inds for this interaction
                    inter_idxs = (s_inter_ids == i).nonzero().squeeze(1)

                    # weighted pooled scores for this interaction
                    pooled_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma)

                    if start:
                        # unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                        pooled_scores = pooled_score.unsqueeze(0)
                        start = not start
                    else:
                        pooled_scores = torch.cat((pooled_scores, pooled_score.unsqueeze(0)), dim=0)

                if not prediction_scores:
                    prediction_scores = F.softmax(pooled_scores, dim=1)
                else:
                    prediction_scores = torch.cat((prediction_scores,F.softmax(pooled_scores, dim=1)))

        return prediction_scores

