import sys
from os import path
from pathlib import Path
import pickle
import random
import constants.consts as consts
from collections import defaultdict
import copy


class PathState:
    def __init__(self, path, length, entities):
        self.path = path    # array of [entity, entity type, relation to next] triplets
        self.length = length
        self.entities = entities    # set to keep track of the entities alr in the path to avoid cycles

def p(*parts) -> Path:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    return REPO_ROOT.joinpath(*parts)
def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)
    return index_list[:nums]


def apply_mask(path):
    """
    Optional utility to apply masking.
    """
    masked_path = []  # Initialize an empty list to store the masked path

    for entity, entity_type, relation in path:
        # Check if the entity is an ID that needs to be masked (e.g., not a relation like 'END_REL')
        if isinstance(entity, int):
            # Replace the entity with "1"
            masked_entity = 1
        else:
            # If it's not an ID to be masked, keep it unchanged
            masked_entity = entity

        # Append the masked entity, entity type, and relation to the masked path
        masked_path.append([masked_entity, entity_type, relation])

    return masked_path

def find_paths_user_to_items(start_user, item_person, person_item, item_user, user_item, max_length, sample_nums):
    # def find_paths_user_to_items(start_user, item_person, person_item, item_user, user_item, max_length, sample_nums,
    #                              enable_masking=False):
    '''
    Finds sampled paths of max depth from a user to a sampling of organizations
    '''
    item_to_paths = defaultdict(list)
    stack = []
    start = PathState([[start_user, consts.USER_TYPE, consts.END_REL]], 0, {start_user})
    stack.append(start)
    while len(stack) > 0:
        front = stack.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        # add path to org_to_inv_paths dict, just want paths of max_length rn since either length 3 or 5
        # if the path is over (by an organization) - add to paths
        if type == consts.ITEM_TYPE and front.length == max_length:
            item_to_paths[entity].append(front.path)

          # limit of the path, without organization at the end
        if front.length == max_length:
            continue

        # the first path starts from here, and also for other users entities
        if type == consts.USER_TYPE and entity in user_item:
            org_list = user_item[entity]
            # sample positive orgs based on indices
            index_list = get_random_index(sample_nums, len(org_list))
            for index in index_list:
                org = org_list[index]
                # if the org is not already in the path
                if org not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    # replace the last relation with the correct relation: user-song
                    new_path[-1][2] = consts.USER_ITEM_REL
                    new_path.append([org, consts.ITEM_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{org})
                    stack.append(new_state)

        # if the node is org
        elif type == consts.ITEM_TYPE:
            # if this org exist in org for investors
            if entity in item_user:
                user_list = item_user[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_USER_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{user})
                        stack.append(new_state)
            # if this org exist in org of investors
            if entity in item_person:
                person_list = item_person[entity]
                index_list = get_random_index(sample_nums, len(person_list))
                for index in index_list:
                    person = person_list[index]
                    if person not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_PERSON_REL
                        new_path.append([person, consts.PERSON_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{person})
                        stack.append(new_state)

        # if node is a person
        elif type == consts.PERSON_TYPE and entity in person_item:
            org_list = person_item[entity]
            index_list = get_random_index(sample_nums, len(org_list))
            for index in index_list:
                org = org_list[index]
                if org not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = consts.PERSON_ITEM_REL
                    new_path.append([org, consts.ITEM_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{org})
                    stack.append(new_state)
    return item_to_paths

def find_negative_paths_len_2(start_user, neg_list, max_length):
    '''
    Finds sampled paths of max depth from a user to a sampling of paths
    '''
    item_to_paths = defaultdict(list)
    print('im here - find_negative_paths_len_2 ')
    stack = []
    for neg_item in neg_list:
        start = PathState([[start_user, consts.USER_TYPE, consts.USER_ITEM_REL]], 0, {start_user})
        new_path = copy.deepcopy(start.path)
        new_path.append([neg_item, consts.ITEM_TYPE, consts.END_REL])
        new_state = PathState(new_path, start.length + 1, start.entities | {neg_item})
        stack.append(new_state)

    while len(stack) > 0:
        front = stack.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        if type == consts.ITEM_TYPE and front.length == max_length:
            item_to_paths[entity].append(front.path)

    return item_to_paths

def main():
    data_ix_dir = p("Data", consts.DATASET_DOMAIN, "processed_data", consts.ITEM_IX_DATA_DIR)
    prefix = data_ix_dir / "full"  # adjust if your naming differs

    with open(str(prefix) + "_ix_org_biz.dict", "rb") as handle:
        org_biz = pickle.load(handle)
    with open(str(prefix) + "_ix_biz_org.dict", "rb") as handle:
        biz_org = pickle.load(handle)
    with open(str(prefix) + "_ix_org_inv.dict", "rb") as handle:
        org_inv = pickle.load(handle)
    with open(str(prefix) + "_ix_inv_org.dict", "rb") as handle:
        inv_org = pickle.load(handle)

    print(find_paths_user_to_items(0, org_biz, biz_org, org_inv, inv_org, 3, 1))

if __name__ == "__main__":
    main()
