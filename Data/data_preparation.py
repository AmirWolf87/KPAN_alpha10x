import pandas as pd
import pickle
import argparse
from collections import defaultdict
import random
from tqdm import tqdm
import sys
from os import path, mkdir
import os
from pathlib import Path
import constants.consts as consts


from pathlib import Path

REPO_ROOT = Path(__file__).resolve()
REPO_ROOT = REPO_ROOT.parents[1] if REPO_ROOT.parent.name in {"baseline", "src", "scripts", "Data"} else REPO_ROOT.parent


def p(*parts) -> Path:
    return REPO_ROOT.joinpath(*parts)

def create_directory(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orgs_file',
                        default=str(p('Data', consts.ITEM_DATASET_DIR, 'updated_organizations.csv')),
                        help='Path to the CSV file containing organizations information')
    parser.add_argument('--interactions_file',
                        default=str(p('Data',  consts.ITEM_DATASET_DIR, 'cleaned_training.csv')),
                        help='Path to the CSV file containing person organization interactions')
    parser.add_argument('--subnetwork',
                        default='full',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to form from the full KG')
    # Additional arguments (following bug)
    parser.add_argument('--mode',
                        default='client')
    parser.add_argument('--port',
                        default=51131)
    return parser.parse_args()



def make_biz_list(row):
    biz_set = set()
    # if not isinstance(row['name'], float):
    #     for x in row['name'].split('|'):
    #         biz_set.add(x.strip())
    # if not isinstance(row['crb_rank'], float):
    #     for x in row['crb_rank'].split('|'):
    #         biz_set.add(x.strip())
    if not isinstance(row['industry_id'], float):
        for x in row['industry_id'].split('|'):
            biz_set.add(x.strip())
    # # if not isinstance(row['revenue_range'], float):
    # #     for x in row['revenue_range'].split('|'):
    # #         biz_set.add(x.strip())
    if not isinstance(row['scientific_concept'], float):
        for x in row['scientific_concept'].split('|'):
            biz_set.add(x.strip())
    if not isinstance(row['country_code'], float):
         for x in row['country_code'].split('|'):
            biz_set.add(x.strip())
    # # if not isinstance(row['valuation_price_usd'], float):
    # #     for x in row['valuation_price_usd'].split('|'):
    # #         business_set.add(x.strip())
    # if not isinstance(row['money_raised_usd'], float):
    #     for x in row['money_raised_usd'].split('|'):
    #         biz_set.add(x.strip())
    if not isinstance(row['CEO'], float):
        for x in row['CEO'].split('|'):
            biz_set.add(x.strip())
    if not isinstance(row['CFO'], float):
        for x in row['CFO'].split('|'):
            biz_set.add(x.strip())
    if not isinstance(row['CTO'], float):
        for x in row['CTO'].split('|'):
            biz_set.add(x.strip())
    if not isinstance(row['president'], float):
        for x in row['president'].split('|'):
            biz_set.add(x.strip())
    if not isinstance(row['founder'], float):
        for x in row['founder'].split('|'):
            biz_set.add(x.strip())
    if not isinstance(row['COO'], float):
        for x in row['COO'].split('|'):
            biz_set.add(x.strip())
    if not isinstance(row['bod'], float):
        for x in row['bod'].split('|'):
            biz_set.add(x.strip())
    return list(biz_set)


def orgs_data_prep(orgs_file, interactions_file, export_dir):
    '''
    :return: Write out 4 python dictionaries for the edges of KG
    '''

    # orgs = pd.read_csv(orgs_file, on_bad_lines='skip')
    orgs = pd.read_csv(orgs_file) #, usecols=range(17))
    print("# of rows: ", len(orgs))
    interactions = pd.read_csv(interactions_file)
    # org_biz.dict
    # dict where key = alpha_id, value = list of business_related_persons (CEO,CTO,founder,COO, bod_member1, bod_member2 , bod_member3 ) of the organization
    # biz = orgs[['alpha_id', 'CEO', 'CTO', 'founder', 'COO', 'bod_member1', 'bod_member2', 'bod_member3',]] @Changed  as experiment to the below
    # biz = orgs[['alpha_id', 'name', 'industry', 'country_code', 'CEO', 'CTO', 'founder', 'COO']]
    biz = orgs[['alpha_id','industry_id','scientific_concept','country_code', 'CEO', 'CFO', 'COO', 'CTO', 'president', 'founder', 'bod']]
    biz_list = biz.apply(lambda x: make_biz_list(x), axis=1)
    # print("biz list : ", biz_list)
    org_biz = pd.concat([orgs['alpha_id'], biz_list], axis=1)

    org_biz.columns = ['alpha_id', 'biz_list']
    print("org_biz: ", org_biz)
    org_biz_dict = org_biz.set_index('alpha_id')['biz_list'].to_dict()
    print("org_biz_dict: ", org_biz_dict)
    with open(os.path.join(export_dir, consts.ITEM_PERSON_DICT), 'wb') as handle:
        pickle.dump(org_biz_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # biz_org.dict
    # dict where key = a person, value = list of organizations related to this person
    biz_org_dict = {}
    for row in org_biz.iterrows():
        for biz in row[1]['biz_list']:
            if biz not in biz_org_dict:
                biz_org_dict[biz] = []
            biz_org_dict[biz].append(row[1]['alpha_id'])
    with open(os.path.join(export_dir, consts.PERSON_ITEM_DICT), 'wb') as handle:
        pickle.dump(biz_org_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # org_inv.dict
    # dict where key = org_id, value = list of inv_ids
    org_inv = interactions[['org_id', 'inv_id']].set_index('org_id').groupby('org_id')['inv_id'].apply(list).to_dict()
    # person_id is the user_id
    with open(os.path.join(export_dir, consts.ITEM_USER_DICT), 'wb') as handle:
        pickle.dump(org_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # inv_org.dict
    # dict where key = person_id, value = list of org_id
    inv_org = interactions[['inv_id', 'org_id']].set_index('inv_id').groupby('inv_id')['org_id'].apply(list).to_dict()
    with open(os.path.join(export_dir, consts.USER_ITEM_DICT), 'wb') as handle:
        pickle.dump(inv_org, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # inv_org_tuple.txt
    # numpy array of [person_id, org_id] pairs sorted in the order of user_id
    inv_org_tuple = interactions[['inv_id', 'org_id']].sort_values(by='inv_id').to_string(header=False, index=False,
                                                                                         index_names=False).split('\n')
    inv_org_tuple = [row.split() for row in inv_org_tuple]
    with open(os.path.join(export_dir,'inv_org_tuple.txt'), 'wb') as handle:
        pickle.dump(inv_org_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_subnetwork(network_type, dir, factor=0.1):
    dir = str(Path(dir)) + os.sep
    if network_type == 'full':
        return

    # Load Data

    with open(dir + consts.ITEM_USER_DICT, 'rb') as handle:
        org_inv = pickle.load(handle)
    with open(dir + consts.USER_ITEM_DICT, 'rb') as handle:
        inv_org = pickle.load(handle)
    with open(dir + consts.ITEM_PERSON_DICT, 'rb') as handle:
        org_biz = pickle.load(handle)
    with open(dir + consts.PERSON_ITEM_DICT, 'rb') as handle:
        biz_org = pickle.load(handle)
    org_inv = defaultdict(list, org_inv)
    org_biz = defaultdict(list, org_biz)
    inv_org = defaultdict(list, inv_org)
    biz_org = defaultdict(list, biz_org)

    # Sort Nodes By Degree in descending order

    # key: org, value: number of investors + number of person relating to this entity
    org_degree_dict = {}
    for (k, v) in org_inv.items():
        org_degree_dict[k] = v
    for (k, v) in org_biz.items():
        # if the song is already exist in dict, update the value (relations) by adding relevant persons
        if k in org_degree_dict.keys():
            org_degree_dict[k] = org_degree_dict[k] + v
        else:
            org_degree_dict[k] = v
    org_degree = [(k, len(v)) for (k, v) in org_degree_dict.items()]
    org_degree.sort(key=lambda x: -x[1])

    # key: biz, value: number of orgs they create
    biz_degree = [(k, len(v)) for (k, v) in biz_org.items()]
    biz_degree.sort(key=lambda x: -x[1])

    # key: investor, value: number of orgs they invested in
    inv_degree = [(k, len(v)) for (k, v) in inv_org.items()]
    inv_degree.sort(key=lambda x: -x[1])

    # Construct Subnetworks

    # find the nodes
    print('finding the nodes...')
    # if network_type == 'dense':
    if 'dense' in network_type:
        if 'top' in network_type:
            start = 1000
        else:
            start = 0
        org_nodes_holder = org_degree[start:int(len(
            org_degree) * factor)]  # alpha_id is the first item in the tuple element of the returned list
        org_nodes = [node_holder[0] for node_holder in org_nodes_holder]

        inv_nodes_holder = inv_degree[start:int(len(inv_degree) * factor)]
        inv_nodes = [node_holder[0] for node_holder in inv_nodes_holder]

        biz_nodes_holder = biz_degree[start:int(len(biz_degree) * factor)]
        biz_nodes = [node_holder[0] for node_holder in biz_nodes_holder]

    elif network_type == 'rs':
        org_nodes_holder = random.sample(org_degree, int(len(
            org_degree) * factor))  # alpha_id is the first item in the tuple element of the returned list
        org_nodes = [node_holder[0] for node_holder in org_nodes_holder]

        inv_nodes_holder = random.sample(inv_degree, int(len(inv_degree) * factor))
        inv_nodes = [node_holder[0] for node_holder in inv_nodes_holder]

        biz_nodes_holder = random.sample(biz_degree, int(len(biz_degree) * factor))
        biz_nodes = [node_holder[0] for node_holder in biz_nodes_holder]

    elif network_type == 'sparse':
        org_nodes_holder = org_degree[-int(len(
            org_degree) * factor):]  # alpha_id is the first item in the tuple element of the returned list
        org_nodes = [node_holder[0] for node_holder in org_nodes_holder]

        inv_nodes_holder = inv_degree[-int(len(inv_degree) * factor):]
        inv_nodes = [node_holder[0] for node_holder in inv_nodes_holder]

        biz_nodes_holder = biz_degree[-int(len(biz_degree) * factor):]
        biz_nodes = [node_holder[0] for node_holder in biz_nodes_holder]

    nodes = org_nodes + inv_nodes + biz_nodes
    print(f'The {network_type} subnetwork has {len(nodes)} nodes: {len(org_nodes)} organizations, {len(inv_nodes)} investors, {len(biz_nodes)} businesses.')

    # find the edges
    # (node1, node2) and (node2, node1) both exist
    edges_type1 = []  # a list of pairs (song, user)
    edges_type2 = []  # a list of pairs (song, person)
    edges_type3 = []  # a list of pairs (user, song)
    edges_type4 = []  # a list of pairs (person, song)
    nodes_set = set(nodes)

    # for each node in the nodes set, check the intersection with their info in the datasets
    for i in tqdm(nodes_set):  # (node1, node2) and (node2, node1) both exist
        connect_1 = set(org_inv[i]).intersection(nodes_set) # song-user
        for j in connect_1:
            edges_type1.append((i, j))

        connect_2 = set(org_biz[i]).intersection(nodes_set) # song-person
        for j in connect_2:
            edges_type2.append((i, j))

        connect_3 = set(inv_org[i]).intersection(nodes_set) # user-song
        for j in connect_3:
            edges_type3.append((i, j))

        connect_4 = set(biz_org[i]).intersection(nodes_set) # person-song
        for j in connect_4:
            edges_type4.append((i, j))

    edges = edges_type1 + edges_type2 + edges_type3 + edges_type4
    print(f'The {network_type} subnetwork has {len(edges)} edges.')

    # Export the Subnetworks

    # <NETWORK_TYPE>_org_inv.dict
    # key: org, value: a list of investors
    org_inv_dict = defaultdict(list)
    for edge in edges_type1:
        org = edge[0]
        inv = edge[1]
        org_inv_dict[org].append(inv)
    org_inv_dict = dict(org_inv_dict)
    prefix = dir + network_type + '_'
    with open(prefix + consts.ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(org_inv_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_org_biz.dict
    # key: org, value: a list of businesses
    org_biz_dict = defaultdict(list)
    for edge in edges_type2:
        org = edge[0]
        biz = edge[1]
        org_biz_dict[org].append(biz)
    org_biz_dict = dict(org_biz_dict)
    with open(prefix + consts.ITEM_PERSON_DICT, 'wb') as handle:
        pickle.dump(org_biz_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_inv_org.dict
    # key: investor, value: a list of organizations
    inv_org_dict = defaultdict(list)
    for edge in edges_type3:
        inv = edge[0]
        org = edge[1]
        inv_org_dict[inv].append(org)
    inv_org_dict = dict(inv_org_dict)
    with open(prefix + consts.USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(inv_org_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_biz_org.dict
    # key: person, value: a list of organization
    biz_org_dict = defaultdict(list)
    for edge in edges_type4:
        biz = edge[0]
        org = edge[1]
        biz_org_dict[biz].append(org)
    biz_org_dict = dict(biz_org_dict)
    with open(prefix + consts.PERSON_ITEM_DICT, 'wb') as handle:
        pickle.dump(biz_org_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_to_ids(entity_to_ix, rel_dict, start_type, end_type):
    new_rel = {}
    for key, values in rel_dict.items():
        key_id = entity_to_ix[(key, start_type)]
        value_ids = []
        for val in values:
            value_ids.append(entity_to_ix[(val, end_type)])
        new_rel[key_id] = value_ids
    return new_rel

def ix_mapping(network_type, import_dir, export_dir, mapping_export_dir):
    import_dir = str(Path(import_dir)) + os.sep
    export_dir = str(Path(export_dir)) + os.sep
    mapping_export_dir = str(Path(mapping_export_dir)) + os.sep
    pad_token = consts.PAD_TOKEN
    type_to_ix = {'business': consts.PERSON_TYPE, 'investor': consts.USER_TYPE, 'organization': consts.ITEM_TYPE,
                  pad_token: consts.PAD_TYPE}
    relation_to_ix = {'org_biz': consts.ITEM_PERSON_REL, 'biz_org': consts.PERSON_ITEM_REL,
                      'inv_org': consts.USER_ITEM_REL, 'org_inv': consts.ITEM_USER_REL, '#UNK_RELATION': consts.UNK_REL,
                      '#END_RELATION': consts.END_REL, pad_token: consts.PAD_REL}

    # entity vocab set is combination of organizatios, investors, and biz
    if network_type == 'full':
        org_data_prefix = import_dir
    else:
        org_data_prefix = import_dir + network_type + '_'
    with open(Path(org_data_prefix) / consts.ITEM_USER_DICT, 'rb') as handle:
        org_inv = pickle.load(handle)
    with open(Path(org_data_prefix) / consts.ITEM_PERSON_DICT, 'rb') as handle:
        org_biz = pickle.load(handle)
    with open(Path(org_data_prefix) / consts.USER_ITEM_DICT, 'rb') as handle:
        inv_org = pickle.load(handle)
    with open(Path(org_data_prefix) / consts.PERSON_ITEM_DICT, 'rb') as handle:
        biz_org = pickle.load(handle)

    # unique nodes
    orgs = set(org_inv.keys()) | set(org_biz.keys())
    investors = set(inv_org.keys())
    persons = set(biz_org.keys())

    # Id-ix mappings
    entity_to_ix = {(org, consts.ITEM_TYPE): ix for ix, org in enumerate(orgs)}
    entity_to_ix.update({(inv, consts.USER_TYPE): ix + len(orgs) for ix, inv in enumerate(investors)})
    entity_to_ix.update(
        {(biz, consts.PERSON_TYPE): ix + len(orgs) + len(investors) for ix, biz in enumerate(persons)})
    entity_to_ix[pad_token] = len(entity_to_ix)

    # Ix-id mappings
    ix_to_type = {v: k for k, v in type_to_ix.items()}
    ix_to_relation = {v: k for k, v in relation_to_ix.items()}
    ix_to_entity = {v: k for k, v in entity_to_ix.items()}

    # Export mappings
    org_ix_mapping_prefix = mapping_export_dir + network_type + '_'
    # eg. org_ix_data/dense_type_to_ix.dict
    with open(org_ix_mapping_prefix + consts.TYPE_TO_IX, 'wb') as handle:
        pickle.dump(type_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(org_ix_mapping_prefix + consts.RELATION_TO_IX, 'wb') as handle:
        pickle.dump(relation_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(org_ix_mapping_prefix + consts.ENTITY_TO_IX, 'wb') as handle:
        pickle.dump(entity_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(org_ix_mapping_prefix + consts.IX_TO_TYPE, 'wb') as handle:
        pickle.dump(ix_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(org_ix_mapping_prefix + consts.IX_TO_RELATION, 'wb') as handle:
        pickle.dump(ix_to_relation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(org_ix_mapping_prefix + consts.IX_TO_ENTITY, 'wb') as handle:
        pickle.dump(ix_to_entity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Update the KG
    org_inv_ix = convert_to_ids(entity_to_ix=entity_to_ix, rel_dict=org_inv, start_type=consts.ITEM_TYPE, end_type=consts.USER_TYPE)
    inv_org_ix = convert_to_ids(entity_to_ix, inv_org, consts.USER_TYPE, consts.ITEM_TYPE)
    org_biz_ix = convert_to_ids(entity_to_ix, org_biz, consts.ITEM_TYPE, consts.PERSON_TYPE)
    biz_org_ix = convert_to_ids(entity_to_ix, biz_org, consts.PERSON_TYPE, consts.ITEM_TYPE)

    # export
    # eg. org_ix_data/dense_ix_org_inv.dict
    ix_prefix = export_dir + network_type + '_ix_'
    with open(ix_prefix + consts.ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(org_inv_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(inv_org_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.ITEM_PERSON_DICT, 'wb') as handle:
        pickle.dump(org_biz_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.PERSON_ITEM_DICT, 'wb') as handle:
        pickle.dump(biz_org_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_test_split(network_type, dir):
    dir = str(Path(dir)) + os.sep
    with open(dir + network_type + '_ix_' + consts.USER_ITEM_DICT, 'rb') as handle:
        inv_org = pickle.load(handle)

    # KG and positive
    train_inv_org = {}
    test_inv_org = {}
    train_org_inv = defaultdict(list)
    test_org_inv = defaultdict(list)

    for inv in inv_org:
        pos_orgs = inv_org[inv]
        random.shuffle(pos_orgs)
        cut = int(len(pos_orgs) * 0.8)

        # train
        train_inv_org[inv] = pos_orgs[:cut]
        for org in pos_orgs[:cut]:
            train_org_inv[org].append(inv)

        # test
        test_inv_org[inv] = pos_orgs[cut:]
        for org in pos_orgs[cut:]:
            test_org_inv[org].append(inv)

    # Export
    # eg. org_ix_data/dense_train_ix_org_inv.dict
    with open('%s%s_train_ix_%s' % (dir, network_type, consts.USER_ITEM_DICT), 'wb') as handle:
        pickle.dump(train_inv_org, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_test_ix_%s' % (dir, network_type, consts.USER_ITEM_DICT), 'wb') as handle:
        pickle.dump(test_inv_org, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_train_ix_%s' % (dir, network_type, consts.ITEM_USER_DICT), 'wb') as handle:
        pickle.dump(train_org_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_test_ix_%s' % (dir, network_type, consts.ITEM_USER_DICT), 'wb') as handle:
        pickle.dump(test_org_inv, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_triplets(export_dir):

    # Load the dictionaries from the export directory.
    # These files were saved by your orgs_data_prep() function.
    org_inv_path = os.path.join(export_dir, consts.ITEM_USER_DICT)
    org_biz_path = os.path.join(export_dir, consts.ITEM_PERSON_DICT)

    with open(org_inv_path, 'rb') as f:
        org_inv = pickle.load(f)
    with open(org_biz_path, 'rb') as f:
        org_biz = pickle.load(f)

    triplets = []

    # For each organization and its investors (from org_inv)
    for org, investors in org_inv.items():
        for investor in investors:
            # Here, 'org_inv' denotes the relation: organization → investor.
            triplets.append(f"{org}\torg_inv\t{investor}")

    # For each organization and its business-related personnel (from org_biz)
    for org, biz_persons in org_biz.items():
        for biz in biz_persons:
            # Here, 'org_biz' denotes the relation: organization → business_person.
            triplets.append(f"{org}\torg_biz\t{biz}")

    # (Optional: if you want inverse relations, you could add those too.)

    # Write the triplets to a file
    triplets_file = os.path.join(export_dir, "triplets.txt")
    with open(triplets_file, "w") as f:
        for triplet in triplets:
            f.write(triplet + "\n")

    print(f"Triplets file saved to {triplets_file}")


def main():
    print("Data preparation:")
    args = parse_args()

    network_prefix = args.subnetwork
    # if network_prefix == 'full':
    #     network_prefix = ''

    print("Forming knowledge graph...")
    print(consts.DATASET_DOMAIN)
    DATA_LOC = p('Data', consts.DATASET_DOMAIN)
    PROCESSED_DATA = DATA_LOC / 'processed_data'
    create_directory(PROCESSED_DATA)
    create_directory(PROCESSED_DATA / consts.ITEM_DATA_DIR)

    org_file_loc = Path(args.orgs_file)
    interaction_file_loc = Path(args.interactions_file)
    orgs_data_prep(orgs_file=str(org_file_loc),interactions_file=str(interaction_file_loc),export_dir=str(PROCESSED_DATA / consts.ITEM_DATA_DIR))

    print("Forming network...")
    find_subnetwork(network_type=args.subnetwork, dir=str(PROCESSED_DATA / consts.ITEM_DATA_DIR))

    create_directory(PROCESSED_DATA / consts.ITEM_IX_DATA_DIR)
    create_directory(PROCESSED_DATA / consts.ITEM_IX_MAPPING_DIR)

    ix_mapping(
        network_type=network_prefix,
        import_dir=str(PROCESSED_DATA / consts.ITEM_DATA_DIR),
        export_dir=str(PROCESSED_DATA / consts.ITEM_IX_DATA_DIR),
        mapping_export_dir=str(PROCESSED_DATA / consts.ITEM_IX_MAPPING_DIR)
    )

    train_test_split(
        network_type=network_prefix,
        dir=str(PROCESSED_DATA / consts.ITEM_IX_DATA_DIR)
    )

    generate_triplets(export_dir=str(PROCESSED_DATA / consts.ITEM_DATA_DIR))

    print('Data Preparation is done!')

if __name__ == "__main__":
    main()
