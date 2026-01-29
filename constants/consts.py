
DATASET_DOMAIN = 'orgs'
# ORGS
ITEM_DATASET_DIR = 'orgs_dataset/'
ITEM_DATA_DIR = 'orgs_data/'
ITEM_IX_DATA_DIR = 'orgs_data_ix/'
ITEM_IX_MAPPING_DIR = 'orgs_ix_mapping/'
PATH_DATA_DIR = 'path_data/'

PERSON_ITEM_DICT = 'biz_org.dict'
ITEM_PERSON_DICT = 'org_biz.dict'
USER_ITEM_DICT = 'inv_org.dict'
ITEM_USER_DICT = 'org_inv.dict'

# COMMON
TYPE_TO_IX = 'type_to_ix.dict'
RELATION_TO_IX = 'relation_to_ix.dict'
ENTITY_TO_IX = 'entity_to_ix.dict'
IX_TO_TYPE = 'ix_to_type.dict'
IX_TO_RELATION = 'ix_to_relation.dict'
IX_TO_ENTITY = 'ix_to_entity.dict'

PAD_TOKEN = '#PAD_TOKEN'
ITEM_TYPE = 0
USER_TYPE = 1
PERSON_TYPE = 2
PAD_TYPE = 3

ITEM_PERSON_REL = 0
PERSON_ITEM_REL = 1
USER_ITEM_REL = 2
ITEM_USER_REL = 3
UNK_REL = 4
END_REL = 5
PAD_REL = 6

ENTITY_EMB_DIM = 16
TYPE_EMB_DIM = 8
REL_EMB_DIM = 8
HIDDEN_DIM = 256
TAG_SIZE = 32

MAX_PATH_LEN = 4
NEG_SAMPLES_TRAIN = 4
NEG_SAMPLES_TEST = 4


LEN_3_BRANCH = 50
LEN_5_BRANCH_TRAIN = 6
LEN_5_BRANCH_TEST= 10
