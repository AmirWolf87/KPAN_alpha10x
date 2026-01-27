# Modeling Venture Investors with Knowledge-Graph Path Attention

## Usage Information

This repository implements a path-based recommender system over a heterogeneous knowledge graph.

The pipeline supports multiple ways of sampling the data:

- **Full graph** (`subnetwork = full`)
- **Random sampling** (`subnetwork = rs`)  
  A random 10% sample of entities
- **Dense sampling** (`subnetwork = dense`)  
  Top 10% of entities with the highest degree
- **Custom subnetwork**  
  Defined by the user

The raw data files must be placed in:

Data/orgs_dataset/

Required input files:
- `organizations.csv`
- `train.csv`

---

## How to Run

Because several stages depend on artifacts created earlier, the following order **must be respected**.

### 1. Construct the Knowledge Graph

Run the preprocessing stage to construct indexed knowledge-graph structures:

python data_preparation.py

### 2. Convert Pickle Files (If Required)
To ensure compatibility across Python versions, run:

python convert_pickle_3to2.py

### 3. Initial Recommender Run (Without MF Initialization)

Run the recommender without matrix factorization initialization, popularity baseline, or validation:

python recommender.py \
  --init_mf_embeddings False \
  --np_baseline False \
  --validation False


This step generates path-based training artifacts that are later required by the matrix factorization stage.

### 4. Matrix Factorization Baseline

Run the matrix factorization pipeline:

python matrix_factorization.py


This step generates MF embeddings and produces the file:

processed_train_w_negatives_full_{user_limit}.pkl

### 5. Re-run Recommender With MF Initialization and Baselines

Re-run the recommender using the same arguments as Step 3, but now enabling MF initialization, validation, and the popularity baseline:

python recommender.py \
  --init_mf_embeddings True \
  --np_baseline True \
  --validation True

 ### 6. Popularity Baseline Consistency

Make sure the user_limit value in:

Baseline/popularity.py

matches the user_limit used in recommender.py.

## Knowledge Graph Construction

The knowledge graph is constructed by running data_preparation.py.

### Main arguments:

#### --interactions_file
Path to the CSV file containing user–organization interactions

#### --subnetwork
Subnetwork type to construct (full, rs, dense)

For evaluation, the full subnetwork is used.


## recommender.py Arguments

#### --train
Train the model

#### --validation
Enable validation

#### --eval
Evaluate the model

#### --find_paths
Find paths before training or evaluation

#### --subnetwork
Subnetwork to train and evaluate on

#### --model
Model checkpoint file

#### --model_name
Model type 

#### --path_agg_method
Path aggregation method (attention or weighted_pooling)

#### --load_checkpoint
Load an existing model checkpoint

#### --kg_path_file
File used to save/load train and test paths

#### --user_limit
Maximum number of users for path finding

#### -b
Batch size

#### -e
Number of training epochs

#### --not_in_memory
Use when paths cannot fit entirely in memory

#### --lr, --l2_reg
Learning rate and regularization parameters

#### --nhead, --dropout
Transformer hyperparameters

#### --path_nhead
Number of heads in path aggregation

#### --entity_agg
Entity aggregation method

#### --random_samples
Enable random path sampling

#### --item_to_item
Enable item-to-item inference

## Notes

Some .pkl files are generated implicitly during execution

Re-running earlier stages may overwrite serialized artifacts

Fixed random seeds are used for reproducibility
