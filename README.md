# KPAN
Python version is 3.6.9, and environment requirements can be installed using `KPAN_requirements.yml`

## Usage Information
To train and evaluate the KPAN model, you have multiple choices for sample the data:
- all data (subnetwork = full) 
- random sampling (subnetwork = rs) - rs contains a random 10% sample of entities
- "smart" sampling (subnetwork = dense) - contains the top 10% entities with highest degree
- create a subnetwork yourself. 

***Data will be uploaded separatly. both files ( Organizations and train) should be copied to Data\orgs_dataset folder***

1st step:  construct the knowledge graph with data_preparation.py 

2nd step: run baseline/matrix_factorization.py for initial MF baseline of the KPAN Model


3rd step: run baseline/train_mf.py for improving and training the MF

4rth step:  path-find, train, and evaluate using recommender.py

*you may also need to run some files separately for creating few needed .pkl files.

### Knowledge Graph Construction
Run data_preparation.py to create relevant dictionaries from the datasets.

Arguments:

`--interactions_file` to specify path to CSV containing user-item interactions

`--subnetwork` to specify data to create knowledge graph from. For our evaluation we use 'full'.


### recommender.py arguments

`--train` to train model, `--validation` to add validation. `--eval` to evaluate

`--find_paths` if you want to find paths before training or evaluating

`--subnetwork` to specify subnetwork training and evaluating on.

`--model` designates the model to train or evaluate from

`--model_name` designates the specific model to train or evaluate: KPAN or KPRN

`--path_agg_methos` designates the way of path aggregation: attention (for cross attention) or weighted pooling

`--load_checkpoint` if you want to load a model checkpoint (weights and parameters) before training

`--kg_path_file` designates the file to save/load train/test paths from

`--user_limit` designates the max number of train/test users to find paths for

`-b` designates model batch size and `-e` number of epochs to train model for

`--not_in_memory` if training on entire dense subnetwork, whose paths cannot fit in memory all at once

`--lr`, `--l2_reg` specify model hyperparameters (learning rate, l2 regularization)

`--nhead`,`--dropout` specify hyperparameters for transformer layer

`--path_nhead` specify number of heads in path aggregation

`--entity_agg` designates the method for aggregate paths

`--random_samples` designates if the paths sampling will be random 

`--item-to-item` True for inference task of item-to-item similarity
