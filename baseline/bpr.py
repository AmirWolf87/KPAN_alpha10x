"""
Bayesian Personalized Ranking

Matrix Factorization model and a variety of classes
implementing different sampling strategies.
"""

import numpy as np
import pandas as pd
from math import exp
import random
import time


class BPRArgs(object):

    def __init__(self,learning_rate=0.001,
                 bias_regularization=1.0,
                 user_regularization=0.025,
                 positive_item_regularization=0.025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors

class BPR(object):

    def __init__(self,D,args):
        """initialise BPR matrix factorization model
        D: number of factors
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors
        self.user_mapping = {}  # User mapping dictionary
        self.item_mapping = {}  # Item mapping dictionary

    def train(self, data, sampler, num_iters, train_new_model=True, max_samples=None, validation_data=None, early_stopping_rounds=None):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        """
        if train_new_model:
            print("training a new model - as stated in arg")
            self.init(data)

        print('initial loss = {0}'.format(self.loss()))
        start_time = time.time()
        # flag_baises_update = 0
        # flag_vectors_update = 0
        best_val_loss = float('inf')
        rounds_since_best_loss = 0
        stop_training = False
        for it in range(num_iters):
            if stop_training:
                break
            for u,i,j in sampler.generate_samples(self.data, max_samples):
                # if flag_baises_update == 0:
                #     flag_baises_update +=1
                #     print("training only biases")
                self.update_factors(u,i,j)
            # print('Biases only : iteration {0}: loss = {1}, time = {2} seconds'.format(it,self.loss(),round(time.time()-start_time, 3)))
            print('iteration {0}: loss = {1}, time = {2} seconds'.format(it,self.loss(),round(time.time()-start_time, 3)))

            if validation_data:
                val_loss = self.validation_loss(validation_data)
                print('validation loss = {0}'.format(val_loss))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_since_best_loss = 0
                else:
                    rounds_since_best_loss += 1

                if early_stopping_rounds and rounds_since_best_loss >= early_stopping_rounds:
                    print(f"Stopping training after {it} , no validation loss improvement")
                    stop_training = True

            if validation_data:
                val_loss = self.validation_loss(validation_data)
                print('Final validation loss = {0}'.format(val_loss))

            start_time = time.time()
        # for iterate in range(num_iters):
        #     for u,i,j in sampler.generate_samples(self.data, max_samples):
        #         if flag_vectors_update == 0:
        #             flag_vectors_update +=1
        #             print("training including vectors")
        #         self.update_factors(u,i,j)
        #     print('iteration {0}: loss = {1}, time = {2} seconds'.format(it,self.loss(),round(time.time()-start_time, 3)))
        #     start_time = time.time()

    def init(self,data):
        self.data = data
        self.num_users,self.num_items = self.data.shape

        self.item_bias = np.zeros(self.num_items)
        np.random.seed(0)
        self.user_factors = np.random.random_sample((self.num_users,self.D))
        self.item_factors = np.random.random_sample((self.num_items,self.D))
        # Create users and items mapping
        num_users = data.shape[0]
        num_items = data.shape[1]
        self.user_mapping = {idx: user_id for idx, user_id in enumerate(range(num_users))}
        # self.user_mapping = (
        #     pd.DataFrame(list(self.user_mapping.items()), columns=['user_idx', 'user'])
        #     .drop_duplicates()
        #     .reset_index(drop=True)
        #     .reset_index()
        #     .rename(columns={'index': 'user_idx'})
        # )
        # self.user_mapping= self.user_mapping.to_numpy()
        self.item_mapping = {idx: item_id for idx, item_id in enumerate(range(num_items))}
        # self.item_mapping = (
        #     pd.DataFrame(list(self.item_mapping.items()), columns=['item_idx', 'item'])
        #     .drop_duplicates()
        #     .reset_index(drop=True)
        #     .reset_index()
        #     .rename(columns={'index': 'item_idx'})
        # )
        self.create_loss_samples(data)
        # self.item_mapping = self.item_mapping.to_numpy()
    def create_loss_samples(self, data):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_users**0.5)

        print('sampling {0} <user,item i,item j> triples...'.format(num_loss_samples))
        sampler = UniformUserUniformItem(True)
        self.loss_samples = [t for t in sampler.generate_samples(data,num_loss_samples)]
    def update_factors(self,u,i,j,update_u=True,update_i=True):
        """apply SGD update"""
        update_j = self.update_negative_item_factors
        u_mapped = self.user_mapping.get(u)
        # u_mapped = np.where(self.user_mapping == u)[0]
        # Use the item mapping to get the corresponding item indices
        i_mapped = self.item_mapping.get(i)
        # i_mapped = np.where(self.item_mapping == u)[0]
        j_mapped = self.item_mapping.get(j)
        # j_mapped = np.where(self.item_mapping == u)[0]
        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u_mapped, :], self.item_factors[i, :] - self.item_factors[j, :])
        # print('x', x)
        try:
            z = 1.0/(1.0+np.exp(x))
        except OverflowError:
            z = np.nextafter(0, 1) # smallest positive double
        # print('z', z)
        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i_mapped, :] - self.item_factors[j_mapped,
                                                  :]) * z - self.user_regularization * self.user_factors[u_mapped, :]
            self.user_factors[u_mapped, :] += self.learning_rate * d
        if update_i:
            d = self.user_factors[u_mapped, :] * z - self.positive_item_regularization * self.item_factors[i_mapped, :]
            self.item_factors[i_mapped, :] += self.learning_rate * d
        if update_j:
            d = -self.user_factors[u_mapped, :] * z - self.negative_item_regularization * self.item_factors[j_mapped, :]
            self.item_factors[j_mapped, :] += self.learning_rate * d
#     def update_biases(self,u,i,j,update_u=True,update_i=True):
#         """apply SGD update"""
#         update_j = self.update_negative_item_factors
#
#         x = self.item_bias[i] - self.item_bias[j] #\
#             # + np.dot(self.user_factors[u,:]*0,self.item_factors[i,:]*0-self.item_factors[j,:]*0)
#         # x = self.item_bias[i] - self.item_bias[j] \
#         #     + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])
#         try:
#             z = 1.0/(1.0+exp(x))
#         except OverflowError:
#             z = np.nextafter(0, 1) # smallest positive double
#
#         # update bias terms
#         if update_i:
#             d = z - self.bias_regularization * self.item_bias[i]
#             self.item_bias[i] += self.learning_rate * d
#         if update_j:
#             d = -z - self.bias_regularization * self.item_bias[j]
#             self.item_bias[j] += self.learning_rate * d
# ## For 1st training phase - train baises while zeroing the vectors
#         # if update_u:
#         #     d = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.user_regularization*self.user_factors[u,:]
#         #     self.user_factors[u,:] += self.learning_rate*d
#         # if update_i:
#         #     d = self.user_factors[u,:]*z - self.positive_item_regularization*self.item_factors[i,:]
#         #     self.item_factors[i,:] += self.learning_rate*d
#         # if update_j:
#         #     d = -self.user_factors[u,:]*z - self.negative_item_regularization*self.item_factors[j,:]
#         #     self.item_factors[j,:] += self.learning_rate*d
    def get_mapping_dataframe(self):
        return pd.DataFrame(list(self.item_mapping.items()), columns=['item_idx', 'item']) , pd.DataFrame(list(self.user_mapping.items()), columns=['user_idx', 'user'])
    def loss(self):
        ranking_loss = 0
        for u,i,j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)

            try:
                ranking_loss += 1.0/(1.0+exp(x))
            except OverflowError:
                ranking_loss = np.nextafter(0, 1) # smallest positive double

        complexity = 0
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2

        return ranking_loss + 0.5*complexity

    def validation_loss(self, validation_data):
        val_data, val_samples = validation_data
        val_loss = 0
        for u, i in val_samples:
            x = self.predict(u, i)
            val_loss += np.log(1 + np.exp(-x))

        return val_loss / len(val_samples)

    def predict(self,u,i):
        return self.item_bias[i] + np.dot(self.user_factors[u],self.item_factors[i])


# sampling strategies

class Sampler(object):

    def __init__(self,sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self,data,max_samples=None):
        self.data = data
        self.num_users,self.num_items = data.shape
        self.max_samples = max_samples

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        j = self.random_item()

        from scipy import sparse
        user_items = sparse.csr_matrix.toarray(sparse.csr_matrix(user_items))

        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0,self.num_items-1)
        return i

    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)

class UniformUserUniformItem(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item
            i = random.choice(self.data[u].indices)
            j = self.sample_negative_item(self.data[u].indices)
            yield u,i,j

class UniformUserUniformItemWithoutReplacement(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(self,data,max_samples)
        # make a local copy of data as we're going to "forget" some entries
        self.local_data = self.data.copy()
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item without replacement if we can
            user_items = self.local_data[u].nonzero()[1]
            if len(user_items) == 0:
                # reset user data if it's all been sampled
                for ix in self.local_data[u].indices:
                    self.local_data[u,ix] = self.data[u,ix]
                user_items = self.local_data[u].nonzero()[1]
            i = random.choice(user_items)
            # forget this item so we don't sample it again for the same user
            self.local_data[u,i] = 0
            j = self.sample_negative_item(user_items)
            yield u,i,j

class UniformPair(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        for _ in range(self.num_samples(self.data.nnz)):
            idx = random.randint(0,self.data.nnz-1)
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u])
            yield u,i,j

class UniformPairWithoutReplacement(Sampler):

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        idxs = list(range(self.data.nnz))
        random.shuffle(idxs)
        self.users,self.items = self.data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in range(self.num_samples(self.data.nnz)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u])
            self.idx += 1
            yield u,i,j

class ExternalSchedule(Sampler):

    def __init__(self,filepath,index_offset=0):
        self.filepath = filepath
        self.index_offset = index_offset

    def generate_samples(self,data,max_samples=None):
        self.init(data,max_samples)
        f = open(self.filepath)
        samples = [map(int,line.strip().split()) for line in f]
        random.shuffle(samples)  # important!
        num_samples = self.num_samples(len(samples))
        for u,i,j in samples[:num_samples]:
            yield u-self.index_offset,i-self.index_offset,j-self.index_offset

if __name__ == '__main__':

    # learn a matrix factorization with BPR like this:

    import sys
    from scipy.io import mmread

    data = mmread(sys.argv[1]).tocsr()


    args = BPRArgs()
    args.learning_rate = 0.3

    num_factors = 16
    model = BPR(num_factors,args)

    sample_negative_items_empirically = True
    sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
    num_iters = 10
    model.train(data,sampler,num_iters)
