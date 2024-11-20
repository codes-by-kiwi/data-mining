import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import multiprocessing
import random
import math
import os
import numpy as np

path = "u.data"
df = pd.read_table(path, sep="\t", names=["UserID", "MovieID", "Rating", "Timestamp"])
df = df.pivot_table(index='UserID', columns='MovieID', values='Rating')
df = df.fillna(0)


def similarity_matrix(matrix, k=5, axis=0):
    data = matrix.to_numpy()
    similarity_dict = {}

    num_users = data.shape[0] # no of users in the matrix/rows
    num_prods = data.shape[1] # no of products in the matrix/cols
    u2u_sim = np.zeros((num_users,num_users))
    p2p_sim = np.zeros((num_prods,num_prods))

    if axis==0: 
        if k>num_users:
            raise ValueError

    #iterate through the data structure to calculate dot product. i and j are row indices representing users
        for i in range(num_users):
            for j in range(i+1,num_users):
                dotprod = np.dot(data[i], data[j]) 
                denominator=np.sqrt(np.sum(data[i]**2))*np.sqrt(np.sum(data[j]**2))
                if denominator!=0:
                    u2u_sim[i,j]= dotprod/denominator
                    u2u_sim[j,i] = dotprod/denominator
 
    #sort it for the top k users with highest cosine similarity 

        for useridx in range(num_users):
            indices_topk=np.argsort(u2u_sim[useridx])[-k:][::-1] # np.argsort(u2u_sim[i]) returns indices in ascending order. you are taking the last k indices from this then reversing them. 
            similarity_dict[useridx]=[(otheruser,u2u_sim[useridx][otheruser]) for otheruser in indices_topk] # otheruser is one of the top k indices
    #result of the func is tuples of the form (otheruser,similarity)
    if axis==1:
        if k>num_prods:
            raise ValueError

        for i in range(num_prods):
            for j in range(i+1,num_prods):
                dotprod = np.dot(data[:,i],data[:,j])
                denominator = np.sqrt(np.sum(data[:,i]**2))*np.sqrt(np.sum(data[:,j]**2))
                if denominator!=0:
                    p2p_sim[i,j]= dotprod/denominator
                    p2p_sim[j,i]=dotprod/denominator
    
        for prodidx in range(num_prods):
            indices_topk=np.argsort(p2p_sim[prodidx])[-k:][::-1]
            similarity_dict[prodidx]=[(otherprod,p2p_sim[prodidx][otherprod]) for otherprod in indices_topk]
    
    return similarity_dict

# 2. COLLABORATIVE FILTERING

def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    numerator=0
    denominator=0

    for user,cossim in user_similarity[user_id]:
        rating = user_item_matrix.loc[user][movie_id]
        if rating==0 or np.isnan(rating):
            continue
        else:
            numerator += cossim * rating 
            denominator += cossim
    
    if denominator==0:
        return np.nan

    predicted_rating = numerator/denominator
    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):

    numerator = 0  
    denominator = 0  

    numerator = sum([(cossim*user_item_matrix.loc[user_id,kmovie_id]) for (kmovie_id,cossim) in item_similarity[movie_id]]) 
    denominator = sum(cossim for (kmovie_id,cossim) in item_similarity[movie_id])
    if denominator==0:
        return np.nan
    predicted_rating = numerator/denominator
    return predicted_rating

class UVDecomposition:
    def __init__(self, path, num_factors, num_iters, seed, save=False, feature_matrices_path="", random=False, data=None):
        
        self.num_factors = num_factors
        self.num_iters = num_iters
        self.seed = seed
        self.save = save
        self.feature_matrices_path = feature_matrices_path
        
         # Read the data
        if data is not None:
            self.data = pd.DataFrame(data, columns=["UserID", "MovieID", "Rating"])
        else:
             # Read data from file if no data is provided#             
            self.data = pd.read_table(path, sep='\t', names=["UserID", "MovieID", "Rating", "Timestamp"])
        
        self.user_item_matrix = self.data.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
        self.R = self.user_item_matrix.to_numpy()
    
        np.random.seed(self.seed)
        
         # Initialize the user feature matrix U and the item feature matrix V
        num_users, num_items = self.R.shape
        self.U = np.random.normal(scale=1./self.num_factors, size=(num_users, self.num_factors))
        self.V = np.random.normal(scale=1./self.num_factors, size=(num_items, self.num_factors))

    def update_U(self, index, data_train, U, V):
        user_index, feature_update = index
        M = []
        # Get rating data related to user R
        for row in data_train:
            if row[0] == (user_index + 1):
                M.append(row)
        M = np.array(M)
        sum_1, sum_2 = 0, 0
        for row in range(M.shape[0]):
            m = int(M[row, 1]) - 1
            pred = np.dot(U[user_index, :], V[:, m]) - U[user_index, feature_update] * V[feature_update, m]
            sum_1 += V[feature_update, m] * (M[row, 2] - pred)
            sum_2 += V[feature_update, m] ** 2
        if sum_2 == 0:
            sum_2 = 0.001  # Prevent the denominator from being 0    
        U[user_index, feature_update] = sum_1 / sum_2
        return U

    def update_V(self, index, data_train, U, V):
        feature_update, movie_index = index
        M = []
        # Get rating data related to user R
        for row in data_train:
            if row[1] == (movie_index + 1):
                M.append(row)
        # Convert the list to a numpy array  
        M = np.array(M)
        sum_1, sum_2 = 0, 0
        for row in range(M.shape[0]):
            m = int(M[row, 0]) - 1
            pred = np.dot(U[m, :], V[:, movie_index]) - U[m, feature_update] * V[feature_update, movie_index]
            sum_1 += U[m, feature_update] * (M[row, 2] - pred)
            sum_2 += U[m, feature_update] ** 2

        if sum_2 == 0:
            sum_2 = 0.001  # Prevent the denominator from being 0
            
        V[feature_update, movie_index] = sum_1 / sum_2
        return V


    def _train_iteration(self, U, V, data_train):
        u_index = list(np.ndindex(U.shape))
        v_index = list(np.ndindex(V.shape))
        
        random.shuffle(u_index)
        random.shuffle(v_index)
        
        while len(u_index) > 0 or len(v_index) > 0:
            if len(u_index) > 0:
                u = u_index.pop()
                U = self.update_U(u, data_train, U, V)

            if len(v_index) > 0:
                v = v_index.pop()
                V = self.update_V(v, data_train, U, V)
                
        return U, V

    def train(self):
        data_train = self.data[['UserID', 'MovieID', 'Rating']].to_numpy()
        for i in range(self.num_iters):
            print(f"Iteration {i+1}/{self.num_iters}")
            self.U, self.V = self._train_iteration(self.U, self.V.T, data_train)
            self.V = self.V.T

        if self.save:
            np.savez(self.feature_matrices_path, U=self.U, V=self.V)

    def predict(self, user, item):
       
        user_index = user - 1  
        item_index = item - 1 
        
        prediction = np.dot(self.U[user_index, :], self.V[item_index, :].T)
        return prediction

if __name__ == "__main__":
    # This main section is intended for testing your implemented functions:

    # You can use this section for testing the similarity_matrix function: 
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    print(user_similarity_matrix.get(3,[]))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(10,[]))

    # You can use this section for testing the user_based_cf and the item_based_cf functions:
    # Return the predicted ratings assigned by user 13 to movie 100:
    user_id = 13  
    movie_id = 100  

    u_predicted_rating = user_based_cf(user_id, movie_id, user_similarity_matrix, user_item_matrix = df, k=5)
    print(f"predicted user {user_id} rating for movie {movie_id}, according to user-based collaborative filtering is: {u_predicted_rating:.2f}")

    i_predicted_rating = item_based_cf(user_id, movie_id, item_similarity_matrix, user_item_matrix = df, k=5)
    print(f"predicted user {user_id} rating for movie {movie_id}, according to item-based collaborative filtering is: {i_predicted_rating:.2f}")


    # You can use this section for testing the update_V function within the UVDecomposition class:
    num_factors = 10
    num_iters = 5
    seed = 42
    save = True

    uv_model = UVDecomposition(path, num_factors, num_iters, seed, save)
    uv_model.train()

    # Return the predicted ratings assigned by user 13 to movie 100:

    user_id = 13
    movie_id = 100
    predicted_rating = uv_model.predict(user_id, movie_id)
    print(f"predicting User {user_id} rating for movie {movie_id}, according to the UV-decomposition method is: {predicted_rating:.2f}")


    
