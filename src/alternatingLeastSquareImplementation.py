from scripts.dataLoader import loadData
import numpy as np
from sklearn.metrics import mean_squared_error


class ExplicitMF:
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix
    
    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm
        
    n_factors : int
        number of latent factors to use in matrix 
        factorization model, some machine-learning libraries
        denote this as rank
        
    reg : float
        regularization term for item/user latent factors,
        since lambda is a keyword in python we use reg instead
    """

    def __init__(self, n_iters, n_factors, reg, users_mean):
        self.reg = reg
        self.n_iters = n_iters
        self.n_factors = n_factors  
        self.user_means = users_mean
        
    def fit(self, train, test, lr):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user = train.shape[0]
        self.n_item = train.shape[1]
        
        self.user_factors = np.random.rand(self.n_user, self.n_factors)
        self.item_factors = np.random.rand(self.n_item, self.n_factors)
        
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        self.test_mse_record  = []
        self.train_mse_record = []   
        for _ in range(self.n_iters):
            self.user_factors = self._als_step(train, self.user_factors, self.item_factors, lr)
            self.item_factors = self._als_step(train.T, self.item_factors, self.user_factors, lr) 

            # self.user_factors = s_f
            # self.item_factors = i_f
            predictions = self.predict()
            test_mse = self.compute_mse(test, predictions)
            train_mse = self.compute_mse(train, predictions)
            print("Iteration ",_," test_mse: ",test_mse ," train_mse: ",train_mse)
            self.test_mse_record.append(test_mse)
            self.train_mse_record.append(train_mse)
        
        return self    
    
    def _als_step(self, ratings, solve_vecs, fixed_vecs):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """

#         alpha = lr
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.reg #* alpha
        b = ratings.dot(fixed_vecs) 
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv) 
        return solve_vecs
    
    def predict(self):
        """predict ratings for every user and item"""
        pred = self.user_factors.dot(self.item_factors.T) + self.user_means.reshape(-1,1)
        return pred
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse


trainPath = "../data/train.csv"

data_df = loadData(trainPath)

users_mean=np.array(data_df.mean(axis=1))
R_demeaned=R_df.sub(data_df.mean(axis=1), axis=0)
R_demeaned=R_demeaned.fillna(0).values

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

test, train = train_test_split(R_demeaned)

als = ExplicitMF(n_iters = 20, n_factors = 50, reg = 0.0001, users_mean= users_mean)
als.fit(train, test)
