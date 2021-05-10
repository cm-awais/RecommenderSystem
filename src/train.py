from scripts.dataLoader import loadData
import numpy as np
from sklearn.metrics import mean_squared_error

trainPath = "../data/train.csv"

test, train = loadData(trainPath)

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

    def __init__(self, n_iters, n_factors, reg):
        self.reg = reg
        self.n_iters = n_iters
        self.n_factors = n_factors  
        
    def fit(self, train, test, lr):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user, self.n_item = train.shape
        self.user_factors = np.random.random((self.n_user, self.n_factors))
        self.item_factors = np.random.random((self.n_item, self.n_factors))
        
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
    
    def _als_step(self, ratings, solve_vecs, fixed_vecs, lr):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """

        alpha = lr
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.reg * alpha
        b = ratings.dot(fixed_vecs) 
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv) 
        return solve_vecs
    
    def predict(self):
        """predict ratings for every user and item"""
        pred = self.user_factors.dot(self.item_factors.T)
        return pred
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse

als = ExplicitMF(n_iters = 20, n_factors = 50, reg = 0.0001)
als.fit(train, test, lr = 0.0001)
