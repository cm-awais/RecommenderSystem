import pandas as pd
import numpy as np

def loadData(path):
  df = pd.read_csv(path, engine='python')

  df_loaded = df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

  ratings = demean(df_loaded)

  train, test = train_test_split(ratings)

  return train, test

def demean(df):
  users_mean=np.array(df.mean(axis=1))
  R_demeaned=df.sub(df.mean(axis=1), axis=0)
  R_demeaned=R_demeaned.fillna(0).values
  return R_demeaned

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
