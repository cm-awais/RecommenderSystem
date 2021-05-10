import pandas as pd

def loadData(path):
  df = pd.read_csv(path, engine='python')

  df_loaded = df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

  return df_loaded, df
