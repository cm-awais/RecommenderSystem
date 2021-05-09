import pandas as pd

def loadData(path):
  df = pd.read_csv(path, engine='python')

  return df

