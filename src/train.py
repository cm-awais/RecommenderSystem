from context import scripts

from scripts.dataLoader import loadData
from scripts.ExplicitModel import RunEALS
from scripts.NN_Model import runNNModel

trainPath = "../data/train.csv"

R_df, traindf = loadData(trainPath)

RunEALS(r_df=R_df, traindf=traindf)
runNNModel(R_df, traindf)
