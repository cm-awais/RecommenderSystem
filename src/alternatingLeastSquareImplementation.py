from context import scripts

from scripts.dataLoader import loadData
from scripts.ExplicitModel import RunEALS
from scripts.NN_Model import runNNModel

trainPath = "../data/train.csv"

R_df, traindf = loadData(trainPath)

print("Collaborative Filtering model using ALS")
test_loss, train_loss = RunEALS(R_df, traindf)
print(test_loss, train_loss)
print("Deep Learning based Collaborative Filtering model")
rmse_loss, mse_loss = runNNModel(R_df, traindf)
print(rmse_loss, mse_loss)
