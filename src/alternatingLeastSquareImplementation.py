from context import scripts

from scripts.dataLoader import loadData
from scripts.ExplicitModel import RunEALS
from scripts.NN_Model import runNNModel
import time

def comparison(NNModelloss, explicitModelLoss, NNModelTime, explicitModelTime):
  print("The mse test loss of Neural Network model is ",NNModelloss)
  print("The mse test loss of Hand crafted MF model is ",explicitModelLoss)

  print("Time Taken for NCF Model is ", NNModelTime)
  print("Time Taken for NCF Model is ", explicitModelTime)


trainPath = "../data/train.csv"

R_df, traindf = loadData(trainPath)
start = time.time()
print("Collaborative Filtering model using ALS")
test_loss, train_loss = RunEALS(R_df, traindf)
print(test_loss, train_loss)
end = time.time()
timeTakenALS = end - start

start = time.time()
print("Deep Learning based Collaborative Filtering model")
rmse_loss, mse_loss = runNNModel(R_df, traindf)
print(rmse_loss, mse_loss)
end = time.time()
timeTakenNN = end - start

comparison(mse_loss, test_loss, timeTakenNN, timeTakenALS)
# print("Collaborative Filtering model using ALS")
# test_loss, train_loss = RunEALS(R_df, traindf)
# print(test_loss, train_loss)
# print("Deep Learning based Collaborative Filtering model")
# rmse_loss, mse_loss = runNNModel(R_df, traindf)
# print(rmse_loss, mse_loss)


