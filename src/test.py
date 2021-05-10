
from context import scripts

from scripts.dataLoader import loadData

testPath = "../data/test.csv"

R_df, traindf = loadData(testPath)
