import csv
import pandas as pd
import sklearn


# READ DATASET
train_x = []
train_y = []
test_x = []
train_y = []

dataset = pd.read_csv('datasets/Fish.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)
