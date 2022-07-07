import pandas as pd
import numpy as np


# READ DATASET
train_x = []
train_y = []
test_x = []
train_y = []

dataset = pd.read_csv('datasets/Fish.csv')

r_dataset = dataset.iloc[:, :].values

# x = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values


# ENCODE DATA
# one hot encode categorical data (fish specie)
fish_species = []

for i in r_dataset:
  if (i[0] not in fish_species):
    fish_species.append(i[0])

# one hot encode algorithm
oneHotEncode = {}

for i, fish_specie in enumerate(fish_species):
  if (fish_specie not in oneHotEncode.keys()):
    oneHotEncode[fish_specie] = [0]*len(fish_species)
    oneHotEncode[fish_specie][i] = 1

# replace list item to encode with one hot encoded array
new_x = np.empty((0, len(fish_species) + len(r_dataset[0][1:])), dtype=int)

for i, array in enumerate(r_dataset):
  if (array[0] in oneHotEncode.keys()):
    new_array = np.append(oneHotEncode[array[0]], array[1:])
    new_x = np.append(new_x, np.array([new_array]), axis=0)

r_dataset = new_x

# SPLIT DATA
np.random.shuffle(r_dataset)

x_train = r_dataset[:round(len(r_dataset) * 0.8), :-1]
x_test = r_dataset[:round(len(r_dataset) * 0.8), -1]
y_train = r_dataset[round(len(r_dataset) * 0.8):, :-1]
y_test = r_dataset[round(len(r_dataset) * 0.8):, -1]

print(len(x_train))
print(len(y_train))


# FEATURE SCALE