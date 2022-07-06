import pandas as pd
import numpy as np


# READ DATASET
train_x = []
train_y = []
test_x = []
train_y = []

dataset = pd.read_csv('datasets/Fish.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# ENCODE DATA
# one hot encode categorical data (fish specie)
fish_species = []

for i in x:
  if (i[0] not in fish_species):
    fish_species.append(i[0])

# one hot encode algorithm
oneHotEncode = {}

for i, fish_specie in enumerate(fish_species):
  if (fish_specie not in oneHotEncode.keys()):
    oneHotEncode[fish_specie] = [0]*len(fish_species)
    oneHotEncode[fish_specie][i] = 1

# replace list item to encode with one hot encoded array
new_x = np.empty((0, len(fish_species) + len(x[0][1:])), dtype=int)

for i, array in enumerate(x):
  if (array[0] in oneHotEncode.keys()):
    new_array = np.append(oneHotEncode[array[0]], array[1:])
    new_x = np.append(new_x, np.array([new_array]), axis=0)

x = new_x