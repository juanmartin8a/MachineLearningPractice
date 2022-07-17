# POLYNOMIAL REGRESSION ON FISH DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

np.set_printoptions(threshold=np.inf)


# READ DATASET
train_x = []
train_y = []
test_x = []
train_y = []

dataset = pd.read_csv('datasets/Fish.csv')

r_dataset = dataset.iloc[:, :].values
original_ds = r_dataset

fish_species = []
usingOneHot = False
speciesLength: int

# ENCODE DATA
# one hot encode categorical data (fish specie)
def oneHotEncode(dataset, usingOneHot):
  usingOneHot = True
  for i in dataset:
    if (i[0] not in fish_species):
      fish_species.append(i[0])

  oneHotEncode = {}

  for i, fish_specie in enumerate(fish_species):
    if (fish_specie not in oneHotEncode.keys()):
      oneHotEncode[fish_specie] = [0]*len(fish_species)
      oneHotEncode[fish_specie][i] = 1

  # replace list item to encode with one hot encoded array
  new_x = np.empty((0, len(fish_species) + len(dataset[0][1:])), dtype=int)

  for i, array in enumerate(dataset):
    if (array[0] in oneHotEncode.keys()):
      new_array = np.append(oneHotEncode[array[0]], array[1:])
      new_x = np.append(new_x, np.array([new_array]), axis=0)

  dataset = new_x

  return dataset, usingOneHot

# FEATURE SCALING
def feature_scale_standard_scaler(mean, std, dataset):
  for i, fish in enumerate(dataset[:, speciesLength:]):
    for j, fish_data_tile in enumerate(fish):
      new_fish_data_tile = (fish_data_tile - mean) / std
      dataset[i][j + speciesLength] = new_fish_data_tile
    
  return dataset

def feature_scale_min_max_scaler(min, max, dataset):
  for i, fish in enumerate(dataset[:, speciesLength:]):
    for j, fish_data_tile in enumerate(fish):
      new_fish_data_tile = (fish_data_tile - min)/(max -min)
      dataset[i][j + speciesLength] = new_fish_data_tile

  return dataset

def make_ds_from_x_y(x, y):
  ds = np.empty((0, len(x[0]) + 1), dtype=int)

  for i, array in enumerate(x):
    new_array = np.append(array, y[i])
    ds = np.append(ds, np.array([new_array]), axis=0)

  return ds

# TRAIN MODEL USING POLYNOMIAL FEATURES AND RIDGE REGRESSION
def train(degree, x_train, y_train, x_test, y_test):
  steps = [
      ("poly", PolynomialFeatures(degree = degree)),
      ("model", LinearRegression())
  ]

  pipe = Pipeline(steps)

  pipe.fit(x_train, y_train)

  print("Training accuracy: {}%".format(pipe.score(x_train, y_train) * 100, 3))
  print("Test accuracy: {}%".format(pipe.score(x_test, y_test) * 100, 3))

  return pipe

x_train, x_test, y_train, y_test = train_test_split(r_dataset[:, :-1], r_dataset[:, -1], test_size = 0.2, random_state = 42)

train_ds = make_ds_from_x_y(x_train, y_train)
train_ds, usingOneHot = oneHotEncode(train_ds, usingOneHot)
speciesLength = len(fish_species) if usingOneHot == True else 1

mean = train_ds[:, speciesLength:].mean()
std = train_ds[:, speciesLength:].std()
max = train_ds[:, speciesLength:].max()
min = train_ds[:, speciesLength:].min()

train_ds = feature_scale_standard_scaler(mean, std, train_ds)
test_ds = make_ds_from_x_y(x_test, y_test)
test_ds, usingOneHot = oneHotEncode(test_ds, usingOneHot)
test_ds = feature_scale_standard_scaler(mean, std, test_ds)

x_train = train_ds[:, speciesLength:-1]
y_train = train_ds[:, -1]
x_test = test_ds[:, speciesLength:-1]
y_test = test_ds[:, -1]

pipe = train(2, x_train, y_train, x_test, y_test)