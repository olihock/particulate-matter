import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import glob


# Beautify NumPy print
np.set_printoptions(precision=3, suppress=True)

print("TensorFlow version: ", tf.__version__)

# Read the sensor and wind data from file and create a csv handler
path = os.getcwd()
data_file = glob.glob(os.path.join(path, "2022-09_sensor-wind-p1.csv"))
print("data_file: ", data_file)

# Pandas CSV package
column_header = ['Daytime', 'Wind Direction', 'P1']
dataset = pd.read_csv(data_file[0], names=column_header)
print(dataset.tail())

# Split the dataset into a training set and a test set
training_set = dataset.sample(frac=0.7, random_state=0)
print("train_set.size: ", training_set.__len__())
test_set = dataset.drop(training_set.index)
print("test_set.size: ", test_set.__len__())

#sns.pairplot(training_set[['Daytime', 'Wind Direction', 'P1']], diag_kind='kde')

training_features = training_set.copy()
training_labels = training_features.pop('P1')

test_features = test_set.copy()
test_labels = test_features.pop('P1');




