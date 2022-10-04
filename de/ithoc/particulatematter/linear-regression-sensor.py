import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import glob

print("TensorFlow version: ", tf.__version__)

#
# Data Set
#

# Read the sensor and wind data from file and create a csv handler
path = os.getcwd()
data_file = glob.glob(os.path.join(path, "2022-09_sensor-wind-p1.csv"))
print("data_file: ", data_file)

# Pandas CSV package
column_header = ['Daytime', 'Wind_Direction', 'P1']
raw_dataset = pd.read_csv(data_file[0], names=column_header)

# Split the dataset into a training set and a test set
dataset = raw_dataset.copy()
print(dataset.tail())

training_set = dataset.sample(frac=0.7, random_state=0)
print("train_set.size: ", training_set.__len__())
test_set = dataset.drop(training_set.index)
print("test_set.size: ", test_set.__len__())

# Split features and labels for both the training and test set.
training_features = training_set.copy()
training_labels = training_features.pop('P1')

test_features = test_set.copy()
test_labels = test_features.pop('P1')

#
# Univariate Linear Regression (Daytime)
#

daytime_features = np.array(training_features['Daytime'])
print('daytime_features:    ', daytime_features)

# Input shape of this normalisation is a tensor with dimension 1xn
daytime_normalisation = layers.Normalization(input_shape=[1, ], axis=None)
daytime_normalisation.adapt(daytime_features)

daytime_model = tf.keras.Sequential([
    daytime_normalisation,
    layers.Dense(units=1)
])
daytime_model.summary()

daytime_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

daytime_fit = daytime_model.fit(
    training_features['Daytime'],
    training_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

x = tf.linspace(0, 1, 100)
y = daytime_model.predict(x)
print('y: ', y)
