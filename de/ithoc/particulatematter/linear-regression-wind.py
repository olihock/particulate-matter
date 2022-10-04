import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

import os
import glob

#
# Data Set
#

# Read the sensor and wind data from file and create a csv handler
path = os.getcwd()
data_file = glob.glob(os.path.join(path, "2022-09_sensor-wind-p1.csv"))

# Pandas CSV package
column_header = ['Daytime', 'Wind_Direction', 'P1']
raw_dataset = pd.read_csv(data_file[0], names=column_header)

# Split the dataset into a training set and a test set
dataset = raw_dataset.copy()

training_set = dataset.sample(frac=0.7, random_state=0)
test_set = dataset.drop(training_set.index)

# Split features and labels for both the training and test set.
training_features = training_set.copy()
training_labels = training_features.pop('P1')

test_features = test_set.copy()
test_labels = test_features.pop('P1')

# Univariate Linear Regression (Wind)

wind_features = np.array(training_features['Wind_Direction'])

scaler = MinMaxScaler()
scaler.fit(wind_features.reshape(-1, 1))
wind_features_scaled = scaler.transform(wind_features.reshape(-1, 1))
wind_features_inverse = scaler.inverse_transform(wind_features_scaled)

wind_normalisation = layers.Normalization(input_shape=[1, ], axis=None)
wind_normalisation.adapt(wind_features)

wind_model = tf.keras.Sequential()
wind_model.add(wind_normalisation)
wind_model.add(layers.Dense(units=1))

wind_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

wind_fit = wind_model.fit(
    training_features['Wind_Direction'],
    training_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

x_wind = tf.linspace(0, 360, 100)
y_wind = wind_model.predict(x_wind)
print('y_wind: ', y_wind)
