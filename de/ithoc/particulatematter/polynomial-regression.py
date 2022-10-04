import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

import os
import glob

path = os.getcwd()
data_file = glob.glob(os.path.join(path, "2022-09_sensor-wind-p1.csv"))

column_header = ['Daytime', 'Wind_Direction', 'P1']
raw_dataset = pd.read_csv(data_file[0], names=column_header)

dataset = raw_dataset.copy()
training_set = dataset.sample(frac=0.7, random_state=0)
test_set = dataset.drop(training_set.index)

training_features = training_set.copy()
training_labels = training_features.pop('P1')

test_features = test_set.copy()
test_labels = test_features.pop('P1')

daytime_features = np.array(training_features['Daytime'])
wind_features = np.array(training_features['Wind_Direction'])

#wind_normalisation = layers.Normalization(input_shape=(2, ), axis=None)
#wind_normalisation.adapt(wind_features)

# Daytime is already between 0 and 1, so no normalisation needed.
# Wind direction is normalised manually here.
wind_features_normalised = (wind_features - wind_features.mean()) / wind_features.std()
wind_labels_normalised = (training_labels - training_labels.mean()) / training_labels.std()

# Expand features by more polynomial features (1, x, x^2).
n = 2
polynomial_features = PolynomialFeatures(n)  # comes from Scikit

daytime_features_expanded = np.expand_dims(daytime_features, axis=1)
daytime_features_expanded = polynomial_features.fit_transform(daytime_features_expanded)

wind_features_normalised_expanded = np.expand_dims(wind_features_normalised, axis=1)
wind_features_normalised_expanded = polynomial_features.fit_transform(wind_features_normalised_expanded)

training_features_expanded = np.column_stack((daytime_features_expanded, wind_features_normalised_expanded))

input_layer = tf.keras.layers.Input(2*(n+1))
dense1_layer = tf.keras.layers.Dense(1)(input_layer)
wind_model = tf.keras.Model(inputs=input_layer, outputs=dense1_layer)

wind_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
)

wind_fit = wind_model.fit(
    training_features_expanded,
    wind_labels_normalised,
    epochs=100
)

x_daytime = tf.linspace(0, 1, 100)
x_daytime_expanded = np.expand_dims(x_daytime, axis=1)
x_daytime_expanded = polynomial_features.fit_transform(x_daytime_expanded)

x_wind_temp = tf.linspace(0, 360, 100)
x_wind_temp_expanded = np.expand_dims(x_wind_temp, axis=1)
x_wind_temp_expanded = polynomial_features.fit_transform(x_wind_temp_expanded)

x_min = (0 - wind_features.mean()) / wind_features.std()
x_max = (360 - wind_features.mean()) / wind_features.std()
x_wind = tf.linspace(x_min, x_max, 100)
x_wind_expanded = np.expand_dims(x_wind, axis=1)
x_wind_expanded = polynomial_features.fit_transform(x_wind_expanded)

x_expanded = np.column_stack((x_daytime_expanded, x_wind_expanded))

y_wind_normalised = wind_model.predict(x_expanded)

# Inversely transform normalised label to absolut label.
y_wind = y_wind_normalised * training_labels.std() + training_labels.mean()

print('y_wind: ', y_wind)
