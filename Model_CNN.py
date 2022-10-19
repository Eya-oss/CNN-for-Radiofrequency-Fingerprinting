# Importing dependencies



import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import random
from scipy import complex128, float64
from scipy.signal import find_peaks
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot





print('*************************** Building the model***********************************')

# Building model


def model(number_devices):


    model = Sequential()

    # 1st Block
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=(256, 2)))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    # 2nd Block
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    # 3rd Block
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    # 4th Block
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    # flattening for the dense layers
    model.add(Flatten())

    # fully connected layers

    model.add(Dense(256, input_dim=128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(number_devices, activation="softmax"))

    model.summary()
    opt = Adam(learning_rate=0.0001)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        'home/ajendoubi/PycharmProjects/github_repository_PFE/saved_models/',
        monitor='val_loss',

        save_best_only=False,
        save_weights_only=False,

    )

    model.compile(optimizer='adam',
                  # loss='binary_crossentropy',
                  loss='CategoricalCrossentropy',
                  metrics=['accuracy'])


    return model
