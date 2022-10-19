import numpy as np
import pandas as pd
from numpy import array
from numpy import argmax
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import random


from scipy import complex128 , float64
from scipy.signal import find_peaks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot
from sklearn import metrics
from sklearn.model import KFold
from sklearn.metrics import confusion_matrix
from Data_load import one_hot_encode
from Model_CNN import model




def train_model(X,Y):
    """ This function takes as input the labelled data and trains the model
    using Keras K-folds tool fo  cross validation
    :parameter : X = I and Q sequences shaped into a 2-dimensional tensor
                 Y = One hot encoded labels """

    oos_y = []   #oos means 'out of sample' , which is assigned to the samples that the model doesn't train on
    oos_pred = []
    fold = 0
    kf = KFold(5, shuffle=True, random_state=42)
    for train, test in kf.split(X, Y):
        fold += 1
        print(f"Fold #{fold}")

        x_train = X[train]
        y_train = Y[train]
        x_test = X[test]
        y_test = Y[test]
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs=80)

        pred = model.predict(x_test)

        oos_y.append(y_test)
        # raw probabilities to chosen class (highest probability)
        pred = np.argmax(pred, axis=1)
        oos_pred.append(pred)

        # Measure this fold's accuracy
        y_compare = np.argmax(y_test, axis=1)  # For accuracy calculation
        score = metrics.accuracy_score(y_compare, pred)
        print(f"Fold score (accuracy): {score}")

    # Build the oos prediction list and calculate the error.
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    oos_y_compare = np.argmax(oos_y, axis=1)  # For accuracy calculation

    score = metrics.accuracy_score(oos_y_compare, oos_pred)
    print(f"Final score (accuracy): {score}")

    # Write the cross-validated prediction
    oos_y = pd.DataFrame(oos_y)
    oos_pred = pd.DataFrame(oos_pred)
    oosDF = pd.concat([oos_y, oos_pred], axis=1)

    return history

# Plotting results : the evolution of loss function and the accuracy

def plot_results(history):
    """ This function makes use of the output of the training function in order to
    plot the evolution of the accuracy and the loss function of the model"""


    #Initializing performance variables

    history.history.keys()
    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    Xc = range(80)



    # Plotting of the evolution of the loss function


    plt.figure(1, figsize=(14, 10))
    plt.plot(Xc, train_loss)
    plt.plot(Xc, validation_loss)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')



    # Plotting of the evolution of the training function


    plt.figure(2, figsize=(14, 10))
    plt.plot(Xc, train_accuracy)
    plt.plot(Xc, validation_accuracy)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')


def plot_confusion_matrix(X_test, Y_test, labels):
    """ This function plots the confusion matrix by predicting the label for
    a sequence of I/Q samples and comparing it with the given label
    :parameter  X_test = 2D array of I/Q samples for which the label(s) will be predicted
                Y_test = The actual labels of the testing I/Q samples
                labels : list of labels (IDs) """


    Y_predict = model.predict(X_test)
    Y_predict = np.argmax(Y_predict, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    confusion_matrix = confusion_matrix(Y_test, Y_predict)
    plt.imshow(cm=confusion_matrix, classes=labels, interpolation='nearest', title='Confusion Matrix')
    plt.ylabel = ('True label')
    plt.xlabel = ('Predicted label')



