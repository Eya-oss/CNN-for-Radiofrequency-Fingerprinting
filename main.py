
import numpy as np
from pathlib import Path

from Data_load import generate_path, extract_id, extract_list_ID, read_convert_data, load_data
from Model_CNN import model
from Train_and_Evaluate import train_model, plot_results, plot_confusion_matrix

distances = ['2ft', '8ft', '14ft', '20ft', '26ft', '32ft', '38ft', '44ft', '50ft', '56ft', '62ft']
path_datafolder = '/home/ajendoubi/PycharmProjects/github_repository_PFE/raw_dataset/'

path = generate_path(path_datafolder, distances[0])
List_ID = extract_list_ID(path)
print(len(List_ID))
number_devices = len(List_ID)


# Initialize the model
model = model(number_devices)

#Initiate the labeled input data
X, Y = load_data(path)

#Train the model
history = train_model(X, Y)

#Plot the results of the performance of the model

plot_results(history)
labels = [i for i in range(number_devices)]
X_test = X[:(len(X)//5)]
Y_test = Y[:(len(Y)//2):]
plot_confusion_matrix(X_test, Y_test, labels)






