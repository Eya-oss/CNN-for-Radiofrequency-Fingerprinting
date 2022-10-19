# Importing dependencies



import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import complex128, float64
from numpy import array, argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


distances = ['2ft', '8ft', '14ft', '20ft', '26ft', '32ft', '38ft', '44ft', '50ft', '56ft', '62ft']
path_datafolder = '/home/ajendoubi/PycharmProjects/github_repository_PFE/raw_dataset/'


def extract_id (file_name):
    """ This function allows to extract the device's ID out of the file's name
     :parameter file_name : the name of the file in question """
    index_id_st = file_name.find('312')
    elt_md = file_name[index_id_st:]
    index_id_end = elt_md.find('_')
    device_id = file_name[index_id_st:index_id_st+index_id_end]
    return device_id

def generate_path(path_datafolder, distance):
    return(path_datafolder+distance+'/')


def extract_list_ID (path):
    """ This function the generation of a list of device IDs present in a folder
    :parameter path : math fo the folder containing all the files recorded from the same distance """
    List_ID = []
    for filename in os.listdir(path):
        device_ID = extract_id(filename)
        List_ID.append(device_ID)
    return (list(set(List_ID)))


def read_data(path):
    data = np.fromfile(path, complex128)
    return data


def partition(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]


def Chunck_middle(path):
    chunckSize = 1536000  # (random.randint(1,1024))
    # path = '/content/drive/MyDrive/Dataset_to_train_test/ID5_WiFi_air_X310_3124E4A_8ft_run1.sigmf-data'
    recording = np.fromfile(path, complex128, chunckSize)
    recording = recording[768000:]
    return recording


def Chunck_first(path):
    chunckSize = 768000
    recording = np.fromfile(path, complex128, chunckSize)
    return recording


def Chunck_overlapping(path):
    recording = Chunck_first(path)
    Chunck = np.array([])
    i = 0
    for i in range(768000 // 128):
        Chunck = np.append(Chunck, recording[i:i + 128])
        i += 1
    Chunck = Chunck[:768000]
    return Chunck


def read_overlapping(path, n):
    data = read_data(path)
    file_rec = np.array([])
    windows = list(partition(data, n))
    for idx, num in enumerate(windows):
        file_rec = np.append(file_rec, num)
    return file_rec


def generate_I(recording):
    I = np.real(recording)
    return I


def generate_Q(recording):
    Q = np.imag(recording)
    return Q


def normalize(vals):
    max_value = np.abs(vals).max()
    m = 0
    for m in range(len(vals)):
        vals[m] = vals[m] / max_value
        m += 1
    return vals


def Filter_peaks(vals):
    positives = np.abs(vals)
    indices = find_peaks(positives)
    peak_pos = indices[0]
    i = 0
    for elt in peak_pos:
        np.delete(vals, (elt - i))
        i += 1
    return vals


def reshape_array(I, Q):
    matrix = np.stack((I, Q), axis=1)
    print(matrix.shape)
    a = matrix.shape[0]
    matrix.shape = (a // 256, 256, 2)
    return matrix


def one_hot_encode(Y):
    integer_encoded = Y
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded



#Testing
path = generate_path(path_datafolder, distances[0])
List_ID = extract_list_ID(path)
print(len(List_ID))



#Loading and converting data



def read_convert_data (path):
    """  This function reads I/Q samples from the binary files as numpy arrays,
     generate I and Q separate sequences, filter non-significant peaks ,
     normalize data and save it as numpy arrays in numpy files for later use   """



    for elt in os.listdir(path):
        data_path = generate_path(path, elt)
        data_files = [f for f in os.listdir(data_path) if f.endswith('sigmf-data')]
        for f in data_files:
            record_path = generate_path(data_path, f)
            ID = extract_id(f)
            label = List_ID.index(ID)
            if f.index('run1') != -1:
                recording_1 = read_overlapping(record_path, 128)

                #  Generating I and Q npy files names
                I_file = 'I1' + str(label)
                Q_file = 'Q1' + str(label)

                #  Generating I sequence for the first run , filtering peaks and normalizing it
                I1 = generate_I(recording_1)
                I1 = Filter_peaks(I1)
                I1 = normalize(I1)

                #  Generating Q sequence for the second run, filtering peaks and normalizing it
                Q1 = generate_Q(recording_1)
                Q1 = Filter_peaks(Q1)
                Q1 = normalize(Q1)



                # Saving the numpy arrays into files

                I_npy = os.path.join(path,"npy_files", str(ID), I_file)
                Q_npy = os.path.join(path, "npy_files", str(ID),  Q_file)

                np.save(I1, I_npy)
                np.save(Q1, Q_npy)

            else :
                recording_2 = read_overlapping(record_path, 128)

                #  Generating I and Q npy files names
                I_file = 'I2' + str(label)
                Q_file = 'Q2' + str(label)

                #  Generating I sequence for the first run , filtering peaks and normalizing it
                I2 = generate_I(recording_2)
                I2 = Filter_peaks(I2)
                I2 = normalize(I2)

                #  Generating Q sequence for the second run, filtering peaks and normalizing it
                Q2 = generate_Q(recording_2)
                Q2 = Filter_peaks(Q2)
                Q2 = normalize(Q2)

                # Saving the numpy arrays into files

                I_npy = os.path.join(path, "npy_files", I_file)
                Q_npy = os.path.join(path, "npy_files", Q_file)

                np.save(I2, I_npy)
                np.save(Q2, Q_npy)


def load_data (path):
    """ This function loads pre-processed I and Q data from numpy files and reshape them
    into the input shape to be fed to the neural network
    :parameter path_npy : the path to the numpy files of I and Q"""


    X = np.array([])
    Y = np.array([])

    for elt in os.listdir(path):
        path_per_ID = os.path.join(path , 'elt')
        for elt in os.listdir(path_per_ID):
            data_files = [f for f in os.listdir(path_per_ID) if f.endswith('sigmf-data')]
            for f in data_files:
                file_path = os.path.join(path_per_ID, 'elt')
                ID = extract_id(f)
                label = List_ID.index(ID)
                recording = read_overlapping(file_path, 128)  # 128 is the the length of the overlapping sequences

                # Generating I and Q sequences and pre-process them


                I = generate_I(recording)
                I = Filter_peaks(I) #Filter the non-significant peaks within the I sequence
                I = normalize(I) # Normalize the values of I


                Q = generate_Q(recording)
                Q = Filter_peaks(Q)   # Filter the non-significant peaks within the Q sequence
                Q = normalize(Q)  # Normalize the values of Q


                m = reshape_array(I, Q)
                X = np.concatenate((X,m), axis=0)
                Y = np.concatenate((Y, label * np.ones(len(I))))


                # Configure Y into one hot encoded vector for multiclassification purposes

                Y = one_hot_encode(Y)

    return X, Y














