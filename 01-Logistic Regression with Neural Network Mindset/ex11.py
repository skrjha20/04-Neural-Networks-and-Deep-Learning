import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

def load_dataset():
    train_data = h5py.File('train_catvnoncat.h5', "r")
    X_train = np.array(train_data["train_set_x"][:])
    y_train = np.array(train_data["train_set_y"][:])
    y_train = y_train.reshape((y_train.shape[0], 1))

    test_data = h5py.File('test_catvnoncat.h5', "r")
    X_test = np.array(test_data["test_set_x"][:])
    y_test = np.array(test_data["test_set_y"][:])
    y_test = y_test.reshape((y_test.shape[0], 1))

    classes = np.array(test_data["list_classes"][:])
    return X_train, y_train, X_test, y_test, classes

if __name__ == "__main__":

    X_train, y_train, X_test, y_test, classes = load_dataset()

    m_train = X_train.shape[0]
    m_test  = X_test.shape[0]
    
    plt.imshow(X_train[25])
    plt.show()