import os
import numpy as np
from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt
import statsmodels.api as sm
import glob
import scipy
from scipy import signal
import wfdb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


def non_fiducial_main():
    dataset = os.listdir('DataSet')
    data = []
    labels = []
    for patient in dataset:
        records = os.listdir('DataSet\\'+patient)
        for record in records:
            labels.append(patient)
            a = wfdb.rdrecord('Dataset' + '/' + patient + '/' + record + "/" + record,channels=[11])
            sub1_first_cols = a.p_signal.flatten()
            b, a = signal.butter(2, [1/(0.51000),40.0/(0.51000) ], btype="bandpass")
            preprocessed = signal.filtfilt(b, a, sub1_first_cols)
            seg1 = np.array(preprocessed)
            ac1 = sm.tsa.acf(seg1, nlags=100000)
            s1 = ac1[0:500]
            dct1 = scipy.fftpack.dct(s1, type=2)
            dct1 = np.array(dct1)
            data.append(dct1[0:len(dct1)])
    return dct1