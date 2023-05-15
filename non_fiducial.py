import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from scipy import signal
import wfdb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = os.listdir('DataSet')
data = []
for patient in dataset:
    records = os.listdir('DataSet\\'+patient)
    for record in records:
        a = wfdb.rdrecord('Dataset' + '/' + patient + '/' + record + "/" + record,channels=[11])
        sub1_first_cols = a.p_signal.flatten()
        b, a = signal.butter(2, [1/(0.5*1000),40.0/(0.5*1000) ], btype="bandpass")
        preprocessed = signal.filtfilt(b, a, sub1_first_cols)
        seg1 = np.array(preprocessed)

        ac1 = sm.tsa.acf(seg1, nlags=100000)

        s1 = ac1[0:500]

        dct1 = scipy.fftpack.dct(s1, type=2)
        dct1 = np.array(dct1)
        data.append(dct1[0:len(dct1)])

        fig, axs = plt.subplots(4, 1, figsize=(8, 6))

        axs[0].plot(np.arange(0, len(seg1)), seg1)
        axs[0].set_title(patient + ' seg ' + record)

        axs[1].plot(np.arange(0, len(ac1)), ac1)
        axs[1].set_title('ac')

        axs[2].plot(np.arange(0, len(s1)), s1)
        axs[2].set_title('s')

        axs[3].plot(np.arange(0, len(dct1)), dct1)
        axs[3].set_title('dct')

        plt.tight_layout()
data = np.array(data)
labels = np.array(['p1','p1','p2','p2','p3','p3','p4','p4'])
x_train,x_test,y_train,y_test = train_test_split(data,labels,random_state=104,test_size=0.3)
clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy : ",accuracy_score(y_test,y_pred))