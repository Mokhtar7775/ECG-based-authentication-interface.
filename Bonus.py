import os
import numpy as np
from scipy import signal
from scipy import stats
import wfdb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score

dataset = os.listdir('DataSet')
features = []
labels = []
for patient in dataset:
    records = os.listdir('DataSet\\'+patient)
    for record in records:
        a = wfdb.rdrecord('Dataset' + '/' + patient + '/' + record + "/" + record, channels=[11])
        sub1_first_cols = a.p_signal.flatten()
        # print(sub1_first_cols[0:100])
        b, a = signal.butter(4, [1 / (1000 / 2), 40 / (1000 / 2)], btype='bandpass')
        filtered_ecg_signal = signal.filtfilt(b, a, sub1_first_cols)
        #1
        diff_ecg_signal = np.diff(filtered_ecg_signal)
        #2
        abs_diff_ecg_signal = np.abs(diff_ecg_signal)
        #3
        rms_diff_ecg_signal = np.sqrt(np.mean(abs_diff_ecg_signal ** 2))
        #4
        std_diff_ecg_signal = np.std(abs_diff_ecg_signal)
        #5
        var_diff_ecg_signal = np.var(abs_diff_ecg_signal)
        #6
        #Median absolute deviation
        mad_diff_ecg_signal = np.median(np.abs(abs_diff_ecg_signal - np.median(abs_diff_ecg_signal)))
        #7
        skew_diff_ecg_signal = stats.skew(abs_diff_ecg_signal)
        #8
        kurtosis_diff_ecg_signal = stats.kurtosis(abs_diff_ecg_signal)
        feature = [rms_diff_ecg_signal, std_diff_ecg_signal, var_diff_ecg_signal, mad_diff_ecg_signal, skew_diff_ecg_signal, kurtosis_diff_ecg_signal]
        features.append(feature)
        labels.append(patient)

features = np.array(features)
labels = np.array(labels)
print(features.shape)
print(labels.shape)

clf = SVC(kernel='linear')
x_train,x_test,y_train,y_test = train_test_split(features,labels,random_state=0,test_size=0.2)
loaded_model = pickle.load(open('model_fiducial_Bonus.sav', 'rb'))
data = np.load("new_non_fiducial_Bonus/patient104/seg.npy",allow_pickle=True)
result = loaded_model.score(features,labels)
print(result)
# clf.fit(features,labels)
# filename = "model_fiducial_Bonus.sav"
# pickle.dump(clf,open(filename,'wb'))
# y_pred = clf.predict(x_test)
# print("Accuracy : ",accuracy_score(y_test,y_pred))
