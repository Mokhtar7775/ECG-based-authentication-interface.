import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wfdb as wfdb


def apply_window(ecg_signal,size):
    window_size =size
    window = np.ones(window_size) / window_size
    smooth_ecg_signal = np.convolve(ecg_signal, window, mode='same')
    return smooth_ecg_signal


def thresholding(integrated_signal):
    factor = 0.8
    threshold = factor*np.max(integrated_signal)
    r_peaks = []
    for i in range(1, len(integrated_signal) - 1):
        if integrated_signal[i] > threshold and integrated_signal[i] > integrated_signal[i - 1] and integrated_signal[i] > integrated_signal[i + 1]:
            r_peaks.append(i)
    return r_peaks

dataset = os.listdir('DataSet')
for patient in dataset:
    records = os.listdir('DataSet\\'+patient)
    for record in records:
        record1 = wfdb.rdrecord('Dataset'+'/'+patient+'/'+record+"/"+record,channels=[1])
        file = record1.p_signal.flatten()

        fs = record1.fs
        f1 = 15/fs
        f2 = 30/fs
        filter_order = 1
        b, a = signal.butter(filter_order, [f1,f2], btype='bandpass')
        filtered = signal.filtfilt(b, a, file)
        diff = np.diff(filtered)
        squared = diff**2
        win = int(0.15*fs)
        integrated = np.convolve(squared,np.ones(win),mode='same')
        r_peaks = thresholding(integrated)
        print(r_peaks)

        fig, axs = plt.subplots(5, 1, figsize=(8, 6))
        axs[0].plot(np.arange(0, len(file)), file)
        axs[0].set_title(patient+'--'+record)
        axs[0].scatter(r_peaks, file[r_peaks], marker='+', color='red')
        axs[1].plot(np.arange(0, len(filtered)), filtered)
        axs[1].set_title('filtered Signal')
        axs[2].plot(np.arange(0, len(diff)), diff)
        axs[2].set_title('diff signal')
        axs[3].plot(np.arange(0,len(squared)), squared)
        axs[3].set_title('squared signal')
        axs[4].plot(np.arange(0, len(integrated)), integrated)
        axs[4].set_title('windowed signal')
        plt.tight_layout()
        plt.show()