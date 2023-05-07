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


def thresholding(signal):
    factor = 0.7
    threshold = factor*np.std(signal)
    r_peaks = []
    check = False
    tmp = []
    indices = []
    for i in range(0, len(signal)):
        if signal[i] > threshold:
            check = True
            tmp.append(i)
        else:
            if check:
                indices.append(tmp)
                tmp = []
            check = False
    for i in indices:
        indx = i[int(len(i)/2)]
        # avg = int(np.sum(i) / len(i))
        r_peaks.append(indx)
        signal[indx] = 1
    for i in range(0, len(signal)):
        if signal[i] != 1:
            signal[i] = 0
    return signal,r_peaks

def qr_peak():
    x= []
    for i in range(len(r_peaks)):
        print(i)
        if i !=0:
            x.append(np.argmin(filtered[r_peaks[i-1]:r_peaks[i]]))
    return x


dataset = os.listdir('DataSet')
for patient in dataset:
    records = os.listdir('DataSet\\'+patient)
    for record in records:
        a = wfdb.rdsamp('Dataset'+'/'+patient+'/'+record+"/"+record)
        b = np.asarray(a[0])
        b = b.reshape(15,115200)
        file = b[8]
        nyq = 0.5 * 1000
        low = 10.0/ nyq
        high = 40.0/ nyq
        file = signal.resample(file,11520)
        fc_notch = 50  # Notch frequency (Hz)
        Q = 30
        w0 = fc_notch / (1000 / 2)
        bb, aa = signal.iirnotch(w0, Q)
        ecg_data = signal.filtfilt(bb, aa, file)
        b_s, a_s = signal.butter(2, [low,high], btype='bandpass')
        filtered = signal.filtfilt(b_s, a_s, ecg_data)
        diff = np.gradient(filtered,edge_order=2)
        squared = diff*diff
        win = apply_window(squared, int(0.2*1000))
        win1 = win.copy()
        thresholded,r_peaks = thresholding(win1)
        print(r_peaks)



        fig, axs = plt.subplots(3, 2, figsize=(8, 6))
        axs[0][0].plot(np.arange(0, len(file)), file)
        axs[0][0].set_title(patient+'--'+record)
        axs[0][1].scatter(r_peaks, filtered[r_peaks], marker='x', color='red')
        axs[0][1].plot(np.arange(0, len(filtered)), filtered)
        axs[0][1].set_title('filtered Signal')
        axs[1][0].plot(np.arange(0, len(diff)), diff)
        axs[1][0].set_title('diff signal')
        axs[1][1].plot(np.arange(0,len(squared)), squared)
        axs[1][1].set_title('squared signal')
        axs[2][0].plot(np.arange(0, len(win)), win)
        axs[2][0].set_title('windowed signal')
        axs[2][1].plot(np.arange(0, len(thresholded)), thresholded)
        axs[2][1].set_title('thresholded')
        plt.tight_layout()
        plt.show()