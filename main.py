import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
def moving_window_integration(signal, window_size, overlap):
    window = np.hanning(window_size)
    step = window_size - overlap
    result = np.zeros_like(signal)
    for i in range(0, len(signal) - window_size, step):
        result[i:i+window_size] += window * signal[i:i+window_size]
    return result

dataset = glob.glob('DataSet/*')
for patient in dataset:
    records = glob.glob(patient + '\\*')
    for record in records:
        file = np.fromfile(record, dtype='byte')
        nyq = 0.5 * 1000
        low = 1.0 / nyq
        high = 40.0 / nyq
        num, den = signal.butter(2, [low, high], btype='bandpass', output='ba')
        plt.plot(np.arange(0, 200), file[slice(200)])
        plt.title(" ECG Signal of " + record[11:])
        plt.show()
        amp = list(file[0:10000])
        filtered = signal.lfilter(num, den, amp)
        plt.plot(np.arange(0, 200), filtered[slice(200)])
        plt.title("Filtered Signal of " + record[11:])
        plt.show()
        diff = np.diff(filtered, n=1)
        plt.plot(np.arange(0, 200), diff[slice(200)])
        plt.title("Derevatived Signal of " + record[11:])
        plt.show()
        sq = diff ** 2
        plt.plot(np.arange(0, 200), sq[slice(200)])
        plt.title("Squaring Signal of " + record[11:])
        plt.show()
        win = moving_window_integration(sq,100,50)
        print(len(win))
        plt.plot(np.arange(0, 200), win[slice(200)])
        plt.title("Windowed Signal of " + record[11:])
        plt.show()
        threshold = 0.7* np.max(win)  # Threshold for QRS detection
        r_signal = np.zeros(win.shape)
        r_signal[win > threshold] = 1
        plt.plot(np.arange(0, 200), r_signal[slice(200)])
        plt.title("Thresholding Signal of " + record[11:])
        plt.show()